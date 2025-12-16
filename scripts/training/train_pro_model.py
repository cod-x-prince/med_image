import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121
from tqdm import tqdm

# === CONFIGURATION ===
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === 1. FIXED LOCAL PATHS (Windows Compatible) ===
CSV_PATH = 'data/nih_labels.csv'   # Matches your local CSV filename
IMAGE_DIR = 'data/images'          # Matches your local flattened image folder

MODEL_SAVE_PATH = 'models/chexnet_pro_v1.pth'

# === PRO SETTINGS ===
IMG_SIZE = 320 
BATCH_SIZE = 4    # Reduced to 4 because 320x320 images use A LOT of VRAM (6GB limit)
LEARNING_RATE = 1e-4
EPOCHS = 5

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split='train'):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Clean dataset: Keep only files that actually exist on disk
        # This skips the missing files automatically
        self.df['exists'] = self.df['Image Index'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.df = self.df[self.df['exists']]

        # Split Data (80/20 split based on random sample)
        if split == 'train':
            self.df = self.df.sample(frac=0.8, random_state=42)
        else:
            self.df = self.df.drop(self.df.sample(frac=0.8, random_state=42).index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Image Index'])
        
        # Parse 14 Labels
        label_vec = torch.zeros(len(LABELS), dtype=torch.float32)
        # Local CSV usually has binary columns 0/1, let's check format
        # If using original NIH labels string:
        if 'Finding Labels' in row:
             for i, label in enumerate(LABELS):
                if label in row['Finding Labels']:
                    label_vec[i] = 1.0
        else:
            # If using your processed nih_labels.csv with columns
            label_vec = torch.tensor(row[LABELS].values.astype(float), dtype=torch.float32)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, label_vec

def compute_pos_weights(dataset, device):
    """
    Calculates how rare each disease is to balance the training.
    """
    print("⚖️ Calculating Class Weights...")
    df = dataset.df
    total = len(df)
    
    pos_weights = []
    
    # Handle both string format and column format
    if 'Finding Labels' in df.columns:
        # String parsing method
        counts = {label: 0 for label in LABELS}
        for labels in df['Finding Labels']:
            for l in LABELS:
                if l in labels:
                    counts[l] += 1
        for l in LABELS:
            pos = counts[l]
            neg = total - pos
            weight = neg / (pos + 1e-5)
            pos_weights.append(weight)
    else:
        # Column method (faster)
        for l in LABELS:
            pos = df[l].sum()
            neg = total - pos
            weight = neg / (pos + 1e-5)
            pos_weights.append(weight)
        
    print(f"✅ Rare Class Boost: Hernia Weight={pos_weights[-1]:.1f}x")
    return torch.tensor(pos_weights, dtype=torch.float32).to(device)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training PRO Model on: {device} | Resolution: {IMG_SIZE}x{IMG_SIZE}")

    # === UPGRADE 2: HEAVY AUGMENTATION ===
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),      
        transforms.RandomRotation(15),          
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📂 Loading Datasets...")
    train_dataset = NIHChestXrayDataset(CSV_PATH, IMAGE_DIR, transform=train_transform, split='train')
    val_dataset = NIHChestXrayDataset(CSV_PATH, IMAGE_DIR, transform=val_transform, split='test')

    # === FIX FOR WINDOWS: num_workers=0 ===
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # === UPGRADE 3: WEIGHTED LOSS FUNCTION ===
    pos_weights = compute_pos_weights(train_dataset, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    print("🧠 Initializing Model...")
    model = densenet121(weights='DEFAULT') # Updated syntax
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    model.to(device)

    # Use AdamW 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    print("🔥 Starting PRO Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        # TQDM progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
        
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_loop.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"✨ Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        # Create 'models' directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Saved Best Model: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()