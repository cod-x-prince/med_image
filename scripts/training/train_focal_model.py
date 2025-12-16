import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import datetime

# === 1. ADVANCED CONFIGURATION ===
# Hardware Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 # Critical for Windows stability
PIN_MEMORY = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'images')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'nih_labels.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'training_output')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'chexnet_focal.pth')

# Training Hyperparameters
IMG_SIZE = 224      
BATCH_SIZE = 16     
LEARNING_RATE = 1e-4
EPOCHS = 10
PATIENCE = 3        

# Focal Loss Settings
ALPHA = 1           
GAMMA = 2           

# Labels
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# === 2. LOGGING SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure Logging (Fixed for Windows)
log_filename = os.path.join(OUTPUT_DIR, f'training_focal_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # Force UTF-8 for file
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === 3. FOCAL LOSS CLASS ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

# === 4. DATASET CLASS ===
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split='train'):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Filter missing files
        self.df['exists'] = self.df['Image Index'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.df = self.df[self.df['exists']]
        
        # Consistent Split
        train_df = self.df.sample(frac=0.8, random_state=42)
        test_df = self.df.drop(train_df.index)
        
        self.df = train_df if split == 'train' else test_df
        logger.info(f"[DATA] {split.upper()} Set: {len(self.df)} images ready.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Image Index'])
        
        label_vec = torch.zeros(len(LABELS), dtype=torch.float32)
        if 'Finding Labels' in row:
            labels_str = row['Finding Labels']
            present = labels_str.split('|')
            for i, label in enumerate(LABELS):
                if label in present:
                    label_vec[i] = 1.0
        else:
            for i, label in enumerate(LABELS):
                if label in row and row[label] == 1:
                    label_vec[i] = 1.0

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Corrupt image skipped: {img_path}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, label_vec

# === 5. ADVANCED PLOTTING ===
def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.style.use('ggplot') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Analysis (Focal Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['lr'], 'g-', marker='o', label='Learning Rate')
    ax2.set_title('Learning Rate Decay')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('LR')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_dashboard.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"[PLOT] Dashboard updated: {save_path}")

# === 6. TRAINING LOOP ===
def train_model():
    logger.info(f"[START] Starting FOCAL Training on {DEVICE}")
    logger.info(f"[CONFIG] Batch={BATCH_SIZE}, LR={LEARNING_RATE}, Gamma={GAMMA}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = NIHChestXrayDataset(CSV_PATH, IMAGE_DIR, transform=train_transform, split='train')
    val_dataset = NIHChestXrayDataset(CSV_PATH, IMAGE_DIR, transform=val_transform, split='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = densenet121(weights='DEFAULT')
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = FocalLoss(alpha=ALPHA, gamma=GAMMA)
    
    # FIX: Removed verbose=True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
        
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_loop.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        
        logger.info(f"[EPOCH {epoch+1}] Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr}")
        
        plot_metrics(history)
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"[SAVE] New Best Model Saved! (Val Loss: {best_val_loss:.4f})")

    logger.info("[DONE] Training Complete.")

if __name__ == '__main__':
    train_model()