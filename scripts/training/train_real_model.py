import os
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# === FIX 1: Allow loading of truncated (incomplete) images ===
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === CONFIGURATION ===
IMAGE_DIR = 'data/images'           
CSV_PATH = 'data/nih_labels.csv'
MODEL_SAVE_PATH = 'models/chexnet_real_finetuned.pth'
BATCH_SIZE = 8  # Keep 8 for 6GB VRAM
LEARNING_RATE = 0.01 
EPOCHS = 5

# 14 Disease Labels
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
        
        if 'fold' in self.df.columns:
            self.df = self.df[self.df['fold'] == split]

        # Verify files exist (Filters out missing files)
        initial_len = len(self.df)
        print(f"🔍 Verifying image files for {split}...")
        
        # We verify checking if file path exists
        self.df['exists'] = self.df['Image Index'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.df = self.df[self.df['exists']]
        
        removed = initial_len - len(self.df)
        print(f"✅ Verified {len(self.df)} images for {split} (Removed {removed} missing files)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = os.path.join(self.root_dir, img_name)
        
        # Get labels
        labels = torch.tensor(row[LABELS].values.astype(float), dtype=torch.float32)

        try:
            # Convert to RGB to ensure 3 channels
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ CORRUPT FILE {img_name}: {e}")
            # Return a black image if file is completely broken
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, labels

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📂 Loading Dataset...")
    train_dataset = NIHChestXrayDataset(CSV_PATH, IMAGE_DIR, transform=transform, split='train')
    val_dataset = NIHChestXrayDataset(CSV_PATH, IMAGE_DIR, transform=transform, split='test')

    # === FIX 2: num_workers=0 prevents Windows freeze/crash ===
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("🧠 Initializing Model...")
    model = densenet121(pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(LABELS))
    model.to(device)

    # Optimizer: SGD with Momentum (Standard for CheXNet)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Scheduler: Reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

    # TensorBoard
    writer = SummaryWriter('runs/chexnet_experiment_1')

    print("🔥 Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        
        # Progress Bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update progress bar
            loop.set_postfix(loss=loss.item())
            
            # Log to TensorBoard every 10 batches
            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Training Loss', loss.item(), global_step)

        # === Validation Phase ===
        print("\n🔍 Running Validation...")
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        
        print(f"✨ Epoch {epoch+1} Complete. Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Model saved to {MODEL_SAVE_PATH}")

    writer.close()

if __name__ == '__main__':
    train_model()
