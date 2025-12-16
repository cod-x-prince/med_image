import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import pandas as pd
import os
import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

# === CONFIGURATION ===
MODELS_DIR = 'models'
IMAGE_DIR = 'data/images'  # Make sure this matches your flattened folder
CSV_PATH = 'data/nih_labels.csv'
TEST_SIZE = 10000          # Number of images to check
BATCH_SIZE = 32            # Process 32 images at a time

# Handle corrupt images
ImageFile.LOAD_TRUNCATED_IMAGES = True

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# === 1. DATASET CLASS ===
class NIHBenchmarkDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, limit=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Use the official 'test' fold if possible, or sample random
        if 'fold' in self.df.columns:
            self.df = self.df[self.df['fold'] == 'test']
        
        # Limit to specific number of images for speed
        if limit:
            self.df = self.df.sample(n=min(limit, len(self.df)), random_state=42)
        
        # Verify existence
        self.df['exists'] = self.df['Image Index'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.df = self.df[self.df['exists']]
        print(f"📊 Benchmark Set: Loaded {len(self.df)} validated images.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = os.path.join(self.root_dir, img_name)
        
        # Get labels
        labels = torch.tensor(row[DISEASE_LABELS].values.astype(float), dtype=torch.float32)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, labels

# === 2. ROBUST MODEL LOADER ===
def load_model_safe(model_path, device):
    try:
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, len(DISEASE_LABELS))
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Unwrap state_dict if needed
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Fix Key Names
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('densenet121.', '')
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(model_path)}: {e}")
        return None

# === 3. MAIN BENCHMARK LOOP ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Benchmark on {device}...")

    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = NIHBenchmarkDataset(CSV_PATH, IMAGE_DIR, transform=transform, limit=TEST_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Find Models
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    print(f"🔎 Found {len(model_files)} models to test.")

    results = {}

    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\n🧠 Testing Model: {model_name}")
        
        model = load_model_safe(model_path, device)
        if not model: continue

        all_preds = []
        all_labels = []

        # Inference Loop
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Running Inference"):
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                all_preds.append(probs)
                all_labels.append(labels.numpy())

        # Aggregate
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate AUC per disease
        try:
            aucs = []
            for i, disease in enumerate(DISEASE_LABELS):
                # Only calculate if the disease is present in the test set
                if np.unique(all_labels[:, i]).size > 1:
                    score = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    aucs.append(score)
            
            mean_auc = np.mean(aucs)
            results[model_name] = mean_auc
            print(f"📈 Average AUC: {mean_auc:.4f}")
            
        except Exception as e:
            print(f"⚠️ Metric Calculation Error: {e}")

    # === 4. THE LEADERBOARD ===
    print("\n" + "="*50)
    print("🏆 FINAL MODEL LEADERBOARD (Average AUC)")
    print("="*50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, score) in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank}. {name:<30} : {score:.4f}")
    
    print("="*50)
    print("Note: >0.80 is Professional Grade. >0.70 is Good.")

if __name__ == '__main__':
    main()