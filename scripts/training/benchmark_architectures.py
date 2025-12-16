import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# === 1. CONFIGURATION ===
# We verify these files exist before running
# === CONFIGURATION ===
MODEL_DENSE_PATH = 'models/chexnet_model.pth'       # The 0.89 Winner (224px)
MODEL_EFF_PATH = 'models/efficientnet_b4_512px.pth'

IMAGE_DIR = 'data/images'
CSV_PATH = 'data/nih_labels.csv'
TEST_SAMPLES = 1000  # Number of patients to test
BATCH_SIZE = 32

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# === 2. DATASET CLASS ===
class NIHTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, limit=1000):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Use 'test' fold if available to ensure we haven't trained on these
        if 'fold' in self.df.columns:
            self.df = self.df[self.df['fold'] == 'test']
        
        # Verify files exist
        self.df['exists'] = self.df['Image Index'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.df = self.df[self.df['exists']]
        
        # Limit to specific number
        if len(self.df) > limit:
            self.df = self.df.sample(n=limit, random_state=42) 
            
        print(f"📊 Benchmarking on {len(self.df)} validated images.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Image Index'])
        
        labels = torch.tensor(row[LABELS].values.astype(float), dtype=torch.float32)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, labels

# === 3. MODEL LOADER ===
def load_model(path, device):
    if not os.path.exists(path):
        print(f"❌ Error: Model file not found at {path}")
        return None
        
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    
    try:
        checkpoint = torch.load(path, map_location=device)
        # Handle different save formats
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        
        # Clean keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('densenet121.', '')
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return None

# === 4. THE TRIAD BENCHMARK ===
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Triad Benchmark on {device}...")

    # Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = NIHTestDataset(CSV_PATH, IMAGE_DIR, transform=transform, limit=TEST_SAMPLES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load Models
    print("\n🧠 Loading Models...")
    model_real = load_model(MODEL_REAL_PATH, device)
    model_focal = load_model(MODEL_FOCAL_PATH, device)

    if not model_real or not model_focal:
        print("❌ Missing models. Check file paths!")
        return

    # Storage
    preds_real = []
    preds_focal = []
    preds_ensemble = []
    ground_truth = []

    # Inference Loop
    print("\n🔥 Running Inference on 3 Configurations...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            # 1. Get Raw Outputs
            out_real = model_real(images)
            out_focal = model_focal(images)
            
            # 2. Convert to Probabilities
            prob_real = torch.sigmoid(out_real).cpu().numpy()
            prob_focal = torch.sigmoid(out_focal).cpu().numpy()
            
            # 3. Calculate Ensemble (Weighted Average)
            # We give slightly more weight to the Stable model (Real) to ground the Focal model
            ensemble = (0.6 * prob_real) + (0.4 * prob_focal)
            
            # Store all 3 results
            preds_real.append(prob_real)
            preds_focal.append(prob_focal)
            preds_ensemble.append(ensemble)
            
            ground_truth.append(labels.numpy())

    # Concatenate batches
    y_true = np.vstack(ground_truth)
    y_real = np.vstack(preds_real)
    y_focal = np.vstack(preds_focal)
    y_ens = np.vstack(preds_ensemble)

    # Calculate AUCs
    auc_real = roc_auc_score(y_true, y_real, average='macro')
    auc_focal = roc_auc_score(y_true, y_focal, average='macro')
    auc_ens = roc_auc_score(y_true, y_ens, average='macro')

    # === 5. THE LEADERBOARD ===
    print("\n" + "="*50)
    print("🏆 TRIAD LEADERBOARD (Average AUC)")
    print("="*50)
    
    # Sort results
    results = [
        ("Ensemble (Combo)", auc_ens),
        ("Focal Only (Specialist)", auc_focal),
        ("Real Finetuned (Stable)", auc_real)
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (name, score) in enumerate(results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"{medal} {rank}. {name:<25} : {score:.4f}")
    
    print("-" * 50)
    
    winner = results[0][0]
    print(f"✅ WINNER: {winner}")
    
    if winner == "Ensemble (Combo)":
        print("👉 Action: Update app.py to load BOTH models and average them.")
    elif winner == "Focal Only (Specialist)":
        print("👉 Action: Rename 'chexnet_focal.pth' to 'chexnet_model.pth' and use it alone.")
    else:
        print("👉 Action: Rename 'chexnet_real_finetuned.pth' to 'chexnet_model.pth' and use it alone.")
        
    print("="*50)

if __name__ == "__main__":
    run_benchmark()