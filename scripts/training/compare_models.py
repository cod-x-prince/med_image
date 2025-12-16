import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
from PIL import Image
import os
import random
import glob

# === SETTINGS ===
MODELS_DIR = 'models'
IMAGES_DIR = 'data/images'
TEST_IMAGE = 'app/static/uploads/edema.jpg' # The tricky case
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# === 1. ROBUST MODEL LOADER ===
def load_model(model_path, device):
    try:
        # Initialize Architecture
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, len(DISEASE_LABELS))
        
        # Load Weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different save formats (some wrap weights in 'state_dict')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Fix Key Names (Remove 'module.' or 'densenet121.' prefixes)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('densenet121.', '')
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_path}: {e}")
        return None

# === 2. PREDICTION FUNCTION ===
def get_predictions(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
        # Return dict of results
        return {label: round(float(prob), 3) for label, prob in zip(DISEASE_LABELS, probs)}
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# === 3. MAIN COMPARISON LOOP ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Comparison on: {device}")
    
    # Find all .pth files
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    if not model_files:
        print("❌ No models found in 'models/' folder!")
        return

    print(f"🔎 Found {len(model_files)} models: {[os.path.basename(m) for m in model_files]}")
    
    # Load all models into memory
    loaded_models = {}
    for m_path in model_files:
        name = os.path.basename(m_path)
        print(f"Loading {name}...", end=" ")
        model = load_model(m_path, device)
        if model:
            loaded_models[name] = model
            print("✅")
    
    # --- TEST 1: The "Edema" Challenge ---
    print("\n" + "="*60)
    print(f"🧪 TEST CASE 1: KNOWN EDEMA ({os.path.basename(TEST_IMAGE)})")
    print("="*60)
    
    if os.path.exists(TEST_IMAGE):
        print(f"{'Model Name':<30} | {'Edema Score':<12} | {'Top Prediction':<20}")
        print("-" * 70)
        
        for name, model in loaded_models.items():
            preds = get_predictions(model, TEST_IMAGE, device)
            if preds:
                edema_score = preds['Edema']
                top_pred = max(preds, key=preds.get)
                top_val = preds[top_pred]
                
                # Highlight good performance
                score_display = f"{edema_score:.3f}"
                if edema_score > 0.15: score_display += " ⭐" # Good detection
                
                print(f"{name:<30} | {score_display:<12} | {top_pred} ({top_val:.3f})")
    else:
        print(f"⚠️ Test image {TEST_IMAGE} not found.")

    # --- TEST 2: Random Real X-Rays ---
    print("\n" + "="*60)
    print("🧪 TEST CASE 2: RANDOM REAL X-RAYS")
    print("="*60)
    
    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    if all_images:
        random_images = random.sample(all_images, 2) # Pick 2 random images
        
        for img_path in random_images:
            print(f"\n📷 Image: {os.path.basename(img_path)}")
            print(f"{'Model Name':<30} | {'Top 1':<20} | {'Top 2':<20}")
            print("-" * 80)
            
            for name, model in loaded_models.items():
                preds = get_predictions(model, img_path, device)
                if preds:
                    # Sort predictions
                    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
                    top1 = f"{sorted_preds[0][0]} ({sorted_preds[0][1]:.3f})"
                    top2 = f"{sorted_preds[1][0]} ({sorted_preds[1][1]:.3f})"
                    print(f"{name:<30} | {top1:<20} | {top2:<20}")
    else:
        print("⚠️ No images found in data/images to test.")

if __name__ == "__main__":
    main()