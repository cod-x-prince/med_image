import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODELS_DIR = "models"  # Directory to search for .pth files
IMAGE_PATH = "edema.jpg" 

NIH_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]

# ==========================================
# 2. IMPROVED MODEL LOADING
# ==========================================
def load_model(path, device):
    """Load model with proper error handling"""
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(NIH_CLASSES))
    
    try:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Clean up module prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        print(f"✅ Loaded {os.path.basename(path)}")
        return model.to(device)
    except Exception as e:
        print(f"❌ Failed to load {os.path.basename(path)}: {e}")
        return None

# ==========================================
# 3. IMPROVED STRESS TESTER
# ==========================================
class ImprovedStressTester:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def preprocess_image(self, image_np):
        """Proper preprocessing for CheXNet"""
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert numpy to PIL for consistent processing
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        return transform(image_pil).unsqueeze(0).to(self.device)
    
    def predict(self, image_tensor):
        """Get predictions with probabilities for all classes"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs)  # Use sigmoid for multi-label
            return probabilities.cpu().numpy()[0]
    
    def apply_perturbations(self, image_np):
        """Apply various perturbations"""
        perturbations = {}
        
        # Original
        perturbations["Original"] = image_np.copy()
        
        # Gaussian Noise
        noise = np.random.normal(0, 0.1, image_np.shape)
        perturbations["Noise"] = np.clip(image_np + noise, 0, 1)
        
        # Brightness change
        perturbations["Brightness +"] = np.clip(image_np * 1.3, 0, 1)
        perturbations["Brightness -"] = np.clip(image_np * 0.7, 0, 1)
        
        # Blur
        blurred = cv2.GaussianBlur((image_np * 255).astype(np.uint8), (15, 15), 5)
        perturbations["Blur"] = blurred.astype(np.float32) / 255.0
        
        # Contrast
        mean = np.mean(image_np)
        perturbations["Contrast +"] = np.clip((image_np - mean) * 1.5 + mean, 0, 1)
        
        return perturbations
    
    def fgsm_attack(self, image_tensor, epsilon=0.03):
        """FGSM attack for multi-label classification"""
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        
        outputs = self.model(image_tensor)
        
        # For multi-label, we target all classes to decrease confidence
        target = torch.zeros_like(outputs)
        loss = F.binary_cross_entropy_with_logits(outputs, target)
        
        self.model.zero_grad()
        loss.backward()
        
        if image_tensor.grad is not None:
            perturbation = epsilon * image_tensor.grad.sign()
            adversarial = image_tensor + perturbation
            return torch.clamp(adversarial, 0, 1).detach()
        return image_tensor.detach()

# ==========================================
# 4. COMPREHENSIVE EVALUATION
# ==========================================
def load_image(path):
    """Load and preprocess image"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load with PIL for consistency
    image = Image.open(path).convert('RGB')
    image = image.resize((224, 224))
    
    # Convert to numpy for processing
    image_np = np.array(image).astype(np.float32) / 255.0
    return image_np

def evaluate_model_performance(tester, image_np, model_name):
    """Comprehensive evaluation of model performance"""
    print(f"\n🔍 Evaluating {model_name}...")
    
    # Test original image
    original_tensor = tester.preprocess_image(image_np)
    original_probs = tester.predict(original_tensor)
    
    print("📊 Original Image Predictions (Top 5):")
    top_indices = np.argsort(original_probs)[-5:][::-1]
    for idx in top_indices:
        print(f"   {NIH_CLASSES[idx]}: {original_probs[idx]:.3f}")
    
    # Test perturbations
    perturbations = tester.apply_perturbations(image_np)
    
    perturbation_results = {}
    for name, perturbed_img in perturbations.items():
        perturbed_tensor = tester.preprocess_image(perturbed_img)
        perturbed_probs = tester.predict(perturbed_tensor)
        
        # Calculate consistency with original prediction
        top_original = np.argmax(original_probs)
        top_perturbed = np.argmax(perturbed_probs)
        consistency = 1.0 if top_original == top_perturbed else 0.0
        
        confidence_change = abs(original_probs[top_original] - perturbed_probs[top_perturbed])
        
        perturbation_results[name] = {
            'top_class': NIH_CLASSES[top_perturbed],
            'confidence': perturbed_probs[top_perturbed],
            'consistency': consistency,
            'confidence_change': confidence_change
        }
    
    # Test adversarial robustness
    adversarial_tensor = tester.fgsm_attack(original_tensor)
    adversarial_probs = tester.predict(adversarial_tensor)
    
    adv_top_class = NIH_CLASSES[np.argmax(adversarial_probs)]
    adv_confidence = np.max(adversarial_probs)
    
    original_top_class = NIH_CLASSES[np.argmax(original_probs)]
    
    return {
        'original_prediction': original_top_class,
        'original_confidence': np.max(original_probs),
        'perturbation_results': perturbation_results,
        'adversarial_prediction': adv_top_class,
        'adversarial_confidence': adv_confidence,
        'adversarial_robustness': 1.0 if original_top_class == adv_top_class else 0.0
    }

def compare_models():
    """Main comparison function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load image
    try:
        image_np = load_image(IMAGE_PATH)
        print(f"✅ Loaded image: {IMAGE_PATH}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # --- AUTO-DISCOVERY OF MODELS ---
    print(f"\n🔎 Searching for models in {MODELS_DIR}...")
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    
    if not model_files:
        print(f"❌ No .pth files found in {MODELS_DIR}")
        return

    models_dict = {}
    for path in model_files:
        # Create a clean name from filename
        name = os.path.basename(path).replace(".pth", "")
        model = load_model(path, device)
        if model is not None:
            models_dict[name] = ImprovedStressTester(model, device)
    
    if not models_dict:
        print("❌ No models loaded successfully!")
        return
    
    # Evaluate each model
    results = {}
    for name, tester in models_dict.items():
        results[name] = evaluate_model_performance(tester, image_np, name)
    
    # Display comparative results
    print("\n" + "="*60)
    print("📈 COMPARATIVE RESULTS (ALL MODELS)")
    print("="*60)
    
    # Original predictions comparison
    print("\n🖼️  ORIGINAL IMAGE PREDICTIONS:")
    for model_name, result in results.items():
        print(f"   {model_name:25}: {result['original_prediction']:20} (conf: {result['original_confidence']:.3f})")
    
    # Perturbation robustness comparison
    print("\n🛡️  PERTURBATION ROBUSTNESS (Consistency with Original):")
    perturbations = list(next(iter(results.values()))['perturbation_results'].keys())
    
    for perturbation in perturbations:
        print(f"\n   {perturbation}:")
        for model_name, result in results.items():
            pert_result = result['perturbation_results'][perturbation]
            symbol = "✅" if pert_result['consistency'] else "❌"
            print(f"     {model_name:25}: {symbol} {pert_result['top_class']:20} (conf: {pert_result['confidence']:.3f}, Δ: {pert_result['confidence_change']:.3f})")
    
    # Adversarial robustness
    print("\n⚔️  ADVERSARIAL ATTACK ROBUSTNESS:")
    for model_name, result in results.items():
        symbol = "✅" if result['adversarial_robustness'] else "❌"
        print(f"   {model_name:25}: {symbol} Predicted: {result['adversarial_prediction']:20} (conf: {result['adversarial_confidence']:.3f})")
    
    # Calculate overall scores
    print("\n🏆 OVERALL ROBUSTNESS SCORES:")
    for model_name, result in results.items():
        # Consistency score across perturbations
        consistency_score = np.mean([r['consistency'] for r in result['perturbation_results'].values()])
        
        # Confidence stability (lower change is better)
        confidence_stability = 1.0 - np.mean([r['confidence_change'] for r in result['perturbation_results'].values()])
        
        # Overall score
        overall_score = (consistency_score + confidence_stability + result['adversarial_robustness']) / 3
        
        print(f"   {model_name:25}:")
        print(f"     Consistency:    {consistency_score:.3f}")
        print(f"     Conf Stability: {confidence_stability:.3f}")
        print(f"     Adv Robustness: {result['adversarial_robustness']:.3f}")
        print(f"     ──────────────────────────")
        print(f"     OVERALL SCORE:  {overall_score:.3f}")

if __name__ == "__main__":
    compare_models()