import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121
from torchvision import transforms, datasets
import numpy as np
import os

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class CheXNetFineTuner:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def load_model(self, model_path):
        """Load the pre-trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, len(DISEASE_LABELS))
        
        # Load with strict=False to handle key mismatches
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'module.' and 'densenet121.' prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '').replace('densenet121.', '')
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        
        model.to(self.device)
        return model
    
    def create_synthetic_calibration_data(self):
        """
        Create synthetic data to calibrate the model
        This helps reduce overconfidence on random inputs
        """
        print("🔄 Creating synthetic calibration data...")
        
        # We'll create random images with "normal" labels (all zeros)
        # This teaches the model to be less confident on random patterns
        num_samples = 1000
        synthetic_data = []
        synthetic_labels = []
        
        for i in range(num_samples):
            # Create various types of random/non-medical images
            if i % 4 == 0:
                # Random noise
                img = torch.randn(3, 224, 224)
            elif i % 4 == 1:
                # Solid colors
                color = torch.rand(1).item()
                img = torch.ones(3, 224, 224) * color
            elif i % 4 == 2:
                # Gradients
                x = torch.linspace(0, 1, 224).reshape(1, 224, 1)
                y = torch.linspace(0, 1, 224).reshape(1, 1, 224)
                img = (x + y).repeat(3, 1, 1) * 0.5
            else:
                # Mixed patterns
                img = torch.rand(3, 224, 224) * 0.5 + torch.randn(3, 224, 224) * 0.1
            
            # All synthetic data gets "normal" label (all diseases = 0)
            labels = torch.zeros(len(DISEASE_LABELS))
        test_cases = [
            ("Random Noise", torch.randn(1, 3, 224, 224)),
            ("Black Image", torch.zeros(1, 3, 224, 224)),
            ("White Image", torch.ones(1, 3, 224, 224)),
        ]
        
        for name, image in test_cases:
            with torch.no_grad():
                outputs = self.model(image.to(self.device))
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
            max_conf = np.max(probs)
            high_conf_count = np.sum(probs > 0.5)
            
            print(f"{name}: Max confidence = {max_conf:.4f}, High conf predictions = {high_conf_count}")
            
            if max_conf < 0.3:
                print(f"  ✅ Well calibrated!")
            elif max_conf < 0.6:
                print(f"  ⚠️  Moderately overconfident")
            else:
                print(f"  ❌ Still overconfident")
    
    def save_model(self, save_path):
        """Save the fine-tuned model"""
        torch.save({
            'state_dict': self.model.state_dict(),
            'arch': 'densenet121',
            'epoch': 'fine_tuned',
            'calibrated': True
        }, save_path)
        print(f"💾 Model saved to: {save_path}")

def main():
    print("🚀 CHEXNET CALIBRATION FINE-TUNING")
    print("=" * 50)
    
    # Initialize fine-tuner
    fine_tuner.save_model('models/chexnet_calibrated.pth')
    
    print("\n🎯 NEXT STEPS:")
    print("1. Update your app.py to use the calibrated model")
    print("2. Test with real X-rays to verify performance is maintained")
    print("3. The model should now reject non-X-ray images properly!")

if __name__ == "__main__":
    main()