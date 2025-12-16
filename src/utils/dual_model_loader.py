import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import OrderedDict
import logging
from ..config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualModelManager:
    """Manager for loading and switching between standard and deep scan models"""
    
    def __init__(self, config=config):
        self.config = config
        self.models = {}
        self.transforms = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all models based on configuration"""
        logger.info("🚀 Loading dual-model system...")
        
        # Load Standard Model (224px)
        try:
            standard_config = self.config.MODEL_CONFIGS['standard']
            self.models['standard'] = self._load_single_model(standard_config['path'])
            self.transforms['standard'] = self._get_transform(standard_config['resolution'])
            logger.info(f"✅ Loaded Standard Model")
        except Exception as e:
            logger.error(f"❌ Failed to load Standard Model: {e}")
        
        # Load Deep Scan Model (1024px)
        try:
            deep_config = self.config.MODEL_CONFIGS['deep']
            self.models['deep'] = self._load_single_model(deep_config['path'])
            self.transforms['deep'] = self._get_transform(deep_config['resolution'])
            logger.info(f"✅ Loaded Deep Scan Model")
        except Exception as e:
            logger.error(f"❌ Failed to load Deep Scan Model: {e}")
        
        # Load Ensemble Model
        try:
            ensemble_config = self.config.MODEL_CONFIGS['ensemble']
            self.models['ensemble'] = self._load_ensemble_model(
                ensemble_config['paths'], 
                ensemble_config['weights']
            )
            # Ensemble uses the resolution of the first model (usually 224px for this setup)
            # If models had mixed resolutions, we'd need per-model transforms in the ensemble list.
            # Simplified here assuming both use same resolution or handled via config.
            self.transforms['ensemble'] = self._get_transform(ensemble_config['resolutions'][0]) 
            logger.info(f"✅ Loaded Ensemble Model")
        except Exception as e:
             logger.error(f"❌ Failed to load Ensemble Model: {e}")

    def _get_transform(self, resolution):
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_single_model(self, model_path):
        """Load a single DenseNet121 model"""
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 14)  # NIH 14 pathologies
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean state dict keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('densenet121.', '').replace('model.', '')
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_ensemble_model(self, model_paths, weights):
        """Load multiple models for ensemble inference"""
        models_list = []
        for path in model_paths:
            model = self._load_single_model(path)
            models_list.append(model)
        
        return {
            'models': models_list,
            'weights': weights
        }
    
    def predict(self, image, scan_type='standard', threshold=0.15):
        """Run prediction with selected model type"""
        if scan_type not in self.models:
            # Fallback or error
            if 'standard' in self.models:
                logger.warning(f"Scan type {scan_type} not found, falling back to standard.")
                scan_type = 'standard'
            else:
                 raise ValueError(f"Unknown scan type: {scan_type} and Standard model not available.")
        
        transform = self.transforms[scan_type]
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if scan_type == 'ensemble':
                # Ensemble prediction
                all_probs = []
                ensemble_data = self.models['ensemble']
                
                for model, weight in zip(ensemble_data['models'], ensemble_data['weights']):
                    output = model(image_tensor)
                    prob = torch.sigmoid(output) * weight
                    all_probs.append(prob)
                
                # Weighted Sum (Weights should sum to 1 approx, or logic handles it)
                final_prob = torch.stack(all_probs).sum(dim=0)
            else:
                # Single model prediction
                model = self.models[scan_type]
                output = model(image_tensor)
                final_prob = torch.sigmoid(output)
        
        # Convert to numpy and apply threshold
        probs = final_prob.cpu().numpy()[0]
        
        # Get labels from logic (Assuming standad NIH 14)
        labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # Format results with confidence
        results = []
        for label, prob in zip(labels, probs):
            if prob >= threshold:
                results.append({
                    'label': label,
                    'confidence': float(prob),
                    'percentage': f"{prob * 100:.1f}%"
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predictions': results,
            'scan_type': scan_type,
            'model_info': self.config.MODEL_CONFIGS[scan_type],
            'performance': self.config.PERFORMANCE_METRICS[scan_type]
        }

# Singleton instance
dual_model_manager = None

def get_dual_model_manager(config=config):
    """Get or create the dual model manager"""
    global dual_model_manager
    if dual_model_manager is None:
        dual_model_manager = DualModelManager(config)
    return dual_model_manager
