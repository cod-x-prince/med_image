"""
Robust model loader with fallback support.
Place in: src/med_image/inference/model_loader.py
"""

import os
import torch
import torchvision.models as models
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Robust model loader with multiple fallback strategies."""
    
    def __init__(self, model_dir="models", device="cuda"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.available_models = self._scan_models()
        
    def _scan_models(self):
        """Scan for available model files."""
        models_found = {}
        
        # Common model file patterns
        model_patterns = [
            "chexnet_model.pth",
            "balanced_robust_chexnet.pth",
            "chexnet_real_finetuned.pth",
            "high_res_chexnet.pth"
        ]
        
        for pattern in model_patterns:
            model_path = self.model_dir / pattern
            if model_path.exists():
                models_found[pattern] = model_path
                logger.info(f"Found model: {pattern}")
        
        return models_found
    
    def load_best_available(self):
        """Load the best available model with fallback."""
        model_priority = [
            "chexnet_model.pth",            # Your specified model
            "balanced_robust_chexnet.pth",  # Best model
            "high_res_chexnet.pth",         # High res fallback
            "chexnet_real_finetuned.pth"    # Original name
        ]
        
        for model_name in model_priority:
            if model_name in self.available_models:
                return self.load_model(self.available_models[model_name])
        
        # If no model files found, create a basic model
        logger.warning("No model files found. Creating basic DenseNet121...")
        return self._create_basic_model()
    
    def load_model(self, model_path):
        """Load a specific model file."""
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Create DenseNet121 with 14 classes (CheXNet architecture)
            try:
                from torchvision.models import DenseNet121_Weights
                weights = DenseNet121_Weights.IMAGENET1K_V1
                model = models.densenet121(weights=weights)
            except (ImportError, AttributeError):
                # Fallback for older torchvision versions
                model = models.densenet121(pretrained=False)
                
            num_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_features, 14)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
            
            # Fix state dict keys (remove prefixes)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '').replace('densenet121.', '')
                new_state_dict[new_key] = v
                
            model.load_state_dict(new_state_dict, strict=False)
            
            model = model.to(self.device)
            model.eval()
            
            logger.info("✅ Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def _create_basic_model(self):
        """Create a basic model for fallback."""
        logger.info("Creating basic ImageNet-pretrained DenseNet121...")
        
        model = models.densenet121(pretrained=True)
        model = model.to(self.device)
        model.eval()
        
        logger.warning("⚠️ Using ImageNet-pretrained model (not medical-specific)")
        return model
    
    def get_model_info(self):
        """Get information about available models."""
        info = {
            "device": str(self.device),
            "available_models": list(self.available_models.keys()),
            "cuda_available": torch.cuda.is_available(),
            "selected_model": None
        }
        
        if self.available_models:
            # Get first available model
            first_model = next(iter(self.available_models.keys()))
            info["selected_model"] = first_model
        
        return info


# Global model instance
_model_instance = None
_loaded_model = None

def get_model():
    """Singleton pattern to get model instance."""
    global _model_instance, _loaded_model
    
    if _loaded_model is not None:
        return _loaded_model
    
    if _model_instance is None:
        # Robust path finding
        possible_paths = [
            Path(__file__).parent.parent / "models",          # src/med_image/models (relative to this file)
            Path(__file__).parent.parent.parent / "models",   # src/models (if moved)
            Path.cwd() / "src" / "med_image" / "models",      # CWD/src/med_image/models
            Path.cwd() / "models",                            # CWD/models
        ]
        
        model_dir = None
        for path in possible_paths:
            if path.exists():
                model_dir = path
                logger.info(f"✅ Found models at: {path}")
                break
        
        if model_dir is None:
            logger.warning("⚠️ Could not locate 'models' directory. Using default 'models' relative path.")
            model_dir = Path("models")
             
        _model_instance = ModelLoader(model_dir=model_dir)
    
    try:
        _loaded_model = _model_instance.load_best_available()
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        # Last resort fallback
        _loaded_model = _model_instance._create_basic_model()
        
    return _loaded_model

def get_model_info():
    """Get model information."""
    # Ensure instance exists
    if _model_instance is None:
        get_model()
        
    if _model_instance:
        return _model_instance.get_model_info()
    return {"error": "Model loader not initialized"}
