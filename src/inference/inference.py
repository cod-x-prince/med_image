import torch
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import logging
from typing import Dict, Optional
import os
from .model_loader import get_model

# Configure logging
logger = logging.getLogger(__name__)

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

_CACHED_MODEL = None

def load_model(device: torch.device, model_path: Optional[str] = None) -> nn.Module:
    """
    Wrapper for the new robust ModelLoader.
    Ignores arguments as ModelLoader handles discovery and device.
    """
    return get_model()

def preprocess(image_path: str) -> torch.Tensor:
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        if torch.std(image_tensor) < 0.01:
            raise ValueError("Image seems too uniform or blank. Please upload a valid chest X-ray.")
        
        return image_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise

def predict(model: nn.Module, image_tensor: torch.Tensor, threshold: float = 0.2) -> Dict[str, float]:
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # With the calibrated model, we don't need complex scaling
    # Just return the raw probabilities
    predictions = []
    for label, prob in zip(DISEASE_LABELS, probs):
        if prob >= threshold:
            predictions.append((label, round(float(prob), 3)))

    # Show top 3 predictions if none meet threshold
    if not predictions:
        top_indices = probs.argsort()[-3:][::-1]
        predictions = [(DISEASE_LABELS[i], round(float(probs[i]), 3)) for i in top_indices]

    return dict(predictions)