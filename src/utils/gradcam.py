import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

class GradCAMPlusPlus:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            one_hot = torch.zeros_like(output)
            one_hot[0][class_idx] = 1
            output.backward(gradient=one_hot, retain_graph=True)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Gradients or activations not captured.")

            grads = self.gradients[0]
            activations = self.activations[0]

            numerator = grads.pow(2)
            denominator = 2 * grads.pow(2) + torch.sum(activations * grads.pow(3), dim=(1, 2), keepdim=True) + 1e-8
            alpha = numerator / denominator
            weights = (alpha * F.relu(grads)).sum(dim=(1, 2))

            cam = (weights[:, None, None] * activations).sum(dim=0)
            cam = F.relu(cam)
            
            if cam.max() > 0:
                cam -= cam.min()
                cam /= cam.max()
            
            return cam.cpu().detach().numpy()
        except Exception as e:
            logger.error(f"Error generating GradCAM: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def overlay_heatmap(img_path: str, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
            
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return overlayed
    except Exception as e:
        logger.error(f"Error overlaying heatmap: {e}")
        # Return original image or blank if failed
        if 'img' in locals() and img is not None:
            return img
        return np.zeros((224, 224, 3), dtype=np.uint8)
