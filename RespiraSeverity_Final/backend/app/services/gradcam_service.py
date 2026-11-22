import io
import base64
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm

class GradCAMService:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Computes Grad-CAM activation map for the given input and class index.
        Returns a 2D numpy array of shape (224, 224) with values in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Enable grad for backward pass
        with torch.enable_grad():
            x = input_tensor.clone().requires_grad_(True)
            output = self.model(x)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            score = output[0, class_idx]
            score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured.")

        # Channel-wise pooling of gradients
        # activations: (1, C, H, W), gradients: (1, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    @staticmethod
    def create_visualizations(pil_img: Image.Image, cam_np: np.ndarray):
        """
        Generates base64 encoded strings for:
        1. Original image
        2. Color heatmap (JET colormap)
        3. Heatmap overlaid on original radiograph
        """
        img_resized = pil_img.convert("RGB").resize((224, 224))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # Apply JET colormap via matplotlib or fallback
        try:
            import matplotlib
            colormap = matplotlib.colormaps["jet"]
        except Exception:
            try:
                colormap = cm.get_cmap("jet")
            except Exception:
                colormap = None

        if colormap is not None:
            heatmap_rgba = colormap(cam_np)  # RGBA in [0, 1]
            heatmap_rgb = heatmap_rgba[:, :, :3]  # drop alpha
        else:
            # Fallback manual JET colormap implementation
            heatmap_rgb = np.zeros((224, 224, 3), dtype=np.float32)
            heatmap_rgb[:, :, 0] = np.clip(1.5 - np.abs(4 * cam_np - 3), 0, 1)
            heatmap_rgb[:, :, 1] = np.clip(1.5 - np.abs(4 * cam_np - 2), 0, 1)
            heatmap_rgb[:, :, 2] = np.clip(1.5 - np.abs(4 * cam_np - 1), 0, 1)

        # Create overlay (alpha blend 0.45 heatmap + 0.55 original)
        overlay_np = 0.45 * heatmap_rgb + 0.55 * img_np
        overlay_np = np.clip(overlay_np * 255.0, 0, 255).astype(np.uint8)
        heatmap_uint8 = np.clip(heatmap_rgb * 255.0, 0, 255).astype(np.uint8)

        def to_base64(arr_or_img):
            if isinstance(arr_or_img, np.ndarray):
                im = Image.fromarray(arr_or_img)
            else:
                im = arr_or_img
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "original_base64": to_base64(img_resized),
            "heatmap_base64": to_base64(heatmap_uint8),
            "overlay_base64": to_base64(overlay_np)
        }
