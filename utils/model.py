"""
Flood detection model wrapper.

Loads the pretrained-encoder U-Net (segmentation-models-pytorch, ResNet34
encoder) trained in the CV pipeline. If no checkpoint is found, falls back
to a simple HSV color-heuristic so the app remains usable in demo mode
while the real model is being trained/tuned.
"""

import os
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

CLASS_NAMES = [
    "Background", "Building-flooded", "Building-non-flooded", "Road-flooded",
    "Road-non-flooded", "Water", "Tree", "Vehicle", "Pool", "Grass",
]
FLOOD_CLASSES = {1, 3, 5}  # Building-flooded, Road-flooded, Water
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 256


def build_model(num_classes=NUM_CLASSES):
    """Same architecture used in training: ResNet34 encoder pretrained on ImageNet."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # weights load from our checkpoint below, not re-downloaded
        in_channels=3,
        classes=num_classes,
    )


class FloodDetector:
    """
    Loads the trained checkpoint if present at `weights_path`.
    Falls back to an HSV-based water heuristic ("demo mode") otherwise,
    so the app is usable immediately while the real model is trained.
    """

    def __init__(self, weights_path="models/best_model_pretrained.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.demo_mode = True
        self.model = None
        if weights_path and os.path.exists(weights_path):
            try:
                self.model = build_model(NUM_CLASSES).to(self.device)
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.model.eval()
                self.demo_mode = False
            except Exception as e:
                print(f"Could not load model weights ({e}); using demo heuristic mode.")

    def _preprocess(self, image_bgr):
        img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def _heuristic_mask(self, image_bgr):
        """Rough water-color detector used only when no trained model is available."""
        img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower1, upper1 = np.array([15, 20, 20]), np.array([35, 160, 200])
        lower2, upper2 = np.array([85, 15, 40]), np.array([130, 255, 200])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        water_mask = cv2.bitwise_or(mask1, mask2)
        pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        pred[water_mask > 0] = 5  # class 5 = Water
        return pred

    def predict(self, image_bgr):
        """
        Returns dict with: mask (HxW class ids), flood_pct (float 0-100),
        class_breakdown (dict), severity (str), demo_mode (bool)
        """
        if self.model is not None:
            with torch.no_grad():
                tensor = self._preprocess(image_bgr)
                logits = self.model(tensor)
                mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        else:
            mask = self._heuristic_mask(image_bgr)

        total = mask.size
        breakdown = {}
        flood_pixels = 0
        for cls_id, name in enumerate(CLASS_NAMES):
            cnt = int((mask == cls_id).sum())
            if cnt > 0:
                breakdown[name] = round(100 * cnt / total, 2)
            if cls_id in FLOOD_CLASSES:
                flood_pixels += cnt

        flood_pct = round(100 * flood_pixels / total, 2)
        severity = self._severity(flood_pct)

        return {
            "mask": mask,
            "flood_pct": flood_pct,
            "class_breakdown": breakdown,
            "severity": severity,
            "demo_mode": self.demo_mode,
        }

    @staticmethod
    def _severity(flood_pct):
        if flood_pct < 3:
            return "No flooding detected"
        if flood_pct < 15:
            return "Low"
        if flood_pct < 35:
            return "Moderate"
        if flood_pct < 60:
            return "High"
        return "Severe"


def colorize_mask(mask):
    """Turn a class-id mask into an RGB image for display."""
    palette = np.array([
        [60, 60, 60], [220, 20, 60], [255, 140, 0], [30, 144, 255],
        [255, 215, 0], [0, 90, 200], [34, 139, 34], [148, 0, 211],
        [0, 206, 209], [124, 200, 89],
    ], dtype=np.uint8)
    return palette[mask]
