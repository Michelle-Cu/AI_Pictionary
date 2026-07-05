import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoImageProcessor,
    AutoModel,
)

_clip_model = None
_clip_processor = None
_dino_model = None
_dino_processor = None

def load_models():
    global _clip_model, _clip_processor, _dino_model, _dino_processor

    print("Loading CLIP model (openai/clip-vit-base-patch32)…")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading DINO model (facebook/dinov2-base)…")
    _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _dino_model = AutoModel.from_pretrained("facebook/dinov2-base")

    # Disable TF2 behaviour once at startup so tf.Session is available
    print("Initializing TensorFlow for LPIPS…")
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    print("Models ready.")


def _lpips_sim(img1: Image.Image, img2: Image.Image) -> float:
    """Compute LPIPS similarity (1 - distance) between two PIL images."""
    import tensorflow.compat.v1 as tf
    from utils.lpips import lpips_tf as lpips_module

    size = (64, 64)
    arr1 = np.array(img1.resize(size)).astype(np.float32)[np.newaxis] / 255.0
    arr2 = np.array(img2.resize(size)).astype(np.float32)[np.newaxis] / 255.0

    # Use a fresh graph each call to avoid accumulating ops
    with tf.Graph().as_default():
        t0 = tf.constant(arr1, dtype=tf.float32)
        t1 = tf.constant(arr2, dtype=tf.float32)
        dist_t = lpips_module.lpips(t0, t1, model='net-lin', net='alex')
        with tf.Session() as sess:
            dist = float(np.squeeze(sess.run(dist_t)))

    # LPIPS is a distance (0 = identical); convert to a similarity in [0, 1]
    return max(0.0, 1.0 - dist)


def calculate_score(ref_path: str, submission_path: str,
                    weight_clip: float = 0.40,
                    weight_dino: float = 0.45,
                    weight_lpips: float = 0.15) -> int:
    """
    Returns a blended similarity score (0–100).
    Weights: DINO 0.45 + CLIP 0.40 + LPIPS 0.15
    """
    img1 = Image.open(ref_path).convert("RGB")
    img2 = Image.open(submission_path).convert("RGB")

    # --- 1. CLIP Score (semantic meaning) ---
    clip_inputs = _clip_processor(images=[img1, img2], return_tensors="pt")
    with torch.no_grad():
        clip_vision = _clip_model.vision_model(pixel_values=clip_inputs["pixel_values"])
        clip_features = _clip_model.visual_projection(clip_vision.pooler_output)
    clip_features = F.normalize(clip_features, p=2, dim=1)
    clip_sim = torch.dot(clip_features[0], clip_features[1]).item()

    # --- 2. DINO Score (structural / visual layout) ---
    dino_inputs = _dino_processor(images=[img1, img2], return_tensors="pt")
    with torch.no_grad():
        dino_outputs = _dino_model(**dino_inputs)
        dino_features = dino_outputs.last_hidden_state[:, 0, :]
    dino_features = F.normalize(dino_features, p=2, dim=1)
    dino_sim = torch.dot(dino_features[0], dino_features[1]).item()

    # --- 3. LPIPS Score (perceptual pixel-level similarity) ---
    lpips_sim = _lpips_sim(img1, img2)

    # --- 4. Weighted blend ---
    blended_sim = (weight_clip * clip_sim) + (weight_dino * dino_sim) + (weight_lpips * lpips_sim)
    blended_sim = max(0.0, min(1.0, blended_sim))

    # S-curve to spread mid-range scores; k=12 gives a moderate steepness
    k = 12.0
    curve_sim = 1 / (1 + math.exp(-k * (blended_sim - 0.5)))
    curve_sim = max(0.0, min(1.0, curve_sim))

    return round((curve_sim ** 2) * 100)
