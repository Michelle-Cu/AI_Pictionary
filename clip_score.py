import torch
import torch.nn.functional as F
import math
from PIL import Image
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    AutoImageProcessor, 
    AutoModel
)

_clip_model = None
_clip_processor = None
_dino_model = None
_dino_processor = None

def load_models():
    """Loads both CLIP and DINO models"""
    global _clip_model, _clip_processor, _dino_model, _dino_processor
    
    print("Loading CLIP model (openai/clip-vit-base-patch32)…")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading DINO model (facebook/dinov2-base)…")
    _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
    
    print("Models ready.")

def calculate_score(ref_path: str, submission_path: str, weight_clip: float = 0.7, weight_dino: float = 0.3) -> int:
    """
    Returns a blended similarity score (0-100) using both CLIP and DINO.
    Defaults to a 50/50 split.
    """
    # Open images once
    img1 = Image.open(ref_path).convert("RGB")
    img2 = Image.open(submission_path).convert("RGB")

    # --- 1. CLIP Score (Semantic Meaning) ---
    clip_inputs = _clip_processor(images=[img1, img2], return_tensors="pt")
    with torch.no_grad():
        clip_vision = _clip_model.vision_model(pixel_values=clip_inputs["pixel_values"])
        clip_features = _clip_model.visual_projection(clip_vision.pooler_output)
    
    clip_features = F.normalize(clip_features, p=2, dim=1)
    clip_sim = torch.dot(clip_features[0], clip_features[1]).item()

    # --- 2. DINO Score (Structural/Visual Layout) ---
    dino_inputs = _dino_processor(images=[img1, img2], return_tensors="pt")
    with torch.no_grad():
        dino_outputs = _dino_model(**dino_inputs)
        # Extract the [CLS] token for global image features
        dino_features = dino_outputs.last_hidden_state[:, 0, :]
        
    dino_features = F.normalize(dino_features, p=2, dim=1)
    dino_sim = torch.dot(dino_features[0], dino_features[1]).item()

    # --- 3. Blended Score Calculation ---
    blended_sim = (weight_clip * clip_sim) + (weight_dino * dino_sim)
    
    # Clamp the result to ensure floating point inaccuracies don't push it slightly above 1.0
    blended_sim = max(0.0, min(1.0, blended_sim))

    # Sigmoid function to change steepness for values in the middle
    # 'k' controls the steepness. A value of 10-15 preferable
    k = 12.0 
    midpoint = 0.5
    
    # Apply the math formula for an S-Curve
    curve_sim = 1 / (1 + math.exp(-k * (blended_sim - midpoint)))
    
    # The math naturally outputs between ~0.0 and ~1.0, but we clamp it just in case
    curve_sim = max(0.0, min(1.0, curve_sim))

    return round(curve_sim * 100)