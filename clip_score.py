import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None


def load_clip_model():
    global _model, _processor
    print("Loading CLIP model (openai/clip-vit-base-patch32)…")
    _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model ready.")


def calculate_score(ref_path: str, submission_path: str) -> int:
    """Return raw CLIP cosine similarity × 100, rounded to int."""
    img1 = Image.open(ref_path).convert("RGB")
    img2 = Image.open(submission_path).convert("RGB")

    inputs = _processor(images=[img1, img2], return_tensors="pt")
    with torch.no_grad():
        vision_out = _model.vision_model(pixel_values=inputs["pixel_values"])
        features = _model.visual_projection(vision_out.pooler_output)

    features = F.normalize(features, p=2, dim=1)
    similarity = torch.dot(features[0], features[1]).item()
    return round(similarity * 100)
