import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import requests
from pathlib import Path

def main():
    model_path = Path(__file__).parent.parent / "published_model"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please run convert_to_hf.py first.")
        return

    print(f"Loading model from {model_path}...")
    # Trust remote code is needed because the model class is custom and local
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    print("Model loaded successfully!")
    print(f"Config: {model.config}")

    # Load image processor from the backbone
    # We can get the backbone name from the config
    backbone_name = model.config.backbone_model_name
    print(f"Loading image processor for {backbone_name}...")
    processor = AutoImageProcessor.from_pretrained(backbone_name)

    # Load a sample image
    # Try to find a local image first, otherwise download one
    image = None
    try:
        # Check for a local image in data/raw/ or similar if accessible, but let's use a reliable web image for the demo
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        print("Loaded sample image from web.")
    except Exception as e:
        print(f"Could not load sample image: {e}")
        # Create dummy image
        image = Image.new('RGB', (224, 224), color='red')
        print("Created dummy image.")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        # The model returns a tuple (embeddings,) or dict if return_dict=True
        if isinstance(outputs, dict):
            embeddings = outputs['embeddings']
        else:
            embeddings = outputs[0]

    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings, dim=1).item():.4f} (Expected ~1.0)")
    print("Success!")

if __name__ == "__main__":
    main()

