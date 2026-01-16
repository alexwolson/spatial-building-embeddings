# Publish Spatial Building Embeddings to Hugging Face

This directory contains tools to package your trained "Spatial Building Embeddings" model into a standard Hugging Face format. This allows anyone to load the end-to-end model (DINOv2 Backbone + Your Trained Projector) with a few lines of code.

## Prerequisites

Ensure you have the project dependencies installed (`uv sync` or similar).

You need:
1. `config.toml` in the project root (defines the model architecture).
2. A trained checkpoint file (e.g., `data/checkpoints/checkpoint_best.pt`).

## 1. Convert the Model

Run the conversion script to generate the Hugging Face model files in `published_model/`:

```bash
python publish_model/convert_to_hf.py
```

This script:
- Reads architecture parameters from `config.toml`.
- Loads the backbone (e.g., `facebook/dinov2-base`).
- Loads your trained projector weights.
- Saves the combined model and configuration to `published_model/`.

## 2. Verify the Model

Run the demo script to verify that the saved model can be loaded and run:

```bash
python publish_model/demo_usage.py
```

You should see output indicating successful loading and inference, with an embedding norm close to 1.0.

## 3. Upload to Hugging Face Hub

To publish the model, you need a Hugging Face account and the `huggingface_hub` library.

1. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

2. Upload the folder:
   ```bash
   huggingface-cli upload your-username/spatial-building-embeddings published_model/ .
   ```
   *(Replace `your-username` with your actual Hugging Face username)*

## 4. Usage for End Users

### Quick Start: Using the Helper Script

The easiest way to generate embeddings is using the provided script:

```bash
# Single image
python publish_model/generate_embeddings_from_hf.py path/to/image.jpg

# Multiple images
python publish_model/generate_embeddings_from_hf.py image1.jpg image2.jpg image3.jpg

# Save embeddings to file
python publish_model/generate_embeddings_from_hf.py image.jpg --output embeddings.npy

# Use a different model ID
MODEL_ID="custom/org/model-name" python publish_model/generate_embeddings_from_hf.py image.jpg
```

### Python API: Loading from HuggingFace Hub

Users can load your model and generate embeddings programmatically:

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch
import numpy as np

# Load model from HuggingFace Hub
model_id = "alexwaolson/spatial-building-embeddings"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load image processor from the backbone model name
backbone_name = model.config.backbone_model_name
processor = AutoImageProcessor.from_pretrained(backbone_name)

# Load and preprocess image
image = Image.open("path/to/image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate embedding
with torch.no_grad():
    outputs = model(**inputs)
    # The model returns a dict with 'embeddings' key or a tuple
    if isinstance(outputs, dict):
        embeddings = outputs["embeddings"]
    else:
        embeddings = outputs[0]
    
    # Remove batch dimension and convert to numpy
    embeddings = embeddings.squeeze(0).cpu().numpy()

print(f"Embedding shape: {embeddings.shape}")  # (256,)
print(f"Embedding norm: {np.linalg.norm(embeddings):.4f}")  # Should be ~1.0
```

### Batch Processing

For processing multiple images efficiently:

```python
# Load images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
images = [Image.open(path).convert("RGB") for path in image_paths]

# Preprocess batch
inputs = processor(images=images, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate embeddings for batch
with torch.no_grad():
    outputs = model(**inputs)
    if isinstance(outputs, dict):
        embeddings = outputs["embeddings"]
    else:
        embeddings = outputs[0]
    
    embeddings_np = embeddings.cpu().numpy()

print(f"Embeddings shape: {embeddings_np.shape}")  # (batch_size, 256)
```

### Notes

- **Gated Models**: If the model is gated, you'll need to set the `HF_TOKEN` environment variable or login via `huggingface-cli login`
- **Custom Code**: The `trust_remote_code=True` parameter is required because the model uses custom configuration and modeling classes
- **Device**: The model works on both CPU and GPU, but GPU is recommended for faster inference
- **Output**: Embeddings are 256-dimensional vectors with L2 norm approximately 1.0

