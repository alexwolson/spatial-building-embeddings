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

Once uploaded, users can load your model like this:

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import requests
import torch

# Load model
model_id = "your-username/spatial-building-embeddings"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

# Load processor (from the backbone)
# The config knows the backbone name, but users might need to specify it if they don't want to trust remote code for config
backbone_name = model.config.backbone_model_name 
processor = AutoImageProcessor.from_pretrained(backbone_name)

# Inference
image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs)[0] # shape: [1, 256]
```

