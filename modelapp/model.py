"""
Model loading, preprocessing, and prediction.

This file contains the EXACT preprocessing pipeline used in the training notebook:
- Read image as grayscale using cv2.IMREAD_GRAYSCALE
- Optional CLAHE preprocessing (disabled for inference consistency)
- Resize to 224x224 using cv2.resize
- Normalize with mean=0.5, std=0.5
- Convert to tensor format (1, H, W)

Model architecture: Transfer Learning ResNet50 with 1 input channel, 9 output classes
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import timm

# Class names in ALPHABETICALLY SORTED order (matching training notebook)
# The notebook sorts labels: labels_sorted = sorted(self.df['label'].unique())
CLASS_NAMES = [
    "Brain Healthy",
    "Brain Tumor",
    "Kidney Cyst",
    "Kidney Normal",
    "Kidney Stone",
    "Kidney Tumor",
    "Lung Malignant cases",  # 'M' comes before 'a' alphabetically
    "Lung adenocarcinoma",
    "Lung normal",
]

# Preprocessing constants from the notebook
IMG_SIZE = 224
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5


def apply_clahe(img_gray):
    """
    Apply CLAHE preprocessing (from notebook).
    Note: This is optional and typically disabled during inference for consistency.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def preprocess_image(pil_image, use_clahe=False):
    """
    Preprocess a PIL Image using the EXACT preprocessing pipeline from the notebook.

    This replicates the validation transform (val_transform) from the notebook:
    - Convert PIL to numpy array (grayscale)
    - Optionally apply CLAHE
    - Resize to IMG_SIZE x IMG_SIZE
    - Normalize with mean=0.5, std=0.5
    - Convert to tensor format

    Args:
        pil_image: PIL Image object (can be RGB or grayscale)
        use_clahe: Whether to apply CLAHE preprocessing (default: False for consistency)

    Returns:
        torch.Tensor of shape (1, 1, 224, 224) ready for model inference
    """
    # Convert PIL to numpy array as grayscale (matching notebook's gray=True)
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")

    img = np.array(pil_image)

    # Optional CLAHE (matching notebook's extra_preproc, but disabled for validation)
    if use_clahe:
        try:
            img = apply_clahe(img)
        except Exception:
            pass

    # Resize to IMG_SIZE x IMG_SIZE (matching notebook's A.Resize)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Expand dimensions to add channel dimension (H, W) -> (H, W, 1)
    img = np.expand_dims(img, axis=-1)

    # Normalize: mean=0.5, std=0.5 (matching notebook's A.Normalize)
    # This is applied to pixel values in range [0, 255]
    img = img.astype(np.float32) / 255.0  # Scale to [0, 1]
    img = (img - NORMALIZE_MEAN) / NORMALIZE_STD  # Normalize

    # Convert to tensor format: (H, W, C) -> (C, H, W)
    img = np.transpose(img, (2, 0, 1))

    # Convert to torch tensor and add batch dimension
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img_tensor


def build_transfer_model(name="resnet50", pretrained=False, in_ch=1, num_classes=9):
    """
    Build a transfer learning model using timm (matching notebook's build_transfer_model).

    Args:
        name: Model architecture name
        pretrained: Whether to use pretrained weights (should be False when loading saved model)
        in_ch: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (9 for our task)

    Returns:
        Model instance
    """
    model = timm.create_model(
        name, pretrained=pretrained, num_classes=num_classes, in_chans=in_ch
    )
    return model


# Global variable to hold the loaded model (lazy loading)
_model = None
_device = torch.device("cpu")  # Force CPU-only for memory efficiency


def load_model(model_path="tl_resnet50_best.pt"):
    """
    Load the trained model from disk (lazy loading for memory efficiency).

    This function implements lazy loading - the model is only loaded into memory
    when the first prediction request is made, reducing startup memory usage.

    The model is loaded on CPU-only to:
    - Reduce memory footprint (no GPU libraries needed)
    - Work on free-tier hosting (Render, Heroku, etc.)
    - Avoid CUDA dependencies

    The model file should be located at the project root (same folder as manage.py).
    You can specify a different path by updating the MODEL_PATH setting in settings.py.

    Args:
        model_path: Path to the saved model weights (.pt file)

    Returns:
        Loaded model in evaluation mode (CPU-only)
    """
    global _model

    if _model is None:
        print(f"[Model Loading] Loading model from {model_path} on CPU...")

        # Build the model architecture (matching notebook)
        model = build_transfer_model(
            name="resnet50",
            pretrained=False,  # We're loading saved weights
            in_ch=1,
            num_classes=len(CLASS_NAMES),
        )

        # Load the saved state dict on CPU (memory efficient)
        state_dict = torch.load(model_path, map_location=_device, weights_only=True)
        model.load_state_dict(state_dict)

        # Set to evaluation mode (disables dropout, batch norm)
        model.eval()

        # Move to CPU device
        model = model.to(_device)

        # Store globally for reuse
        _model = model

        print(f"[Model Loading] Model loaded successfully on {_device}")

    return _model


def predict_pil_image(pil_image, model_path="tl_resnet50_best.pt"):
    """
    Predict the class of a PIL Image.

    Uses lazy loading and CPU-only inference for memory efficiency.

    Args:
        pil_image: PIL Image object
        model_path: Path to the model file (optional)

    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # Load model (lazy loaded, cached after first load)
    model = load_model(model_path=model_path)

    # Preprocess the image (exact notebook preprocessing)
    img_tensor = preprocess_image(pil_image, use_clahe=False)

    # Move tensor to CPU device
    img_tensor = img_tensor.to(_device)

    # Run inference with no gradient computation (saves memory)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item() * 100  # Convert to percentage

    return predicted_class, confidence_score


def predict_image_bytes(image_bytes, model_path="tl_resnet50_best.pt"):
    """
    Predict the class of an image from raw bytes.

    Args:
        image_bytes: Image data as bytes
        model_path: Path to the model file (optional)

    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # Convert bytes to PIL Image
    from io import BytesIO

    pil_image = Image.open(BytesIO(image_bytes))

    # Use the PIL prediction function
    return predict_pil_image(pil_image, model_path=model_path)
