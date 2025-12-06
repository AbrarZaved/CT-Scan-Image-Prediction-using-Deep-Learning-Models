"""
Utility functions for image handling and processing.
"""

from PIL import Image
from io import BytesIO


def validate_image(image_bytes):
    """
    Validate that the provided bytes represent a valid image.

    Args:
        image_bytes: Raw image data as bytes

    Returns:
        PIL.Image object if valid, raises exception otherwise
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()  # Verify it's a valid image

        # Re-open after verify (verify closes the file)
        img = Image.open(BytesIO(image_bytes))
        return img
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def bytes_to_pil(image_bytes):
    """
    Convert image bytes to PIL Image.

    Args:
        image_bytes: Raw image data as bytes

    Returns:
        PIL.Image object
    """
    return Image.open(BytesIO(image_bytes))
