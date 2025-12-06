"""
Preprocessing Parity Test Script

This script verifies that the preprocessing pipeline in the Django backend
EXACTLY matches the preprocessing used during training in the Jupyter notebook.

It loads a sample image, applies both:
1. The notebook preprocessing pipeline (validation transform)
2. The Django backend preprocessing pipeline

Then compares the resulting tensors to ensure they are identical (within floating point tolerance).

Usage:
    python preprocess_parity_test.py --image path/to/sample_image.jpg

    Or use a sample image from your dataset:
    python preprocess_parity_test.py --image "F:\Sorted Data set\Brain Tumor\ct_tumor (2).png"
"""

import argparse
import sys
import numpy as np
import cv2
from PIL import Image
import torch

# Notebook preprocessing (validation transform - EXACT COPY from notebook)
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224

# Exact validation transform from notebook
notebook_val_transform = A.Compose(
    [A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()]
)


def notebook_preprocess(image_path):
    """
    Apply the EXACT preprocessing from the notebook's validation transform.

    This replicates:
    - val_transform from notebook Cell 2
    - CTFolderDataset.__getitem__ with transform=val_transform, extra_preproc=False
    """
    # Read as grayscale (matching gray=True in notebook)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        # Fallback to PIL if cv2 fails
        img = Image.open(image_path).convert("L")
        img = np.array(img)

    # Apply albumentations transform (matching notebook)
    # Note: albumentations expects (H, W, C) format
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    augmented = notebook_val_transform(image=img)
    img_tensor = augmented["image"]

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def backend_preprocess(image_path):
    """
    Apply the preprocessing from the Django backend (modelapp/model.py).
    """
    # Import from the backend module
    sys.path.insert(0, ".")
    from modelapp.model import preprocess_image

    # Load image as PIL
    pil_image = Image.open(image_path)

    # Apply backend preprocessing
    img_tensor = preprocess_image(pil_image, use_clahe=False)

    return img_tensor


def compare_tensors(
    tensor1, tensor2, name1="Tensor 1", name2="Tensor 2", tolerance=1e-6
):
    """
    Compare two tensors and report differences.
    """
    print(f"\n{'='*70}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"{'='*70}")

    # Check shapes
    print(f"{name1} shape: {tensor1.shape}")
    print(f"{name2} shape: {tensor2.shape}")

    if tensor1.shape != tensor2.shape:
        print("❌ FAILED: Shapes do not match!")
        return False

    # Convert to numpy for comparison
    arr1 = tensor1.numpy()
    arr2 = tensor2.numpy()

    # Check if identical
    are_equal = np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance)

    if are_equal:
        print(f"✅ PASSED: Tensors are identical (within tolerance {tolerance})")
    else:
        print(f"❌ FAILED: Tensors differ!")

        # Calculate differences
        abs_diff = np.abs(arr1 - arr2)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        print(f"\nDifference statistics:")
        print(f"  Max absolute difference: {max_diff:.10f}")
        print(f"  Mean absolute difference: {mean_diff:.10f}")
        print(f"  Number of differing elements: {np.sum(abs_diff > tolerance)}")

        # Show sample values
        print(f"\nSample values (first 5 pixels):")
        print(f"  {name1}: {arr1.flatten()[:5]}")
        print(f"  {name2}: {arr2.flatten()[:5]}")

    # Show statistics
    print(f"\n{name1} statistics:")
    print(f"  Min: {arr1.min():.6f}")
    print(f"  Max: {arr1.max():.6f}")
    print(f"  Mean: {arr1.mean():.6f}")
    print(f"  Std: {arr1.std():.6f}")

    print(f"\n{name2} statistics:")
    print(f"  Min: {arr2.min():.6f}")
    print(f"  Max: {arr2.max():.6f}")
    print(f"  Mean: {arr2.mean():.6f}")
    print(f"  Std: {arr2.std():.6f}")

    return are_equal


def main():
    parser = argparse.ArgumentParser(
        description="Test preprocessing parity between notebook and Django backend"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to a sample image for testing"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for floating point comparison (default: 1e-6)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PREPROCESSING PARITY TEST")
    print("=" * 70)
    print(f"\nTesting with image: {args.image}")
    print(f"Tolerance: {args.tolerance}")

    try:
        # Apply notebook preprocessing
        print("\n[1/2] Applying notebook preprocessing...")
        notebook_tensor = notebook_preprocess(args.image)
        print(f"     Result shape: {notebook_tensor.shape}")

        # Apply backend preprocessing
        print("\n[2/2] Applying Django backend preprocessing...")
        backend_tensor = backend_preprocess(args.image)
        print(f"     Result shape: {backend_tensor.shape}")

        # Compare tensors
        parity_passed = compare_tensors(
            notebook_tensor,
            backend_tensor,
            name1="Notebook preprocessing",
            name2="Backend preprocessing",
            tolerance=args.tolerance,
        )

        print("\n" + "=" * 70)
        if parity_passed:
            print("✅ PARITY TEST PASSED!")
            print(
                "The backend preprocessing exactly matches the notebook preprocessing."
            )
            print("You can safely use this Django application for predictions.")
        else:
            print("❌ PARITY TEST FAILED!")
            print("The preprocessing differs between notebook and backend.")
            print("Please review the preprocessing implementation in modelapp/model.py")
        print("=" * 70 + "\n")

        sys.exit(0 if parity_passed else 1)

    except FileNotFoundError as e:
        print(f"\n❌ Error: Image file not found: {args.image}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during parity test: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
