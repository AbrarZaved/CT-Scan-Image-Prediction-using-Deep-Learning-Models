# Medical Image Classification - Disease Prediction System

A production-ready Django application for medical image classification using PyTorch and Transfer Learning (ResNet50). This system predicts diseases from medical scan images across 9 different categories covering brain, kidney, and lung conditions.

## 🎯 Features

- **Transfer Learning ResNet50** model trained on medical imaging data
- **REST API** for programmatic access (`POST /api/predict/`)
- **Web UI** with Tailwind CSS for easy image upload and prediction
- **Exact preprocessing pipeline** matching training notebook
- **Multi-format support**: Multipart file upload and Base64 JSON input
- **9 disease categories**: Brain, Kidney, and Lung conditions
- **CPU-only inference** for deployment flexibility

## 📋 Supported Conditions

The model can detect the following 9 conditions:

1. Brain Healthy
2. Brain Tumor
3. Kidney Cyst
4. Kidney Normal
5. Kidney Stone
6. Kidney Tumor
7. Lung adenocarcinoma
8. Lung Malignant cases
9. Lung normal

## 🔧 Prerequisites

- Python 3.8 or higher
- pip package manager
- The trained model file: `tl_resnet50_best.pt`

## 🚀 Installation & Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the model file

**CRITICAL:** Copy `tl_resnet50_best.pt` to the project root directory (same folder as `manage.py`).

```
disease_prediction/
├── manage.py
├── tl_resnet50_best.pt  ← Place model file here
├── project/
├── modelapp/
└── ...
```

If you want to use a different location, update the `MODEL_PATH` setting in `project/settings.py`:

```python
MODEL_PATH = '/path/to/your/tl_resnet50_best.pt'
```

### 4. Run database migrations

```bash
python manage.py migrate
```

### 5. (Optional but Recommended) Verify preprocessing parity

Before deploying, verify that the backend preprocessing exactly matches the training preprocessing:

```bash
# Use a sample image from your dataset
python preprocess_parity_test.py --image "path/to/sample_image.jpg"

# Example with actual path
python preprocess_parity_test.py --image "F:\Sorted Data set\Brain Tumor\ct_tumor (2).png"
```

Expected output:
```
✅ PARITY TEST PASSED!
The backend preprocessing exactly matches the notebook preprocessing.
```

If the test fails, do not proceed to production. Review the preprocessing implementation.

### 6. Start the development server

```bash
python manage.py runserver
```

The application will be available at: `http://127.0.0.1:8000/`

## 📖 Usage

### Web UI

1. Open your browser and navigate to `http://127.0.0.1:8000/`
2. Click "Click to upload" or drag and drop a medical image
3. Click "Analyze Image"
4. View the prediction result with confidence score

### REST API

#### Endpoint: `POST /api/predict/`

**Option 1: Multipart file upload**

```bash
curl -X POST -F "image=@path/to/image.jpg" http://127.0.0.1:8000/api/predict/
```

Example:
```bash
curl -X POST -F "image=@test_brain_tumor.png" http://127.0.0.1:8000/api/predict/
```

**Option 2: Base64 JSON**

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}'
```

**Response format:**

```json
{
  "prediction": "Brain Tumor",
  "accuracy_score": 97.5
}
```

**Error responses:**

```json
{
  "error": "Invalid image file: cannot identify image file"
}
```

Status codes:
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input (missing image, invalid format)
- `500 Internal Server Error`: Server error (model not found, prediction failed)

### Python Client Example

```python
import requests

# Multipart upload
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/api/predict/',
        files={'image': f}
    )
    print(response.json())

# Base64 JSON
import base64

with open('test_image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://127.0.0.1:8000/api/predict/',
    json={'image_base64': image_base64}
)
print(response.json())
```

## 🧪 Testing

### Test preprocessing parity

```bash
python preprocess_parity_test.py --image path/to/test_image.jpg
```

Options:
- `--image`: Path to test image (required)
- `--tolerance`: Floating point comparison tolerance (default: 1e-6)

### Test API endpoint

```bash
# Test with a sample image
curl -X POST -F "image=@test_sample.png" http://127.0.0.1:8000/api/predict/

# Expected output:
# {"prediction":"Brain Tumor","accuracy_score":97.5}
```

## 📁 Project Structure

```
disease_prediction/
├── manage.py                      # Django management script
├── tl_resnet50_best.pt           # Trained model weights (place here)
├── requirements.txt              # Python dependencies
├── preprocess_parity_test.py     # Preprocessing validation script
├── README.md                     # This file
│
├── project/                      # Django project configuration
│   ├── __init__.py
│   ├── settings.py              # Django settings
│   ├── urls.py                  # Root URL configuration
│   └── wsgi.py                  # WSGI configuration
│
└── modelapp/                     # Main application
    ├── __init__.py
    ├── apps.py
    ├── model.py                 # Model architecture & preprocessing
    ├── views.py                 # Web UI and API views
    ├── urls.py                  # App URL configuration
    ├── serializers.py           # DRF serializers
    ├── utils.py                 # Utility functions
    └── templates/
        └── modelapp/
            └── upload.html      # Web UI template
```

## 🔬 Preprocessing Pipeline

The application uses the **exact** preprocessing pipeline from the training notebook:

1. **Grayscale conversion**: Images are converted to single-channel grayscale
2. **Resize**: Images are resized to 224×224 pixels using OpenCV
3. **Normalization**: Pixel values are normalized with `mean=0.5`, `std=0.5`
4. **Tensor conversion**: Images are converted to PyTorch tensors with shape `(1, 1, 224, 224)`

This preprocessing is implemented in `modelapp/model.py::preprocess_image()` and matches:
- Notebook cell 2: `val_transform` using Albumentations
- Dataset class: `CTFolderDataset` with `gray=True`, `extra_preproc=False`

**Verification**: Always run `preprocess_parity_test.py` before production deployment.

## 🏗️ Model Architecture

- **Base**: ResNet50 (Transfer Learning)
- **Input channels**: 1 (grayscale)
- **Output classes**: 9
- **Framework**: PyTorch with `timm` library
- **Training**: Partial layer freezing (60% frozen for regularization)

The model is built using `timm.create_model()` and matches the training configuration in notebook cell 4.

## ⚙️ Configuration

### Django Settings (`project/settings.py`)

```python
# Model file path
MODEL_PATH = os.path.join(BASE_DIR, 'tl_resnet50_best.pt')

# Change to custom path if needed
# MODEL_PATH = '/absolute/path/to/model.pt'
```

### Model Settings (`modelapp/model.py`)

```python
# Class names (do not modify order)
CLASS_NAMES = [
    "Brain Healthy",
    "Brain Tumor",
    "Kidney Cyst",
    # ... etc
]

# Preprocessing constants
IMG_SIZE = 224
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
```

## 🚨 Important Notes

### ⚠️ Medical Disclaimer

**This application is for research and educational purposes only.** The predictions are generated by an AI model and should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment.

### 🔒 Production Deployment Checklist

Before deploying to production:

- [ ] Run preprocessing parity test and confirm PASS
- [ ] Change `SECRET_KEY` in `settings.py`
- [ ] Set `DEBUG = False` in `settings.py`
- [ ] Configure `ALLOWED_HOSTS` properly
- [ ] Use a production-grade database (PostgreSQL/MySQL instead of SQLite)
- [ ] Set up static file serving with `collectstatic`
- [ ] Use a production WSGI server (gunicorn/uWSGI)
- [ ] Configure HTTPS/TLS
- [ ] Add authentication/authorization for API if needed
- [ ] Set up monitoring and logging
- [ ] Test with diverse medical images from your dataset

### 🐛 Troubleshooting

**Model file not found error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'tl_resnet50_best.pt'
```
Solution: Ensure `tl_resnet50_best.pt` is in the project root directory (same folder as `manage.py`).

**Import errors:**
```
ModuleNotFoundError: No module named 'timm'
```
Solution: Activate virtual environment and run `pip install -r requirements.txt`

**CUDA out of memory (if using GPU):**
Solution: This application is configured for CPU inference. If you modify for GPU, reduce batch size or use a smaller model.

**Preprocessing parity test fails:**
Solution: Do not proceed to production. Review `modelapp/model.py::preprocess_image()` and compare with notebook cell 2 `val_transform`.

## 📚 Dependencies

Key packages:
- **Django 4.2.7**: Web framework
- **djangorestframework 3.14.0**: REST API
- **torch 2.1.0**: Deep learning framework
- **torchvision 0.16.0**: Vision utilities
- **timm 0.9.12**: Model architectures
- **opencv-python 4.8.1**: Image processing
- **albumentations 1.3.1**: Image augmentation
- **Pillow 10.1.0**: Image I/O

See `requirements.txt` for complete list.

## 📝 License

This project is provided as-is for research and educational purposes.

## 👥 Support

For issues or questions:
1. Verify model file is in correct location
2. Run preprocessing parity test
3. Check Django error logs
4. Review API error responses (400/500 status codes)

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Model**: tl_resnet50_best.pt (Transfer Learning ResNet50)  
**Python**: 3.8+  
**Platform**: CPU-only (cross-platform)
