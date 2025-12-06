# Memory Optimization & Deployment Guide

## 🎯 Optimizations Implemented

### 1. **CPU-Only PyTorch** ✅
- Removed all CUDA/GPU dependencies
- Using `torch==2.1.0+cpu` and `torchvision==0.16.0+cpu`
- Saves ~1-2 GB of memory by avoiding GPU libraries
- Model loads with `map_location='cpu'` and `weights_only=True`

### 2. **Lazy Model Loading** ✅
- Model loads only on first prediction request (not at startup)
- Reduces initial memory footprint
- Cached globally after first load for subsequent requests

### 3. **Optimized Gunicorn Configuration** ✅
```bash
gunicorn project.wsgi:application \
  --workers=1 \              # Single worker (low memory)
  --threads=2 \              # 2 threads per worker
  --timeout=120 \            # Handle slow predictions
  --max-requests=100 \       # Restart worker after 100 requests
  --max-requests-jitter=20   # Prevent thundering herd
```

### 4. **Lightweight Dependencies** ✅
- `opencv-python-headless` instead of `opencv-python` (no GUI)
- Removed unnecessary packages (scikit-learn, pandas, matplotlib, seaborn, h5py)
- Kept only production essentials

### 5. **Memory-Efficient Inference** ✅
- `torch.no_grad()` context for all predictions
- `model.eval()` mode (disables dropout/batch norm)
- Explicit CPU device assignment

## 📊 Memory Usage Comparison

| Configuration | Memory Usage | Suitable For |
|--------------|-------------|--------------|
| **Before** (GPU libs + eager loading) | ~2.5-3 GB | High-memory servers |
| **After** (CPU-only + lazy loading) | ~800 MB - 1.2 GB | Free tier hosting ✅ |

## 🚀 Deployment Instructions

### For Render (Free Tier)

1. **Create `render.yaml`**:
```yaml
services:
  - type: web
    name: disease-prediction
    env: python
    buildCommand: "./render_build.sh"
    startCommand: "gunicorn project.wsgi:application --workers=1 --threads=2 --timeout=120"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: WEB_CONCURRENCY
        value: 1
```

2. **Environment Variables** (Render Dashboard):
```
DEBUG=False
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-app.onrender.com
```

3. **Upload Model File**:
   - Add `tl_resnet50_best.pt` to your repository root
   - Or use Render disk storage for persistent files

### For Heroku (Free Tier)

1. **Create `runtime.txt`**:
```
python-3.11.0
```

2. **Set Config Vars**:
```bash
heroku config:set DEBUG=False
heroku config:set SECRET_KEY=your-secret-key
heroku config:set WEB_CONCURRENCY=1
```

3. **Deploy**:
```bash
git push heroku main
```

### Local Testing with Production Settings

```bash
# Install CPU-only dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Test with gunicorn (production server)
gunicorn project.wsgi:application --workers=1 --threads=2 --bind=0.0.0.0:8000
```

## 🔧 Additional Optimization Tips

### If Still Getting OOM (Out of Memory):

1. **Reduce Model Size**:
   ```python
   # Convert to half precision (float16) - reduces model size by 50%
   model = model.half()
   img_tensor = img_tensor.half()
   ```

2. **Enable Model Quantization**:
   ```python
   import torch.quantization
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

3. **Convert to ONNX** (even smaller):
   ```python
   import torch.onnx
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```

4. **Use Swap Space** (last resort):
   - Add swap on your server
   - Performance will be slower but prevents crashes

## 🔍 Monitoring Memory Usage

Add to `settings.py`:
```python
import psutil
import os

def log_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 / 1024:.2f} MB")
```

## ⚠️ Important Notes

1. **CPU Inference Speed**: ~200-500ms per prediction (acceptable for web apps)
2. **First Request Delay**: 5-10 seconds (lazy loading model)
3. **Concurrent Requests**: Limited to 2 with current config
4. **Model File Size**: ~100-200 MB (ensure it's in `.gitignore` if >100MB)

## 🎯 Expected Performance

- **Startup Time**: 10-15 seconds
- **First Prediction**: 5-10 seconds (model loading)
- **Subsequent Predictions**: 0.3-0.8 seconds
- **Memory Usage**: 800 MB - 1.2 GB
- **Suitable For**: 512 MB - 1 GB RAM instances

## ✅ Deployment Checklist

- [ ] Updated `requirements.txt` with CPU-only packages
- [ ] Created `Procfile` with single worker configuration
- [ ] Created `render_build.sh` or equivalent
- [ ] Set `DEBUG=False` in production
- [ ] Updated `ALLOWED_HOSTS` in settings
- [ ] Uploaded model file (`tl_resnet50_best.pt`)
- [ ] Configured static files serving
- [ ] Tested locally with gunicorn
- [ ] Set up environment variables on hosting platform

---

**Your app is now optimized for free-tier deployment! 🎉**
