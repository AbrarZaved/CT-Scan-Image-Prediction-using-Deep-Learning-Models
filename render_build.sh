#!/bin/bash
# Render deployment script for Django Disease Prediction

echo "🚀 Starting Render deployment..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --no-input

# Run database migrations
echo "🗄️ Running migrations..."
python manage.py migrate --no-input

echo "✅ Deployment script completed successfully!"
