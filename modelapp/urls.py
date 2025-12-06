"""
URL configuration for modelapp.
"""

from django.urls import path
from .views import upload_view, PredictAPIView

urlpatterns = [
    path("", upload_view, name="upload"),
    path("api/predict/", PredictAPIView.as_view(), name="api_predict"),
]
