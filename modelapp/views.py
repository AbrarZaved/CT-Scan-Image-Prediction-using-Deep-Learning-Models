"""
Views for web UI and REST API.
"""

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from .serializers import PredictionSerializer, PredictionResponseSerializer
from .model import predict_image_bytes
from .utils import validate_image


def upload_view(request):
    """
    Web UI view for image upload and prediction.
    """
    if request.method == "POST" and request.FILES.get("image"):
        try:
            # Get uploaded file
            image_file = request.FILES["image"]
            image_bytes = image_file.read()

            # Validate image
            validate_image(image_bytes)

            # Get prediction
            model_path = getattr(settings, "MODEL_PATH", "tl_resnet50_best.pt")
            predicted_class, confidence = predict_image_bytes(
                image_bytes, model_path=model_path
            )

            return render(
                request,
                "modelapp/upload.html",
                {
                    "prediction": predicted_class,
                    "confidence": round(confidence, 2),
                    "success": True,
                },
            )

        except Exception as e:
            return render(
                request, "modelapp/upload.html", {"error": str(e), "success": False}
            )

    return render(request, "modelapp/upload.html")


class PredictAPIView(APIView):
    """
    REST API view for image prediction.

    Accepts:
    - POST /api/predict/ with multipart form data (field name: 'image')
    - POST /api/predict/ with JSON {"image_base64": "base64_encoded_image_data"}

    Returns:
    - {"prediction": "Brain Tumor", "accuracy_score": 97.5}
    """

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, *args, **kwargs):
        # Validate input
        serializer = PredictionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get image bytes
            image_bytes = serializer.get_image_bytes()

            if not image_bytes:
                return Response(
                    {"error": "No image data provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Validate image
            validate_image(image_bytes)

            # Get prediction
            model_path = getattr(settings, "MODEL_PATH", "tl_resnet50_best.pt")
            predicted_class, confidence = predict_image_bytes(
                image_bytes, model_path=model_path
            )

            # Return response
            response_data = {
                "prediction": predicted_class,
                "accuracy_score": round(confidence, 2),
            }

            response_serializer = PredictionResponseSerializer(data=response_data)
            response_serializer.is_valid(raise_exception=True)

            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        except FileNotFoundError as e:
            return Response(
                {
                    "error": f"Model file not found. Please ensure tl_resnet50_best.pt is in the project root."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        except Exception as e:
            return Response(
                {"error": f"Prediction failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
