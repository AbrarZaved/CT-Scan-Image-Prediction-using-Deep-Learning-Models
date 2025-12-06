"""
Serializers for the prediction API.
"""

from rest_framework import serializers
import base64
from io import BytesIO
from PIL import Image


class PredictionSerializer(serializers.Serializer):
    """
    Serializer for prediction requests.
    Accepts either:
    - multipart file upload with field name 'image'
    - JSON with base64 encoded image in 'image_base64' field
    """

    image = serializers.ImageField(required=False, allow_null=True)
    image_base64 = serializers.CharField(required=False, allow_blank=True)

    def validate(self, data):
        """
        Ensure at least one image input method is provided.
        """
        if not data.get("image") and not data.get("image_base64"):
            raise serializers.ValidationError(
                "Either 'image' file or 'image_base64' string must be provided."
            )
        return data

    def get_image_bytes(self):
        """
        Extract image bytes from either multipart or base64 input.
        """
        validated_data = self.validated_data

        if validated_data.get("image"):
            # Multipart file upload
            image_file = validated_data["image"]
            return image_file.read()

        elif validated_data.get("image_base64"):
            # Base64 encoded image
            base64_str = validated_data["image_base64"]

            # Remove data URI prefix if present (e.g., "data:image/png;base64,")
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            try:
                image_bytes = base64.b64decode(base64_str)
                return image_bytes
            except Exception as e:
                raise serializers.ValidationError(f"Invalid base64 image: {str(e)}")

        return None


class PredictionResponseSerializer(serializers.Serializer):
    """
    Serializer for prediction responses.
    """

    prediction = serializers.CharField()
    accuracy_score = serializers.FloatField()
