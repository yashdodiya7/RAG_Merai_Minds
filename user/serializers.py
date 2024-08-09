from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from user.models import User, Chatbot


class UserRegistrationSerializer(serializers.ModelSerializer):
    password2 = serializers.CharField(style={'input_type': 'password'}, write_only=True)

    class Meta:
        model = User
        fields = "__all__"
        extra_kwargs = {
            'password': {'write_only': True}
        }

    def validate(self, attrs):
        password = attrs['password'],
        password2 = attrs['password2'],
        if password != password2:
            raise serializers.ValidationError("password and confirm password not match")
        return attrs

    def validate_password(self, value):
        validate_password(value)
        return value

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class UserLoginSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(max_length=200)

    class Meta:
        model = User
        fields = ['id', 'email', 'password']


class ChatbotSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chatbot
        fields = ['id', 'user', 'name', 'description', 'created_at']
        read_only_fields = ['user']


class PDFUploadSerializer(serializers.Serializer):
    pdf = serializers.FileField(required=True)
