from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import RegisterAPIView, LoginAPIView, ChatbotViewSet, UploadPDFView, ChatAPIView

# APIs for chatbot creation with the chatbot name
router = DefaultRouter()
router.register(r'chatbots', ChatbotViewSet)

urlpatterns = [
    # API for user Registration
    path('register/', RegisterAPIView.as_view(), name='register'),

    # API for user Login
    path('login/', LoginAPIView.as_view(), name='login'),

    # API for pdf upload with chatbot name
    path('upload-pdf/', UploadPDFView.as_view(), name='upload-pdf'),

    # API for chatbot have to give a name
    path('chat-api/', ChatAPIView.as_view(), name='chat-api'),

    path('', include(router.urls)),
]
