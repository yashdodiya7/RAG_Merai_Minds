from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import RegisterAPIView, LoginAPIView, ChatbotViewSet, UploadPDFView, ChatAPIView

router = DefaultRouter()
router.register(r'chatbots', ChatbotViewSet)

urlpatterns = [
    path('register/', RegisterAPIView.as_view(), name='register'),
    path('login/', LoginAPIView.as_view(), name='login'),
    path('upload-pdf/', UploadPDFView.as_view(), name='upload-pdf'),
    path('chat-api/', ChatAPIView.as_view(), name='chat-api'),
    path('', include(router.urls)),
]
