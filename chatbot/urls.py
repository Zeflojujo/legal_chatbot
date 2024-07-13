from django.urls import path
from .views import chatbot_response

urlpatterns = [
    path('api/chatbot', chatbot_response, name='chatbot_response'),
]