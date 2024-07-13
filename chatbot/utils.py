import tensorflow as tf # type: ignore
import os
from django.conf import settings # type: ignore

def load_legalChatbot_model():
    try:
        model_path = os.path.join(settings.BASE_DIR, 'chatbot/static/model', 'legalChatbotModel.keras')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file {model_path} does not exist.")

        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise ValueError(f"File not found: Temba filepath={model_path}. Please ensure the file is an accessible `.keras` zip file.")

