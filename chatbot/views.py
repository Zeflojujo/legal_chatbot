# from django.shortcuts import render # type: ignore
# from django.http import JsonResponse # type: ignore
# from rest_framework.decorators import api_view # type: ignore
# import tensorflow as tf # type: ignore
# import numpy as np # type: ignore
# import json
# from .utils import load_legalChatbot_model

# # Load  trained model
# model = load_legalChatbot_model()

# @api_view(['POST'])
# def chatbot_response(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         input_text = data.get('text', '')

#         # Preprocess the input text
#         preprocessed_input = preprocess_input(input_text)
        
#         # Generate response using the model
#         response = model.predict(preprocessed_input)
        
#         # Postprocess the response if needed
#         postprocessed_response = postprocess_response(response)

#         return JsonResponse({'response': postprocessed_response})
#     else:
#         return JsonResponse({'error': 'Invalid request method'}, status=400)

# def preprocess_input(text):
#     # Implement your preprocessing logic here
#     # Tokenize, pad sequences, etc.
#     return preprocessed_text

# def postprocess_response(response):
#     # Implement your postprocessing logic here
#     # Convert numerical response to text
#     return response_text


from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import tensorflow as tf
import numpy as np
import json
from .utils import load_legalChatbot_model

# Load trained model
model = load_legalChatbot_model()

@api_view(['POST'])
def chatbot_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_text = data.get('text', '')

        # Preprocess the input text
        preprocessed_input = preprocess_input(input_text)
        
        # Generate response using the model
        response = model.predict(preprocessed_input)
        
        # Postprocess the response if needed
        postprocessed_response = postprocess_response(response)

        return JsonResponse({'response': postprocessed_response})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

def preprocess_input(text):
    # Implement your preprocessing logic here
    # Tokenize, pad sequences, etc.
    return np.array([text])  # Example, update according to your preprocessing logic

def postprocess_response(response):
    # Implement your postprocessing logic here
    # Convert numerical response to text
    return str(response)  # Example, update according to your postprocessing logic
