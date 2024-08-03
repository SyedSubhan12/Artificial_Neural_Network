import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import random

# Define the dictionary for class labels
class_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Load the trained model
model_path = 'D:/Artificial Neural Network/Image Classification/fashion_mnist_model.h5'
model = load_model(model_path)

# Directory where images are stored
IMAGE_DIR = 'D:/Artificial Neural Network/Image Classification/static/images'

# List all image files in the directory
image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

# Select a random image
random_image_file = random.choice(image_files)
img_path = os.path.join(IMAGE_DIR, random_image_file)

# Load and preprocess the image
img = Image.open(img_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize to match model input
img_array = np.array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

# Predict the class
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

# Open the image
img.show()

# Print the result
print(f'Image: {random_image_file}')
print(f'Predicted Class Number: {predicted_class}')
print(f'Predicted Class Label: {class_labels[predicted_class]}')
