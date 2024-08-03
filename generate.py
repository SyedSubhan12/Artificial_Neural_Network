import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import os

# Load dataset
(x_train, _), (_, _) = fashion_mnist.load_data()

# Directory to save images
IMAGE_DIR = './static/images'
os.makedirs(IMAGE_DIR, exist_ok=True)

def save_image(data, file_name):
    plt.imsave(os.path.join(IMAGE_DIR, file_name), data, cmap='gray')

# Save a few sample images
for i in range(10):
    save_image(x_train[i], f'image{i}.png')