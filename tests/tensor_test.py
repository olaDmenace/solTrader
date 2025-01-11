# type: ignore


import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Test specific imports
from tensorflow import keras
print("Keras version:", keras.__version__)

from tensorflow.keras import layers
print("Successfully imported layers")

from tensorflow.keras import callbacks
print("Successfully imported callbacks")

from tensorflow.keras import models 
print("Successfully imported models")