import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json

def get_class_names(json_file="label_map.json"):
    with open(json_file, 'r') as f:
        class_names = json.load(f) 
    return class_names


def load_model(model_path="./model.h5"):
    model = tf.keras.models.load_model(model_path,
    custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def get_processed_image(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    return process_image(image)

def process_image(img, size=224):
    image = np.squeeze(img)
    image = tf.image.resize(image, (size, size))/255.0
    return image