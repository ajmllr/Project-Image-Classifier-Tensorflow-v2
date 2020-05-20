import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
from utils import process_image, get_processed_image, load_model, get_class_names

import argparse
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def predict(image_path, model, top_k, category_names, dev=False):
    if dev:
        print(image_path, model, top_k, category_names, dev)

    image = get_processed_image(image_path)

    model = load_model(model)

    class_names = get_class_names(category_names)

    prediction = model.predict(np.expand_dims(image, axis=0))

    values, indices = tf.math.top_k(prediction, top_k)
    values = values.numpy()[0]

    classes = [class_names[str(value)] for value in indices.cpu().numpy()[0]]
    
    if dev:
        print("model.summary")
        print(model.summary())

    print(f'top class: {classes[0]} with % {values[0]*100}')
    print(f'values: {values}\nclasses: {classes}')

    return values, classes

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument ('image_path', default='./test_images/wild_pansy.jpg', help='Path to image.', type=str)
    parser.add_argument('model', default='./my_model.h5', help='Path to model as *.h5', type=str)
    parser.add_argument ('--top_k', default=5, help='Top classes to return.', type=int)
    parser.add_argument ('--category_names', default='label_map.json', help='Path to labels map as *.json', type=str)
    parser.add_argument ('--dev', default=False, help='True / False prints', type=bool)
    args = parser.parse_args()

    # python predict.py './test_images/orange_dahlia.jpg' './my_model.h5'
    # python predict.py './test_images/orange_dahlia.jpg' './my_model.h5' --dev True
    # python predict.py './test_images/wild_pansy.jpg' './my_model.h5' --top_k 3 --dev True
    
    predict(args.image_path, args.model, args.top_k, args.category_names, args.dev)