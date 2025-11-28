import tensorflow as tf
import numpy as np
from PIL import Image

def process_image(image_np):
    image = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()

def load_and_process(image_path):
    image = Image.open(image_path)
    image_np = np.asarray(image)
    processed = process_image(image_np)

    return np.expand_dims(processed, axis=0)

