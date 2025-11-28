import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from utils import load_and_process

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

def predict(image_path, model, top_k):
    processed_image = load_and_process(image_path)
    predictions = model.predict(processed_image)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_classes = [str(i) for i in top_indices]
    return top_probs, top_classes

def main():
    parser = argparse.ArgumentParser(description='Flower Image Classifier')
    
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('model_path', type=str, help='Path to saved model (h5)')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to label_map.json')

    args = parser.parse_args()

    model = load_model(args.model_path)
    probs, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        classes = [class_names[c] for c in classes]

    print("\n=== Prediction Results ===")
    for p, c in zip(probs, classes):
        print(f"{c}: {p:.4f}")

if __name__ == "__main__":
    main()
