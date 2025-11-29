
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from utils import load_and_process

def predict(image_path, model, top_k):
    
    
    processed_image = load_and_process(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)

    preds = model.predict(processed_image)[0] 
    top_k_indices = np.argsort(preds)[-top_k:][::-1]
    top_k_probs = preds[top_k_indices]

    
    corrected_classes = [str(i + 1) for i in top_k_indices]

    return top_k_probs, corrected_classes


def load_model(model_path):
    
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )


def main():
    parser = argparse.ArgumentParser(description="Flower classifier")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("saved_model", help="Path to saved model (H5)")
    parser.add_argument("--top_k", type=int, default=1, help="Top K predictions")
    parser.add_argument("--category_names", help="Path to label_map.json")

    args = parser.parse_args()

    model = load_model(args.saved_model)

    probs, classes = predict(args.image_path, model, args.top_k)

    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)

        named_classes = [class_names[c] for c in classes]

        print("\n=== Prediction Results ===")
        for name, prob in zip(named_classes, probs):
            print(f"{name}: {prob:.4f}")

    else:
        print("\n=== Prediction Results ===")
        for cls, prob in zip(classes, probs):
            print(f"{cls}: {prob:.4f}")


if __name__ == "__main__":
    main()
