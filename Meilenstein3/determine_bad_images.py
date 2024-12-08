from pathlib import Path
import json
import numpy as np
import argparse
import tensorflow as tf
import train as t
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data/predict", type=Path)
    parser.add_argument("-l", "--labels", default="data/labels/test", type=Path)
    parser.add_argument("-m", "--model")
    return parser.parse_args()

def load_data(img_dir: Path, label_dir: Path):
    """
    Loads images and their corresponding age labels from the specified directories.

    Args:
        img_dir (Path): Directory containing image files.
        label_dir (Path): Directory containing JSON label files.

    Returns:
        tuple: A tuple containing:
            - paths (list of Path): List of paths to the image files.
            - age_labels (list of int): List of age labels corresponding to the images.
            - images (numpy.ndarray): Array of preprocessed images.

    Raises:
        Exception: If there is an error loading a label file or processing an image.
    """
    print("Loading images...")
    paths, age_labels, images = [], [], []
    for node in img_dir.iterdir():
        if node.name.endswith("png") or node.name.endswith("jpg"):
            try:
                with open(label_dir / f"{node.stem}.json") as file:
                    label = json.load(file)
                    age = label[0]["faceAttributes"]["age"]
                    paths.append(node)
                    age_labels.append(age)
                    images.append(t.load_and_preprocess_image(str(node)))
            except Exception as e:
                print(e)
    return paths, age_labels, np.vstack(images)

def predict(images, model):
    """
    Predicts the ages for the given images using the provided model.

    Args:
        images (numpy.ndarray): Array of preprocessed images.
        model (tf.keras.Model): The trained model to use for predictions.

    Returns:
        list of float: List of predicted ages.
    """
    print("Predicting...")
    predictions = model.predict(images)
    return [p[0] for p in predictions[1]]

def load_model(model_path):
    """
    Loads a trained model from the specified path.

    Args:
        model_path (str): Path to the trained model file.

    Returns:
        tf.keras.Model: The loaded model.
    """
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy,
            "custom_age_loss": t.custom_age_loss,
            "custom_gender_loss": t.custom_gender_loss,
            "custom_age_metric": t.custom_age_metric,
            "custom_gender_metric": t.custom_gender_metric,
        },
    )

def determine_bad_images(paths, labels, predictions):
    """
    Identifies and copies images with a large prediction error.

    Args:
        paths (list of Path): List of paths to the image files.
        labels (list of int): List of actual age labels.
        predictions (list of float): List of predicted ages.

    Raises:
        ValueError: If the length of labels and predictions do not match.
    """
    print("Calculating MAE per age...")
    if len(labels) != len(predictions):
        raise ValueError("The length of actual_ages and predicted_ages must be the same.")
    
    for file, label, prediction in zip(paths, labels, predictions):
        diff = abs(label - prediction)
        if diff > 10:
            shutil.copy(str(file), f"data/finn_bad_images/{label}_{prediction}_{file.name}")

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model)
    paths, labels, images = load_data(args.data, args.labels)
    predictions = predict(images, model)
    determine_bad_images(paths, labels, predictions)