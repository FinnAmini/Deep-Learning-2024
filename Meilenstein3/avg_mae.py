from pathlib import Path
import json
import numpy as np
import argparse
from collections import defaultdict
import tensorflow as tf
import train as t
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the following attributes:
            - data (Path): Path to the data directory (default: "data/predict").
            - labels (Path): Path to the labels directory (default: "data/labels/test").
            - model (str): Path to the model file.
            - hybrid_model (str): Path to the hybrid model file.
            - gan_model (str): Path to the GAN model file.
            - name (str): Name for the current run or experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data/predict", type=Path)
    parser.add_argument("-l", "--labels", default="data/labels/test", type=Path)
    parser.add_argument("-m", "--model")
    parser.add_argument("-hm", "--hybrid-model")
    parser.add_argument("-gm", "--gan-model")
    parser.add_argument("-n", "--name")
    return parser.parse_args()

def load_data(img_dir: Path, label_dir: Path):
    """
    Loads images and their corresponding age and gender labels from the specified directories.

    Args:
        img_dir (Path): Directory containing image files.
        label_dir (Path): Directory containing JSON label files.

    Returns:
        tuple: A tuple containing:
            - A tuple of two lists:
                - age_labels (list): List of ages extracted from the labels.
                - gender_labels (list): List of gender labels (0 for male, 1 for female).
            - numpy.ndarray: Array of preprocessed images.

    Raises:
        Exception: If there is an error reading a label file or processing an image.
    """
    print("Loading images...")
    age_labels, gender_labels, images = [], [], []
    for node in img_dir.iterdir():
        if node.name.endswith("png") or node.name.endswith("jpg"):
            try:
                with open(label_dir / f"{node.stem}.json") as file:
                    label = json.load(file)
                    age = label[0]["faceAttributes"]["age"]
                    gender = label[0]["faceAttributes"]["gender"]
                    age_labels.append(age)
                    gender_labels.append(0 if gender == 'male' else 1)
                    images.append(t.load_and_preprocess_image(str(node)))
            except Exception as e:
                print(e)
    return (age_labels, gender_labels), np.vstack(images)


def predict(images, model):
    """
    Generate predictions for age and gender using the provided model.

    Parameters:
    images (numpy.ndarray): Array of preprocessed images.
    model (tf.keras.Model): The trained Keras model for prediction.

    Returns:
    tuple: Tuple containing two lists - predicted ages and predicted genders.
    """
    print("Predicting...")
    predictions = model.predict(images)
    age = [p[0] for p in predictions[1]]
    gender = [0 if p[0] < 0.5 else 1 for p in predictions[2]]
    return age, gender

def load_model(model_path):
    """
    Load a Keras model from the specified file path with custom objects.

    Args:
        model_path (str): The file path to the saved Keras model.

    Returns:
        tf.keras.Model: The loaded Keras model with custom objects.
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

def mae_per_age(predictions, labels):
    """
    Calculate the Mean Absolute Error (MAE) per age and gender accuracy.

    Parameters:
    predictions (tuple): Tuple containing predicted ages and genders.
    labels (tuple): Tuple containing actual ages and genders.

    Returns:
    dict: Dictionary with age as keys and MAE as values.
    """
    age_pred, gender_pred = predictions
    age_labels, gender_labels = labels
    print("Calculating MAE per age...")
    if len(age_labels) != len(age_pred):
        raise ValueError("The length of actual_ages and predicted_ages must be the same.")
    
    diff_dict = defaultdict(lambda: {"total_diff": 0, "count": 0})
    total_age_diff, age_count = 0, len(age_pred)
    correct_gender, gender_count = 0, len(gender_pred)
    
    for actual_age, actual_gender, predicted_age, predicted_gender in zip(age_labels, gender_labels, age_pred, gender_pred):
        diff = abs(actual_age - predicted_age)
        diff_dict[actual_age]["total_diff"] += diff
        diff_dict[actual_age]["count"] += 1
        total_age_diff += diff
        correct_gender += 1 if predicted_gender == actual_gender else 0

    print(f"Age MAE: {total_age_diff / age_count}")
    print(f"Gender Accuracy: {correct_gender / gender_count}")
    
    average_diff_dict = {
        age: diff_data["total_diff"] / diff_data["count"] 
        for age, diff_data in diff_dict.items()
    }
    
    return average_diff_dict

def visualize(data, hybrid_data=None, gan_data=None, name=""):    
    """
    Visualizes the Mean Absolute Error (MAE) by age for different datasets.

    Parameters:
    data (dict): Dictionary containing the normal data with age as keys and MAE as values.
    hybrid_data (dict, optional): Dictionary containing the hybrid data with age as keys and MAE as values. Defaults to None.
    gan_data (dict, optional): Dictionary containing the GAN data with age as keys and MAE as values. Defaults to None.
    name (str, optional): Name to be used for the saved plot file. Defaults to an empty string.

    Returns:
    None: The function saves the plot as a PNG file with the specified name.
    """
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

    normal_df = pd.DataFrame(list(data.items()), columns=['age', 'mae'])
    sns.lineplot(data=normal_df, x='age', y='mae', label='Normal Data')

    if hybrid_data:
        hybrid_df = pd.DataFrame(list(hybrid_data.items()), columns=['age', 'mae'])
        sns.lineplot(data=hybrid_df, x='age', y='mae', label='Hybrid Data')

    if gan_data:
        gan_df = pd.DataFrame(list(gan_data.items()), columns=['age', 'mae'])
        sns.lineplot(data=gan_df, x='age', y='mae', label='GAN Data')

    plt.title('Average MAE by Age')
    plt.xlabel('Age')
    plt.ylabel('MAE')
    plt.ylim(0, None)
    plt.legend(title='Training-Data')
    plt.savefig(f"avg_mae_per_age_{name}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    labels, images = load_data(args.data, args.labels)
    avg_mae, hybrid_avg_mae, gan_avg_mae = None, None, None
    
    # Normal model results
    model = load_model(args.model)
    predictions = predict(images, model)
    avg_mae = mae_per_age(predictions, labels)

    # Hybrid model results
    if args.hybrid_model:
        hybrid_model = load_model(args.hybrid_model)
        hybrid_predictions = predict(images, hybrid_model)
        hybrid_avg_mae = mae_per_age(hybrid_predictions, labels)
    
    # GAN model results
    if args.gan_model:
        gan_model = load_model(args.gan_model)
        gan_predictions = predict(images, gan_model)
        gan_avg_mae = mae_per_age(gan_predictions, labels)

    visualize(avg_mae, hybrid_avg_mae, gan_avg_mae, args.name)