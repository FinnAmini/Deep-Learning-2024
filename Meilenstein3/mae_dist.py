from pathlib import Path
import json
import numpy as np
import argparse
from collections import defaultdict
import tensorflow as tf
import train as t
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data/predict", type=Path)
    parser.add_argument("-l", "--labels", default="data/labels/test", type=Path)
    parser.add_argument("-m", "--model")
    parser.add_argument("-n", "--name", default="")
    return parser.parse_args()

def load_data(img_dir: Path, label_dir: Path):
    """
    Loads images and their corresponding age labels from the specified directories.

    Args:
        img_dir (Path): Path to the directory containing image files.
        label_dir (Path): Path to the directory containing label files in JSON format.

    Returns:
        tuple: A tuple containing:
            - age_labels (list): A list of age labels extracted from the JSON files.
            - images (numpy.ndarray): A numpy array of preprocessed images.

    Raises:
        Exception: If there is an error reading a label file or processing an image.
    """
    print("Loading images...")
    age_labels, images = [], []
    for node in img_dir.iterdir():
        if node.name.endswith("png") or node.name.endswith("jpg"):
            try:
                with open(label_dir / f"{node.stem}.json") as file:
                    label = json.load(file)
                    age = label[0]["faceAttributes"]["age"]
                    age_labels.append(age)
                    images.append(t.load_and_preprocess_image(str(node)))
            except Exception as e:
                print(e)
    return age_labels, np.vstack(images)

def predict(images, model):
    """
    Predict the ages for the given images using the provided model.

    Args:
        images (numpy.ndarray): The preprocessed images to predict ages for.
        model (tf.keras.Model): The trained Keras model to use for predictions.

    Returns:
        list: A list of predicted ages.
    """
    print("Predicting...")
    predictions = model.predict(images)
    return [p[0] for p in predictions[1]]

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

def calc_mae_dist(predictions, age_labels):
    """
    Calculate the Mean Absolute Error (MAE) distribution for age predictions.
    This function computes the MAE for each prediction and returns a dictionary
    containing the MAE values, along with the overall MAE, and the percentage of
    predictions within 2, 5, and 10 years of the actual age.
    Args:
        predictions (list of int/float): The predicted ages.
        age_labels (list of int/float): The actual ages.
    Returns:
        tuple: A tuple containing:
            - diff_dict (dict): A dictionary with a key "MAE" containing a list of absolute differences.
            - overall_mae (float): The overall mean absolute error.
            - within_2_years (float): The percentage of predictions within 2 years of the actual age.
            - within_5_years (float): The percentage of predictions within 5 years of the actual age.
            - within_10_years (float): The percentage of predictions within 10 years of the actual age.
    Raises:
        ValueError: If the length of `age_labels` and `predictions` are not the same.
    """
    print("Calculating MAE per age...")
    if len(age_labels) != len(predictions):
        raise ValueError("The length of actual_ages and predicted_ages must be the same.")
    
    diff_dict = {"MAE": []}
    sum_diff, diff2, diff5, diff10, total = 0, 0, 0, 0, 0
    
    for actual, predicted in zip(age_labels, predictions):
        diff = abs(actual - predicted)
        sum_diff += diff
        total += 1
        diff2 += 1 if diff <= 2 else 0
        diff5 += 1 if diff <= 5 else 0
        diff10 += 1 if diff <= 10 else 0
        diff_dict["MAE"].append(diff)
    
    return (
        diff_dict,
        round(sum_diff / total, 2),
        round(diff2 / total * 100, 2),
        round(diff5 / total * 100, 2),
        round(diff10 / total * 100, 2)
    )

def visualize(data, name=""):
    """
    Visualizes the distribution of Mean Absolute Error (MAE) and other statistics from the given data.

    Parameters:
    data (tuple): A tuple containing the following elements:
        - diff_dict (DataFrame): A DataFrame containing the differences between labels and predictions.
        - mae (float): The Mean Absolute Error value.
        - diff2 (float): The percentage of differences less than or equal to 2.
        - diff5 (float): The percentage of differences less than or equal to 5.
        - diff10 (float): The percentage of differences less than or equal to 10.
    name (str, optional): A name to append to the saved plot file. Defaults to an empty string.

    Returns:
    None: The function saves the plot as a PNG file with the specified name.
    """
    diff_dict, mae, diff2, diff5, diff10 = data
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    sns.histplot(data=diff_dict, x='MAE', binwidth=1)
    plt.xlim(0, 15)
    plt.xlabel("Age Difference (Label - Prediction)") 
    plt.ylabel("Anzahl")
    plt.title("Age Difference Distribution")
    custom_legend = [
        Line2D([0], [0], color='none', label=r'$\mathbf{MAE}$:         ' + f'{mae}'),
        Line2D([0], [0], color='none', label=r'$\mathbf{Diff \leq 2}$:    ' + f'{diff2}%'),
        Line2D([0], [0], color='none', label=r'$\mathbf{Diff \leq 5}$:    ' + f'{diff5}%'),
        Line2D([0], [0], color='none', label=r'$\mathbf{Diff \leq 10}$:  ' + f'{diff10}%')
    ]
    plt.legend(handles=custom_legend, loc="upper right", frameon=True, fontsize='medium', title_fontsize='large', handlelength=0)
    plt.savefig(f"mae_dist_hist_{name}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    age_labels, images = load_data(args.data, args.labels)
    model = load_model(args.model)
    predictions = predict(images, model)
    mae_dist = calc_mae_dist(predictions, age_labels)
    visualize(mae_dist, args.name)