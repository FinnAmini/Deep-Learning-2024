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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data/predict", type=Path)
    parser.add_argument("-l", "--labels", default="data/labels/test", type=Path)
    parser.add_argument("-m", "--model")
    return parser.parse_args()

def load_data(img_dir: Path, label_dir: Path):
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
    print("Predicting...")
    predictions = model.predict(images)
    return [p[0] for p in predictions[1]]

def load_model(model_path):
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

def mae_per_age(predictions, age_labels):
    print("Calculating MAE per age...")
    if len(age_labels) != len(predictions):
        raise ValueError("The length of actual_ages and predicted_ages must be the same.")
    
    diff_dict = defaultdict(lambda: {"2": 0, "5": 0, "10": 0, "count": 0})
    
    for actual, predicted in zip(age_labels, predictions):
        diff = abs(actual - predicted)
        if diff >= 2:
            diff_dict[actual]["2"] += 1
        if diff >= 5:
            diff_dict[actual]["5"] += 1
        if diff >= 10:
            diff_dict[actual]["10"] += 1
        diff_dict[actual]["count"] += 1

    diff_dict = {
        age: {
            "2": diff_data["2"] / diff_data["count"],
            "5": diff_data["5"] / diff_data["count"],
            "10": diff_data["10"] / diff_data["count"],
        }
        for age, diff_data in diff_dict.items()
    }
    
    return diff_dict

def visualize(data):
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    rows = []
    for age, diffs in data.items():
        for diff, count in diffs.items():
            rows.append({'age': age, 'diff': f"Diff ≥ {diff}", 'Percentage': count})

    df = pd.DataFrame(rows)
    sns.lineplot(data=df, x='age', y='Percentage', hue='diff')
    plt.title('Percentage by Age and Prediction Difference')
    plt.xlabel('Age')
    plt.ylabel('Percentage of Images')
    plt.ylim(0, None)
    plt.legend(title='Difference between Actual and Predicted Age')
    plt.savefig("counts_per_age.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    age_labels, images = load_data(args.data, args.labels)
    model = load_model(args.model)
    predictions = predict(images, model)
    mae_diff_count = mae_per_age(predictions, age_labels)
    visualize(mae_diff_count)