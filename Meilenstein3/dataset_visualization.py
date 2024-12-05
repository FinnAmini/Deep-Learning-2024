import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

def visualize(data):
    sns.histplot(data=data, x='age', hue='gender', multiple='stack', binwidth=1)
    plt.xlim(min(data["age"]), max(data["age"]))
    plt.xlabel("Age")
    plt.ylabel("Count of Labels")
    plt.title("Age Distribution by Gender")
    plt.savefig("age_gender_histogram.png", dpi=300, bbox_inches='tight')

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize dataset')
    parser.add_argument('-l', '--labels', type=str, help='labels path')
    args = parser.parse_args()
    return args

def process_file(file_path):
    """Process a single file and extract age and gender if the condition is met."""
    try:
        with open(file_path, 'r') as f:
            label = json.load(f)
            if type(label) == dict:
                if label["faceAttributes"].get("face") <= 0.5:
                    age = label["faceAttributes"]["age"]
                    gender = label["faceAttributes"]["gender"]
                    return age, gender
            else:
                age = label[0]["faceAttributes"]["age"]
                gender = label[0]["faceAttributes"]["gender"]
                return age, gender
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return None

def load_labels(labels_path, max_workers=8):
    """Load labels concurrently from files in the specified path."""
    data = {"age": [], "gender": []}

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for node in os.scandir(labels_path):
            if node.is_file():
                futures.append(executor.submit(process_file, node.path))

        for future in futures:
            result = future.result()
            if result is not None:
                age, gender = result
                data["age"].append(age)
                data["gender"].append(gender)

    return pd.DataFrame(data)


if __name__ == '__main__':
    args = parse_args()
    data = load_labels(args.labels)
    visualize(data)