import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

def visualize(data, used_gan_data=False, name=""):
    """
    Visualizes the dataset by creating a histogram of age distribution by gender.

    This function uses seaborn to create a histogram that shows the distribution of ages
    for different genders. It supports both original and GAN-generated data.

    Args:
        data (pd.DataFrame): The dataset containing age and gender information.
        used_gan_data (bool): A flag indicating whether GAN-generated data is used.
        name (str): The filename for saving the visualization.

    Returns:
        None
    """
    print("Visualizing dataset")
    hur_order = ['male - gan', 'female - gan', 'male', 'female']
    palette = sns.color_palette("deep", 4)
    original_palette = {'male': palette[0], 'female': palette[1], 'male - gan': palette[2], 'female - gan': palette[3]}
    sns.histplot(data=data, x='age', hue='gender', hue_order=hur_order, multiple='stack', binwidth=1, palette=original_palette)
    plt.xlim(min(data["age"]), max(data["age"]))
    plt.xlabel("Age")
    plt.ylabel("Count of Labels")
    plt.title("Age Distribution by Gender")
    if used_gan_data:
        plt.savefig(f"age_gender_histogram_gan_{name}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"age_gender_histogram_{name}.png", dpi=300, bbox_inches='tight')

def parse_args():
    """
    Parses command line arguments.

    This function sets up the argument parser to accept command line arguments for the labels path,
    GAN labels path, and the filename for the visualization.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Visualize dataset')
    parser.add_argument('-l', '--labels', type=str, help='labels path')
    parser.add_argument('-g', '--gan-labels', type=str, help='gan labels path')
    parser.add_argument('-n', '--name', type=str, help='filename for the visualization')
    args = parser.parse_args()
    return args

def process_file(file_path):
    """
    Processes a single label file to extract age and gender information.

    This function reads a JSON label file, extracts the age and gender information from the file,
    and returns them as a tuple. If the file contains multiple labels, it processes the first one.

    Args:
        file_path (str): The path to the label file.

    Returns:
        tuple: A tuple containing the age and gender extracted from the label file, or None if an error occurs.
    """
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
    """
    Loads labels from the specified directory using multithreading.

    This function reads label files from the given directory path, extracts age and gender information
    using the process_file function, and appends this information to the 'data' dictionary.

    Args:
        labels_path (str): The path to the directory containing the label files.
        max_workers (int): The maximum number of threads to use for concurrent processing.

    Returns:
        dict: A dictionary with two keys, 'age' and 'gender', each containing lists of ages and genders
              extracted from the label files.
    """
    print(f"Loading labels from {labels_path}")
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
                data["age"].append(int(age))
                data["gender"].append(gender)
    return data

def load_gan_labels(labels_path):
    """
    Loads GAN labels from the specified directory.

    This function reads label files from the given directory path, extracts age and gender information
    from the filenames, and appends this information to the 'data' dictionary.

    Args:
        labels_path (str): The path to the directory containing the label files.

    Returns:
        dict: A dictionary with two keys, 'age' and 'gender', each containing lists of ages and genders
              extracted from the label files.

    Raises:
        ValueError: If the filename does not contain the expected format for age and gender.
    """
    print(f"Loading gan labels from {labels_path}")
    for file in os.scandir(labels_path):
        if file.is_file():
            parts = file.name.split('.')[0].split('_')
            age = int(parts[0])
            gender = f"{parts[1]} - gan"
            with open(file.path, 'r') as f:
                lines = f.readlines()
                data['age'].extend([age] * len(lines))
                data['gender'].extend([gender] * len(lines))
    return data


if __name__ == '__main__':
    args = parse_args()
    data = load_labels(args.labels)
    if args.gan_labels is not None:
        data = load_gan_labels(args.gan_labels)
    visualize(pd.DataFrame(data), args.gan_labels is not None, args.name)