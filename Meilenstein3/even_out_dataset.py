import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

def visualize(data):
    """
    Visualizes the age distribution by gender using a histogram.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the data to be visualized. 
                      It should have columns 'age' and 'gender'.

    Returns:
    None: The function saves the histogram as 'age_gender_histogram.png' in the current directory.
    """
    sns.histplot(data=data, x='age', hue='gender', multiple='stack', binwidth=1)
    plt.xlim(min(data["age"]), max(data["age"]))
    plt.xlabel("Age")
    plt.ylabel("Count of Labels")
    plt.title("Age Distribution by Gender")
    plt.savefig("age_gender_histogram.png", dpi=300, bbox_inches='tight')

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize dataset')
    parser.add_argument('-l', '--labels', type=str, help='labels path')
    parser.add_argument('-g', '--gan-labels', type=str, help='gan labels path')
    args = parser.parse_args()
    return args

def process_file(file_path):
    """
    Processes a JSON file to extract age and gender information.
    Args:
        file_path (str): The path to the JSON file to be processed.
    Returns:
        tuple: A tuple containing age, gender, and file_path if the file is processed successfully.
               Returns None if an error occurs or if the conditions are not met.
    Raises:
        Exception: If there is an error opening or reading the file.
    Notes:
        - The JSON file is expected to contain a dictionary with a "faceAttributes" key.
        - If the "faceAttributes" key contains a "face" value less than or equal to 0.5, the age and gender are extracted.
        - If the JSON file contains a list, the age and gender are extracted from the first element.
    """
    try:
        with open(file_path, 'r') as f:
            label = json.load(f)
            if type(label) == dict:
                if label["faceAttributes"].get("face") <= 0.5:
                    age = label["faceAttributes"]["age"]
                    gender = label["faceAttributes"]["gender"]
                    return age, gender, file_path
            else:
                age = label[0]["faceAttributes"]["age"]
                gender = label[0]["faceAttributes"]["gender"]
                return age, gender, file_path
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return None

def load_labels(labels_path, max_workers=16):
    """
    Load labels from the specified directory path.

    This function scans the given directory for files and processes each file
    concurrently using a thread pool. It categorizes the data based on age and gender.

    Args:
        labels_path (str): The path to the directory containing label files.
        max_workers (int, optional): The maximum number of threads to use for concurrent processing. Defaults to 16.

    Returns:
        defaultdict: A dictionary where keys are ages (int) and values are dictionaries with keys "male" and "female",
                     each containing a list of file paths.

    Raises:
        Exception: If an error occurs during file processing.

    Example:
        data = load_labels("/path/to/labels")
    """
    print(f"Loading labels from {labels_path}")
    data = defaultdict(lambda: {"male": [], "female": []})
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, node.path): node for node in os.scandir(labels_path) if node.is_file()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                age, gender, path = result
                data[int(age)][gender].append(path)
            else:
                errors += 1

    print(f"Completed loading with {errors} errors.")
    return data

def even_out(labels, gan_labels):
    """
    Even out the dataset by removing labels from the GAN dataset.
    This function ensures that the dataset is balanced by filling in the missing labels 
    from the GAN dataset. It calculates the number of labels needed for each age group 
    and gender, and writes the required labels to text files.
    Args:
        labels (dict): A dictionary containing the original dataset labels. 
                       The keys are age groups, and the values are dictionaries 
                       with 'male' and 'female' keys containing lists of labels.
        gan_labels (dict): A dictionary containing the GAN-generated labels. 
                           The structure is the same as the `labels` dictionary.
    Returns:
        None
    """
    print("Evening out the dataset")
    """Even out the dataset by removing labels from the GAN dataset."""
    labels = {key: labels[key] for key in sorted(labels.keys())}
    total_labels = {age: len(values['male']) + len(values['female']) for age, values in labels.items()}
    total_gan_labels = {age: len(values['male']) + len(values['female']) for age, values in gan_labels.items()}
    max_labels = max(total_labels.values())

    print(f"Total labels: {sum(total_labels.values())}")
    print(f"Total GAN labels: {sum(total_gan_labels.values())}")

    for age, values in labels.items():
        males = len(values['male'])
        females = len(values['female'])
        total = males + females

        labels_needed = max_labels - total
        male_labels_needed = max(0, int(max_labels / 2 - males))
        female_labels_needed = max(0, int(max_labels / 2 - females))

        actual_male_labels_needed = male_labels_needed + max(0, female_labels_needed - len(gan_labels[age]['female']))
        actual_female_labels_needed = female_labels_needed + max(0, male_labels_needed - len(gan_labels[age]['male']))

        labels[age]['male_filled'] = gan_labels[age]['male'][:actual_male_labels_needed]
        labels[age]['female_filled'] = gan_labels[age]['female'][:actual_female_labels_needed]

        with open(f'data/labels_for_test/{age}_male.txt', 'w') as f:
            for node in gan_labels[age]['male'][:actual_male_labels_needed]:
                f.write(f"{node}\n")
            
        with open(f'data/labels_for_test/{age}_female.txt', 'w') as f:
            for node in gan_labels[age]['female'][:actual_female_labels_needed]:
                f.write(f"{node}\n")

        print(f'\n####################### {age} #######################')
        print(f"Actual males: {len(labels[age]['male'])}")
        print(f"Actual females: {len(labels[age]['female'])}")
        print(f"gan males: {len(gan_labels[age]['male'])}")
        print(f"gan females: {len(gan_labels[age]['female'])}")
        print(f"total labels: {total}")
        print(f"max labels: {max_labels}")
        print(f"Labels needed: {labels_needed}")
        print(f"Male labels needed: {male_labels_needed}")
        print(f"Females labels needed: {female_labels_needed}")
        print(f"actual male labels needed: {actual_male_labels_needed}")
        print(f"actual female labels needed: {actual_female_labels_needed}")

if __name__ == '__main__':
    args = parse_args()
    labels = load_labels(args.labels)
    gan_labels = load_labels(args.gan_labels)
    filled = even_out(labels, gan_labels)