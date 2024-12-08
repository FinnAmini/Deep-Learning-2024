import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from concurrent.futures import ThreadPoolExecutor, as_completed
from train import (
    custom_age_loss,
    custom_gender_loss,
    custom_age_metric,
    custom_gender_metric,
)

# Configure TensorFlow to use GPU memory efficiently
def configure_gpu_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Configured TensorFlow to allocate GPU memory as needed.")
        except RuntimeError as e:
            print(f"Error configuring GPU memory: {e}")

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_and_generate_labels(model, img_path, output_dir):
    # Preprocess image
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)

    # Extract predictions
    face_detection_pred = predictions[0][0]
    age_pred = predictions[1][0]
    gender_pred = predictions[2][0]

    # Generate labels
    gender = "male" if gender_pred < 0.5 else "female"
    labels = {
        "faceAttributes": {
            "face": float(face_detection_pred),
            "age": float(age_pred),
            "gender": gender,
        }
    }

    # Save as JSON file
    img_name = os.path.basename(img_path)
    json_name = os.path.splitext(img_name)[0] + ".json"
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, "w") as json_file:
        json.dump(labels, json_file, indent=4)

def process_image_task(model, img_path, output_dir):
    try:
        predict_and_generate_labels(model, img_path, output_dir)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def process_images_in_folder(model_path, input_folder, output_folder, range_start, range_end):
    # Configure GPU memory
    configure_gpu_memory()

    # Load the model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy,
            "custom_age_loss": custom_age_loss,
            "custom_gender_loss": custom_gender_loss,
            "custom_age_metric": custom_age_metric,
            "custom_gender_metric": custom_gender_metric,
        },
    )

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Gather image paths within the specified range
    img_paths = []
    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                seed_number = int(img_name.split('seed')[1].split('.png')[0])
                if range_start <= seed_number < range_end:
                    img_paths.append(os.path.join(input_folder, img_name))
            except (IndexError, ValueError):
                continue

    # Process images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_image_task, model, img_path, output_folder)
            for img_path in img_paths
        ]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task error: {e}")

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Process images and generate label files.")
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="The folder where the output JSON labels will be saved."
    )
    parser.add_argument(
        "--range_start",
        type=int,
        required=True,
        help="The start of the range for image filenames."
    )
    parser.add_argument(
        "--range_end",
        type=int,
        required=True,
        help="The end of the range for image filenames."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Define the model and input folder
    model_path = "models/mt_exp3_best/mt_resnet50_freeze=False.keras"  # Replace with your model path
    input_folder = "../Meilenstein3/generatedImages"  # Replace with your input folder

    # Call the processing function with the specified output folder and range
    process_images_in_folder(model_path, input_folder, args.output_folder, args.range_start, args.range_end)
