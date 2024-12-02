import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from train import (
    custom_age_loss,
    custom_gender_loss,
    custom_age_metric,
    custom_gender_metric,
)

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

    # generate labels
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

def process_images_in_folder(model_path, input_folder, output_folder):
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

    # Process each image in the folder
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
            predict_and_generate_labels(model, img_path, output_folder)

if __name__ == "__main__":
    model_path = "../models/mt_exp3_best/mt_resnet50_freeze=False.keras"  # Replace with your model path
    input_folder = "../Meilenstein3/stylegan3/generatedImages"  # Replace with your input folder
    output_folder = "../generatedLabels"  # Replace with your output folder

    process_images_in_folder(model_path, input_folder, output_folder)
