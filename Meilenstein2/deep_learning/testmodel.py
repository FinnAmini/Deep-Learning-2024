from pathlib import Path
import json
from train import predict_single_image
import cv2
import numpy as np
import uuid
import os
import argparse

from collections import defaultdict
import tensorflow as tf
from m1_multi_task import (
    custom_age_loss,
    custom_gender_loss,
    custom_age_metric,
    custom_gender_metric,
)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="data/predict")
parser.add_argument("-m", "--model", default="models/test.keras")
args = parser.parse_args()

preds = defaultdict(list)
total = 0

model = tf.keras.models.load_model(
    args.model,
    custom_objects={
        "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy,
        "custom_age_loss": custom_age_loss,
        "custom_gender_loss": custom_gender_loss,
        "custom_age_metric": custom_age_metric,
        "custom_gender_metric": custom_gender_metric,
    },
)

id = uuid.uuid1()
print("Saved in ", id)
os.makedirs(f"data/output/{id}")

for node in Path(args.data).iterdir():
    if node.name.endswith("png") or node.name.endswith("jpg"):
        label_path = str(node).replace("png", "json")
        age, gender = -1, -1
        try:
            with open(label_path) as file:
                json_data = json.load(file)
                age = json_data[0]["faceAttributes"]["age"]
                gender = json_data[0]["faceAttributes"]["gender"]
        except Exception:
            pass
        face_label = "Face" if gender != -1 else "No Face"

        image = cv2.imread(str(node))
        height, width = image.shape[:2]
        left_column = np.ones((height, 200, 3), dtype=np.uint8) * 255
        right_column = np.ones((height, 200, 3), dtype=np.uint8) * 255
        image_with_columns = cv2.hconcat([left_column, image, right_column])
        font = cv2.FONT_HERSHEY_SIMPLEX

        prediction = predict_single_image(model, str(node))
        face_pred = "Face" if prediction[0][0][0] < 0.5 else "No Face"
        face_conf = round(prediction[0][0][0], 2)
        age_pred = round(prediction[1][0][0], 2)
        gender_pred = "male" if prediction[2][0][0] < 0.5 else "female"
        gender_conf = round(prediction[2][0][0], 2)
        scale = 0.6
        thickness = 2

        cv2.putText(
            image_with_columns,
            face_label,
            (10, height // 4),
            font,
            scale,
            (255, 0, 0),
            thickness,
        )
        if face_label == "Face":
            cv2.putText(
                image_with_columns,
                str(age),
                (10, height // 2),
                font,
                scale,
                (255, 0, 0),
                thickness,
            )
            cv2.putText(
                image_with_columns,
                str(gender),
                (10, 3 * height // 4),
                font,
                scale,
                (255, 0, 0),
                thickness,
            )
        cv2.putText(
            image_with_columns,
            f"{face_pred} ({face_conf})",
            (width + 210, height // 4),
            font,
            scale,
            (0, 0, 255),
            thickness,
        )
        if face_pred == "Face":
            cv2.putText(
                image_with_columns,
                str(age_pred),
                (width + 210, height // 2),
                font,
                scale,
                (0, 0, 255),
                thickness,
            )
            cv2.putText(
                image_with_columns,
                f"{gender_pred} ({gender_conf})",
                (width + 210, 3 * height // 4),
                font,
                scale,
                (0, 0, 255),
                thickness,
            )

        cv2.imwrite(f"data/output/{id}/{node.name}", image_with_columns)
