from m1_multi_task import custom_age_loss, custom_gender_loss
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict_single_image(model_path, img_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "custom_gender_loss": custom_age_loss,
            "custom_age_loss": custom_gender_loss,
        },
    )
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)

    face_detection_pred = predictions[0][0]
    age_pred = predictions[1][0]
    gender_pred = predictions[2][0]

    gender = "male" if gender_pred < 0.5 else "female"

    print(
        f"Face Detection: {'Face' if face_detection_pred < 0.5 else 'No Face', face_detection_pred}"
    )
    print(f"Predicted Age: {age_pred}")
    print(f"Predicted Gender: {gender, gender_pred}")


if __name__ == "__main__":
    model_path = "../../faminikaveh/projects/Deep-Learning-2024/models/mt_backup/mt_efficientnetb4_freeze=False.keras"  # Replace with your model path
    img_path = "data/testing/faces/62525.png"  # Replace with your image path
    predict_single_image(model_path, img_path)
