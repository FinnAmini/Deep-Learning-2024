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
    model = tf.keras.models.load_model(model_path)
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)

    print(predictions)


if __name__ == "__main__":
    model_path = "models/st/st_resnet50_adam_lr=0.0001_lc=0_freeze=False.keras"
    # img_path = "data/testing/gan_faces/seed62523.png"  # Replace with your image path
    img_path = "classificator_gan_non_gan/examples/image.png"  # Replace with your image path
    predict_single_image(model_path, img_path)
