import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

THRESHOLD = 0.4

def calc_distance(anchor, reference):
    """Calculates the distance between the anchor and the reference embeddings."""
    return keras.ops.sum(tf.square(anchor - reference), -1)

@keras.saving.register_keras_serializable()
class DistanceLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = calc_distance(anchor, positive)
        an_distance = calc_distance(anchor, negative)
        return (ap_distance, an_distance)

def build_embedding(input_shape):
    """Builds the embedding network."""
    base_model = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True
    x = keras.layers.Flatten()(base_model.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
    embedding = keras.models.Model(base_model.input, x)
    return embedding


def build_model(input_shape):
    """Builds the Siamese network model."""
    anchor_input = keras.layers.Input(name="anchor", shape=input_shape)
    positive_input = keras.layers.Input(name="positive", shape=input_shape)
    negative_input = keras.layers.Input(name="negative", shape=input_shape)
    embedding = build_embedding(input_shape)

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input)
    )

    # Build the Siamese model
    siamese = keras.models.Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return siamese, embedding

@keras.saving.register_keras_serializable()
class SiameseModel(keras.models.Model):

    def __init__(self, siamese_network=None, **kwargs):
        super().__init__()
        siamese, embedding = build_model((224, 224, 3))
        self.siamese_network = siamese
        self.embedding = embedding

    def call(self, inputs):
        return self.embedding(inputs)

def load_image(image_path, img_width=224, img_height=224):
    """
    Load an image from the path and convert it to a tensor.
    Resize the image to the required dimensions and normalize it.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img = np.array(img, dtype=np.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

class MLManager:

    def __init__(self):
        self.model = self.load_model('model.keras')
        if os.path.exists('db.json'):
            with open('db.json', 'r') as file:
                self.embeddings = json.load(file)
        else:
            self.embeddings = {}

    def load_model(self, path):
        return tf.keras.models.load_model(
            path,
            custom_objects={"SiameseModel": SiameseModel, "DistanceLayer": DistanceLayer}
        )

    def calc_embedding(self, path):
        img = np.array([load_image(path)])
        embedding = self.model.predict(img)
        return embedding

    def recognize(self, embedding, n=5):
        """Function to recognize people"""
        distances = []
        for path, emb in self.embeddings.items():
            distance = calc_distance(embedding, np.array(emb))[0]
            distances.append({
                "distance": float(distance),
                "path": path,
                "recognized": bool(distance <= THRESHOLD)
            })

        distances.sort(key=lambda x: x["distance"])
        closest_n = distances[:n]
        furthest_n = distances[-n:]

        return closest_n, furthest_n