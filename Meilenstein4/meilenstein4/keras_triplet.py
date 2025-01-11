import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from keras import layers
from keras import ops
from keras import metrics
from keras.applications import resnet
from argparse import ArgumentParser
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import (
    Dense,
    BatchNormalization,
    Flatten
)
import datetime
import os

def preprocess_image(filename, target_shape=(224, 224)):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

def create_image_lists(image_dir):
    anchor_images, positive_images = [], []

    # Iterate over each person's directory
    for person_dir in Path(image_dir).iterdir():
        if person_dir.is_dir():
            person_images = sorted(person_dir.glob("*"))  # List images in sorted order
            # Create pairs of anchor and positive images
            for i in range(len(person_images) - 1):
                anchor_images.append(str(person_images[i]))
                positive_images.append(str(person_images[i + 1]))
            anchor_images.append(str(person_images[-1]))
            positive_images.append(str(person_images[0]))

    return anchor_images, positive_images

def load_data(images_path, batch_size=32, val_split=0.2):
    """Function to load the data from the specified paths."""
    # We need to make sure both the anchor and positive images are loaded in
    # sorted order so we can match them together.
    anchor_images, positive_images = create_image_lists(images_path)

    image_count = len(anchor_images)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

    # To generate the list of negative images, let's randomize the list of
    # available images and concatenate them together.
    rng = np.random.RandomState(seed=42)
    rng.shuffle(anchor_images)
    rng.shuffle(positive_images)

    negative_images = anchor_images + positive_images
    np.random.RandomState(seed=32).shuffle(negative_images)

    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
    negative_dataset = negative_dataset.shuffle(buffer_size=4096)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(image_count * (1 - val_split)))
    val_dataset = dataset.skip(round(image_count * (1 - val_split)))

    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset

def build_model(input_shape):
    """Builds the Siamese network model."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  
    x = Flatten()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    embedding = Model(base_model.input, x)

    anchor_input = layers.Input(name="anchor", shape=input_shape)
    positive_input = layers.Input(name="positive", shape=input_shape)
    negative_input = layers.Input(name="negative", shape=input_shape)

    # Compute the distance between the embeddings
    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    # Build the Siamese model
    siamese = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return siamese


def train(data_path, batch_size, epochs, margin, name, val_split):
    train_dataset, val_dataset = load_data(data_path, batch_size, val_split)
    model = build_model((224, 224, 3))
    siamese_model = SiameseModel(model, margin=margin)
    siamese_model.compile(optimizer="adam")

    train_log_dir = (f"logs/triplet/model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    tensorboard_train_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)

    checkpoint_dir = './checkpoints/triplet'
    checkpoint_filepath = f'{checkpoint_dir}/model_epoch_{{epoch:02d}}_{name}'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,  
        save_freq='epoch'          
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models/triplet', exist_ok=True)

    siamese_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[tensorboard_train_callback, checkpoint_callback])
    siamese_model.save(f"models/triplet/{name}")

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", "-d", type=str, default="data/images/train", help="Path to the dataset")
    train_parser.add_argument("--img_width", "-iw", type=int, default=224, help="Image width")
    train_parser.add_argument("--img_height", "-ih", type=int, default=224, help="Image height")
    train_parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--name", "-n", type=str, default="model.keras", help="Output model file")
    train_parser.add_argument("--margin", "-m", type=float, default=1, help="Margin for contrastive loss")
    train_parser.add_argument("--visualize_data", "-v", action="store_true", help="Visualize the data")
    train_parser.add_argument("--evaluate", "-ev", action="store_true", help="Evaluate the model")
    train_parser.add_argument("-val_split", type=float, default=0.2, help="Validation split")
    train_parser.set_defaults(func=train_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.name, args.val_split, args.visualize_data, args.evaluate)

if __name__ == '__main__':
    args = parse_args()
    args.func(args)