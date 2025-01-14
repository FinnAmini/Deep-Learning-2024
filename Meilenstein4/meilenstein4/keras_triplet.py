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
from dataloading_triplet import create_dataloader


class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

    def get_config(self):
        config = super().get_config()
        return config


class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.accuracy_tracker = metrics.Mean(name="accuracy")
        if siamese_network is not None:
            self.siamese_network = siamese_network
        else:
            self.siamese_network = self.build_model(224, 224, 3)

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            accuracy = self._compute_accuracy(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        accuracy = self._compute_accuracy(data)
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(0.0, ap_distance - an_distance + self.margin)
        # loss = tf.nn.relu(ap_distance - an_distance + self.margin)
        return tf.reduce_mean(loss)

    def _compute_accuracy(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        correct = tf.cast(ap_distance + self.margin < an_distance, tf.float32)
        return tf.reduce_mean(correct)

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({
            "siamese_network": self.siamese_network,
            "margin": self.margin
        })
        return config

    @classmethod
    def from_config(cls, config):
        siamese_network = config.pop("siamese_network")
        return cls(siamese_network=siamese_network, **config)


def build_model(input_shape):
    """Builds the Siamese network model."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True  
    x = Flatten()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
    embedding = Model(base_model.input, x)

    anchor_input = layers.Input(name="anchor", shape=input_shape)
    positive_input = layers.Input(name="positive", shape=input_shape)
    negative_input = layers.Input(name="negative", shape=input_shape)

    # Compute the distance between the embeddings
    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    # Build the Siamese model
    siamese = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return siamese


def train(data_path, batch_size, epochs, margin, name, val_split):
    model = build_model((224, 224, 3))
    siamese_model = SiameseModel(model, margin=margin)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

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

    dataloaders = create_dataloader(data_path, batch_size=batch_size, val_split=val_split)
    siamese_model.fit(dataloaders[0], validation_data=dataloaders[1], epochs=epochs, callbacks=[tensorboard_train_callback, checkpoint_callback])
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
    train_parser.add_argument("--name", "-n", type=str, default="model_triplet.keras", help="Output model file")
    train_parser.add_argument("--margin", "-m", type=float, default=1, help="Margin for contrastive loss")
    train_parser.add_argument("-val_split", type=float, default=0.2, help="Validation split")
    train_parser.set_defaults(func=train_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.name, args.val_split)

if __name__ == '__main__':
    args = parse_args()
    args.func(args)