import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
import keras
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
from dataloading_triplet import create_dataloader, load_image
import random
from keras.preprocessing.image import load_img

@keras.saving.register_keras_serializable()
class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(tf.square(anchor - positive), -1)
        an_distance = ops.sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

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

@keras.saving.register_keras_serializable()
class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network=None, margin=0.5, **kwargs):
        super().__init__()
        self.siamese_network = build_model((224, 224, 3))
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.accuracy_tracker = metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            accuracy = self._compute_accuracy(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.

        # print('\n\n##############')
        # print("Batch - Tracker Loss before: ", self.loss_tracker.result().numpy())
        # print(f"Batch - Tracker Accuracy before: {self.accuracy_tracker.result().numpy()}")
        # print("Batch loss:", loss.numpy())
        # print("Batch acc:", accuracy.numpy())
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        # print(f"Batch - Tracker Accuracy after: {self.accuracy_tracker.result().numpy()}")
        # print(f"Batch - Tracker Loss after: {self.loss_tracker.result().numpy()}")
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
        return tf.reduce_mean(loss)

    def _compute_accuracy(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        correct = tf.cast(ap_distance + self.margin < an_distance, tf.float32)
        return tf.reduce_mean(correct)

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]


def train(data_path, batch_size, epochs, margin, name, val_split):
    siamese_model = SiameseModel(margin=margin)
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

def _pred_and_visualize(model, anker_path, pos_path, neg_path, id):
    """Predicts and visualizes the data."""
    anker = load_image(anker_path)
    pos = load_image(pos_path)
    neg = load_image(neg_path)

    prediction = model.predict([
        np.expand_dims(anker, axis=0),
        np.expand_dims(pos, axis=0),
        np.expand_dims(neg, axis=0)
    ])

    ap, an = prediction[0][0], prediction[1][0]

    # Display the images and prediction
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(load_img(anker_path))
    axes[0].set_title("Anker")
    axes[0].axis("off")

    axes[1].imshow(load_img(pos_path))
    axes[1].set_title("Positive")
    axes[1].axis("off")

    axes[2].imshow(load_img(neg_path))
    axes[2].set_title("Negative")
    axes[2].axis("off")

    plt.suptitle(f"AP: {ap:.4f}, AN: {an:.4f}, Good: {ap < an}", fontsize=16)
    anker_name = "-".join(anker_path.split('/')[-2:])
    pos_name = "-".join(pos_path.split('/')[-2:])
    neg_name = "-".join(neg_path.split('/')[-2:])
    plt.savefig(f"data/eval/triplet/{id}_{anker_name}###{pos_name}###{neg_name}.png")

    return ap, an

def evaluate_and_predict(model_path, data_path):
    """Evaluates the model and makes a prediction on a single example."""
    print('right now')
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"SiameseModel": SiameseModel, "DistanceLayer": DistanceLayer}
    )
    print('right here')

    # Make a prediction with one example
    people_dirs = sorted(os.listdir(data_path))

    total, good, ap_dist_sum, an_dist_sum = 0, 0, 0, 0 

    for i in range(1000):
        # positive example and anker
        person1_dir = random.choice(people_dirs)
        person1_dir_path = os.path.join(data_path, person1_dir)
        anker_path = os.path.join(person1_dir_path, random.choice(os.listdir(person1_dir_path)))
        pos_path = os.path.join(person1_dir_path, random.choice([i for i in os.listdir(person1_dir_path) if i != anker_path]))
        person2_dir = random.choice([p for p in people_dirs if p != person1_dir])
        person2_dir_path = os.path.join(data_path, person2_dir)
        neg_path = os.path.join(person2_dir_path, random.choice(os.listdir(person2_dir_path)))
        ap, an = _pred_and_visualize(model, anker_path, pos_path, neg_path, i)
        total += 1
        if ap < an:
            good += 1
        ap_dist_sum += ap
        an_dist_sum += an

    print(f"Predicted {good} out of {total} examples correctly. - {good/total*100:.2f}%")
    print(f"Avg. AP distance: {ap_dist_sum/good:.4f}")
    print(f"Avg. AN distance: {an_dist_sum/good:.4f}")

def test(model_path, data_path, batch_size=32):
    """Tests the Siamese network model."""
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"SiameseModel": SiameseModel, "DistanceLayer": DistanceLayer}
    )
    test_data = create_dataloader(data_path, batch_size=batch_size, val_split=0)
    model.evaluate(test_data)

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
    train_parser.add_argument("--margin", "-m", type=float, default=0.2, help="Margin for contrastive loss")
    train_parser.add_argument("-val_split", type=float, default=0.2, help="Validation split")
    train_parser.set_defaults(func=train_handler)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    test_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    test_parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    test_parser.set_defaults(func=test_handler)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    eval_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    eval_parser.set_defaults(func=evaluate_and_predict_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.name, args.val_split)

def test_handler(args):
    """Handler for the test command."""
    test(args.model, args.data, args.batch_size)

def evaluate_and_predict_handler(args):
    """Handler for the evaluate command."""
    evaluate_and_predict(args.model, args.data)

if __name__ == '__main__':
    args = parse_args()
    args.func(args)