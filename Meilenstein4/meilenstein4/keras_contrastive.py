import random
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Lambda,
    BatchNormalization,
    Flatten
)
import tensorflow as tf
from tensorflow.keras import backend as K
from argparse import ArgumentParser
from dataloading import create_dataloader
import datetime
from sklearn.metrics import accuracy_score
from tensorflow.keras.saving import register_keras_serializable

def preprocess_image(image_path, img_size=(224, 224)):
    """Loads image from the given path and preprocess."""
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array

def load_data(base_path):
    """Loads the dataset from the given path.

    Arguments:
        path: String, path to the dataset.
    Returns:
        Tuple of numpy arrays: (images, labels).
    """
    data = []

    for idx, person_dir in enumerate(sorted(os.listdir(base_path))):
        person_path = os.path.join(base_path, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.scandir(person_path):
                if img_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append((preprocess_image(img_file.path), idx))

    random.shuffle(data)
    images, labels = zip(*data)
    return list(images), list(labels)

def split_data(images, labels, val_split=0.2):
    """Splits the images and labels into training and validation sets."""
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=val_split, stratify=labels, random_state=42
    )
    return train_images, val_images, train_labels, val_labels

def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset."""
    num_row = to_show // num_col if to_show // num_col != 0 else 1
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def build_model(input_shape):
    """Builds the Siamese network model."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  
    x = Flatten()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    embedding_network = Model(base_model.input, x)

    # Define the two inputs for the Siamese network
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    # Compute the Euclidean distance between the two embeddings
    merge_layer = Lambda(euclidean_distance, output_shape=(1,))([tower_1, tower_2])
    normal_layer = BatchNormalization()(merge_layer)
    output_layer = Dense(1, activation="sigmoid")(normal_layer)

    # Build the Siamese model
    siamese = Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese

@register_keras_serializable()
def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'."""
    @register_keras_serializable()
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss."""
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        res = ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)
        return res

    return contrastive_loss

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'. """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()

def train(data_path, batch_size, epochs, margin, name, val_split, visualize_data=False, evaluate=False):
    """Trains the Siamese network model."""
    # Load the dataset and split it into training and validation sets
    # x_train, y_train = load_data(data_path)
    # x_train, x_val, y_train, y_val = split_data(x_train, y_train, val_split=0.2)
    # pairs_train, labels_train = make_pairs(x_train, y_train)
    # pairs_val, labels_val = make_pairs(x_val, y_val)

    # # Extract the images from the pairs for further processing
    # x_train_1 = pairs_train[:, 0]
    # x_train_2 = pairs_train[:, 1]
    # x_val_1 = pairs_val[:, 0]
    # x_val_2 = pairs_val[:, 1]

    # # if specified, visualize the data
    # if visualize_data:
    #     visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)
    #     visualize(pairs_val[:-1], labels_val[:-1], to_show=4, num_col=4)

    # Build and compile the Siamese network model
    siamese = build_model(input_shape=(224, 224, 3))
    siamese.compile(loss=loss(margin=margin), optimizer="adam", metrics=["accuracy"])
    siamese.summary()

    # Train the model
    # history = siamese.fit(
    #     [x_train_1, x_train_2],
    #     labels_train,
    #     validation_data=([x_val_1, x_val_2], labels_val),
    #     batch_size=batch_size,
    #     epochs=epochs,
    # )

    # Define log directory for TensorBoard
    train_log_dir = (f"logs/contrastive/model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    tensorboard_train_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)

    checkpoint_dir = './checkpoints/contrastive'
    checkpoint_filepath = f'{checkpoint_dir}/model_epoch_{{epoch:02d}}_{name}'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,  
        save_freq='epoch'          
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('models/contrastive', exist_ok=True)

    dataloaders = create_dataloader(data_path, batch_size=batch_size, val_split=val_split)
    siamese.fit(
        dataloaders[0], validation_data=dataloaders[1], epochs=epochs, callbacks=[tensorboard_train_callback, checkpoint_callback]
    )

    # If specified, evaluate the model
    # if evaluate:
    #     plt_metric(history=history.history, metric="accuracy", title="Model accuracy")
    #     plt_metric(history=history.history, metric="loss", title="Contrastive Loss")
        
    # save the model
    siamese.save(f"models/contrastive/{name}")
    
def create_test_pairs(data_path):
    """Creates test pairs for the Siamese network."""
    left, right, labels = [], [], []
    examples = 0
    for person_dir in sorted(os.listdir(data_path)):
        person_path = os.path.join(data_path, person_dir)
        if os.path.isdir(person_path):
            img_path = os.path.join(person_path, os.listdir(person_path)[0])

            # positive pair
            pos_path = os.path.join(person_path, random.choice([p for p in os.listdir(person_path) if p != img_path]))
            left.append(preprocess_image(img_path))
            right.append(preprocess_image(pos_path))
            labels.append(0)

            # negative pair
            neg_dir = os.path.join(data_path, random.choice([p for p in os.listdir(data_path) if p != person_dir]))
            neg_path = os.path.join(neg_dir, random.choice(os.listdir(neg_dir)))
            left.append(preprocess_image(img_path))
            right.append(preprocess_image(neg_path))
            labels.append(1)

            examples += 1
            if examples == 100:
                break

    return  np.array(left),  np.array(right), labels

def test(model_path, data_path, batch_size=32):
    """Tests the Siamese network model."""
    left, right, labels = create_test_pairs(data_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'contrastive_loss': loss})
    predictions = model.predict([left, right])
    predicted_labels = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


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

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    test_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    test_parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    test_parser.set_defaults(func=test_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.name, args.val_split, args.visualize_data, args.evaluate)

def test_handler(args):
    """Handler for the test command."""
    test(args.model, args.data, args.batch_size)

if __name__ == '__main__':
    args = parse_args()
    args.func(args)