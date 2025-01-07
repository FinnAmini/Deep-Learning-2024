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
from datetime import datetime

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

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def build_model(input_shape):
    input = keras.layers.Input(input_shape)
    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.Conv2D(4, (5, 5), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(10, activation="tanh")(x)
    embedding_network = keras.Model(input, x)


    input_1 = keras.layers.Input(input_shape)
    input_2 = keras.layers.Input(input_shape)

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
        [tower_1, tower_2]
    )
    normal_layer = keras.layers.BatchNormalization()(merge_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese

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

def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'."""
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss."""
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

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

def train(data_path, batch_size, epochs, margin, output, val_split, visualize_data=False, evaluate=False):
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
    siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
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
    train_log_dir = (f"logs/model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    tensorboard_train_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)

    dataloaders = create_dataloader(data_path, batch_size=batch_size, val_split=val_split)
    history = siamese.fit(dataloaders[0], validation_data=dataloaders[1], epochs=epochs, callbacks=[tensorboard_train_callback])

    # If specified, evaluate the model
    # if evaluate:
    #     plt_metric(history=history.history, metric="accuracy", title="Model accuracy")
    #     plt_metric(history=history.history, metric="loss", title="Contrastive Loss")
        
    # save the model
    siamese.save(output)
    

def test(model):
    """Tests the Siamese network model."""
    x_test, y_test = load_data("data/vgg_faces2_resized/test")
    pairs_test, labels_test = make_pairs(x_test, y_test)
    x_test_1 = pairs_test[:, 0]
    x_test_2 = pairs_test[:, 1]
    visualize(pairs_test[:-1], labels_test[:-1], to_show=4, num_col=4)

    results = model.evaluate([x_test_1, x_test_2], labels_test)
    print("test loss, test acc:", results)
    predictions = model.predict([x_test_1, x_test_2])
    visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)


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
    train_parser.add_argument("--output", "-o", type=str, default="models/model.keras", help="Output model file")
    train_parser.add_argument("--margin", "-m", type=float, default=1, help="Margin for contrastive loss")
    train_parser.add_argument("--visualize_data", "-v", action="store_true", help="Visualize the data")
    train_parser.add_argument("--evaluate", "-ev", action="store_true", help="Evaluate the model")
    train_parser.add_argument("-val_split", type=float, default=0.2, help="Validation split")
    train_parser.set_defaults(func=train_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.output, args.val_split, args.visualize_data, args.evaluate)

if __name__ == '__main__':
    args = parse_args()
    args.func(args)