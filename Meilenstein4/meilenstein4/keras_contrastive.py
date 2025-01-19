import random
import numpy as np
from keras import ops
import matplotlib.pyplot as plt
import os
import sys
from keras.preprocessing.image import load_img
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
from dataloading_contrastive import create_dataloader, verify_dataloader
import datetime
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def build_model(input_shape, base_model=None):
    """Builds the Siamese network model."""
    embedding_network = None
    if base_model is None:
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = True  
        x = Flatten()(base_model.output)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        embedding_network = Model(base_model.input, x)
    else:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        from Meilenstein2.deep_learning.train import (
            custom_age_loss,
            custom_gender_loss,
            custom_age_metric,
            custom_gender_metric,
        )
        base_model = tf.keras.models.load_model(
            base_model,
            custom_objects={
                "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy,
                "custom_age_loss": custom_age_loss,
                "custom_gender_loss": custom_gender_loss,
                "custom_age_metric": custom_age_metric,
                "custom_gender_metric": custom_gender_metric,
            },
        )
        # remove top layers
        base_model = Model(base_model.input, base_model.layers[-13].output)
        base_model.trainable = True
        x = BatchNormalization()(base_model.output)
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

def train(data_path, batch_size, epochs, margin, name, val_split, visualize_data=False, evaluate=False, base_model=None):
    """Trains the Siamese network model."""
    siamese = build_model(input_shape=(224, 224, 3), base_model=base_model)
    siamese.compile(loss=loss(margin=margin), optimizer="adam", metrics=["accuracy"])
    siamese.summary()

    # Define log directory for TensorBoard
    train_log_dir = (f"logs/triplet/model_{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
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

    siamese.save(f"models/contrastive/{name}")

def test(model_path, data_path, batch_size=32):
    """Tests the Siamese network model."""
    model = tf.keras.models.load_model(model_path, custom_objects={'contrastive_loss': loss(margin=1)})
    test_data = create_dataloader(data_path, batch_size=batch_size, val_split=0)
    model.evaluate(test_data)

def _pred_and_visualize(model, img1, img2, img1_path, img2_path, id):
    """Predicts and visualizes the data."""
    prediction = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])

    # Display the images and prediction
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(load_img(img1_path))
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(load_img(img2_path))
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    plt.suptitle(f"Prediction: {prediction[0][0]:.4f}", fontsize=16)
    img1_name = "-".join(img1_path.split('/')[-2:])
    img2_name = "-".join(img2_path.split('/')[-2:])
    plt.savefig(f"data/eval/{id}_{img1_name}#####{img2_name}.png")

    return round(prediction[0][0])

def evaluate_and_predict(model_path, data_path):
    """Evaluates the model and makes a prediction on a single example."""
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'euclidean_distance': euclidean_distance,
            'contrastive_loss': loss(margin=1)
        }
    )

    # Make a prediction with one example
    people_dirs = sorted(os.listdir(data_path))

    total, pred_true, correct_true, correct_false = 0, 0, 0, 0

    for i in range(1000):
        # positive example
        person1_dir = random.choice(people_dirs)
        person1_dir_path = os.path.join(data_path, person1_dir)
        img_1_path = os.path.join(person1_dir_path, random.choice(os.listdir(person1_dir_path)))
        img_2_path = os.path.join(person1_dir_path, random.choice([i for i in os.listdir(person1_dir_path) if i != img_1_path]))
        img_1 = load_img(img_1_path)
        img_2 = load_img(img_2_path)
        pred = _pred_and_visualize(model, img_1, img_2, img_1_path, img_2_path, f"{i}_pos")
        if pred == 0:
            correct_true += 1
            pred_true += 1
        total += 1

        # negative example
        person2_dir = random.choice([p for p in people_dirs if p != person1_dir])
        person2_dir_path = os.path.join(data_path, person2_dir)
        img_2_path = os.path.join(person2_dir_path, random.choice(os.listdir(person2_dir_path)))
        img_2 = load_img(img_2_path)
        pred = _pred_and_visualize(model, img_1, img_2, img_1_path, img_2_path, f"{i}_neg")
        if pred == 1:
            correct_false += 1
        else:
            pred_true += 1
        total += 1

    correct = correct_true + correct_false
    accuracy = correct / total * 100
    same_person_accuracy = correct_true / (total / 2) * 100
    different_person_accuracy = correct_false / (total / 2) * 100
    print(f"Predicted {correct} out of {total} examples correctly. - {accuracy:.2f}%")
    print(f"Predicted {correct_true} out of {total / 2} correctly as same person. - {same_person_accuracy:.2f}%")
    print(f"Predicted {correct_false} out of {total / 2} correctly as different person. - {different_person_accuracy:.2f}%")
    print(f"Predicted {pred_true} out of {total} images as same person. - {pred_true/total*100:.2f}%")

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
    train_parser.add_argument("--base_model", "-bm", type=str, default=None, help="Path to the base model")
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

    verify_parser = subparsers.add_parser("verify", help="Verify the dataloader")
    verify_parser.add_argument("--data", "-d", type=str, default="data/images/train", help="Path to the dataset")
    verify_parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    verify_parser.add_argument("--max_batches", "-m", type=int, default=1, help="Maximum number of batches to verify")
    verify_parser.set_defaults(func=verify_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.name, args.val_split, args.visualize_data, args.evaluate, args.base_model)

def test_handler(args):
    """Handler for the test command."""
    test(args.model, args.data, args.batch_size)

def evaluate_and_predict_handler(args):
    """Handler for the evaluate command."""
    evaluate_and_predict(args.model, args.data)

def verify_handler(args):
    """Handler for the verify command."""
    dataloader = create_dataloader(args.data, batch_size=args.batch_size, val_split=0)
    verify_dataloader(dataloader, max_batch_num=args.max_batches)


if __name__ == '__main__':
    args = parse_args()
    args.func(args)