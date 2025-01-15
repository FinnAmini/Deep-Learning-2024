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
from collections import defaultdict
import json
import shutil

def calc_distance(anchor, reference):
    """Calculates the distance between the anchor and the reference embeddings."""
    return ops.sum(tf.square(anchor - reference), -1)

@keras.saving.register_keras_serializable()
class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = calc_distance(anchor, positive)
        an_distance = calc_distance(anchor, negative)
        return (ap_distance, an_distance)

def build_embedding(input_shape):
    """Builds the embedding network."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = True  
    x = Flatten()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
    embedding = Model(base_model.input, x)
    return embedding


def build_model(input_shape):
    """Builds the Siamese network model."""
    anchor_input = layers.Input(name="anchor", shape=input_shape)
    positive_input = layers.Input(name="positive", shape=input_shape)
    negative_input = layers.Input(name="negative", shape=input_shape)
    embedding = build_embedding(input_shape)

    a = embedding(anchor_input),
    b = embedding(positive_input)
    c = embedding(negative_input)

    # Compute the distance between the embeddings
    distances = DistanceLayer()(a, b, c)

    # Build the Siamese model
    siamese = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return siamese, embedding

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
        siamese, embedding = build_model((224, 224, 3))
        self.siamese_network = siamese
        self.embedding = embedding 
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.accuracy_tracker = metrics.Mean(name="accuracy")
        self.use_embedding = False

    def set_use_embedding(self, use):
        self.use_embedding = use

    def call(self, inputs):
        if self.use_embedding:
            return self.embedding(inputs)
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

    def set_eval(self):
        self.siamese_network.eval()

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]


def train(data_path, batch_size, epochs, margin, name, val_split, lr=0.0001):
    siamese_model = SiameseModel(margin=margin)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(lr))

    train_log_dir = (f"logs/triplet/model_{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
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
    model = load_model(model_path)
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
    model = load_model(model_path)
    test_data = create_dataloader(data_path, batch_size=batch_size, val_split=0)
    model.evaluate(test_data)

def get_people_paths(data_path):
    people = [p for p in os.scandir(data_path) if p.is_dir()]
    data = defaultdict(list)
    for person in people:
        for image_file in os.scandir(person.path):
            if image_file.name.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                data[person.name].append(image_file.path)
    return data

def pred_all(model, data_path):
    """Predicts all data in the dataloader."""
    data = get_people_paths(data_path)
    pos_preds, neg_preds = [], []

    for label, image_paths in data.items():
        anchors, pos, neg = [], [], []
        print(label, len(image_paths))
        for anchor in image_paths:
            positive = random.choice([ip for ip in image_paths if ip != anchor])
            negative_person = random.choice([l for l in data.keys() if l != label])
            negative = random.choice(data[negative_person])

            anchor_img = load_image(anchor)
            positive_img = load_image(positive)
            negative_img = load_image(negative)

            anchors.append(anchor_img)
            pos.append(positive_img)
            neg.append(negative_img)

            break

        anchors = np.array(anchors)
        pos = np.array(pos)
        neg = np.array(neg)
        print(anchors.shape, pos.shape, neg.shape)
        pred = model.predict([anchors, pos, neg])
        pos_preds.extend(pred[0])
        neg_preds.extend(pred[1])
    return pos_preds, neg_preds

def find_optimal_threshold(positive_distances, negative_distances):
    """Find the threshold that maximizes accuracy."""
    thresholds = np.linspace(0, max(max(positive_distances), max(negative_distances)), 1000)
    best_threshold, best_accuracy = 0, 0
    pos_acc, neg_acc = 0, 0
    print(f"Evaluating {len(positive_distances)} positive and {len(negative_distances)} negative distances.")
    
    for threshold in thresholds:
        positive_correct = positive_distances < threshold
        negative_correct = negative_distances >= threshold
        
        # Accuracy
        accuracy = (positive_correct.sum() + negative_correct.sum()) / (len(positive_distances) + len(negative_distances))
        pos_accuracy = positive_correct.sum() / len(positive_distances)
        neg_accuracy = negative_correct.sum() / len(negative_distances)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            pos_acc = pos_accuracy
            neg_acc = neg_accuracy
    
    return best_threshold, best_accuracy, pos_acc, neg_acc

def load_model(model_path, use_embedding=False):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"SiameseModel": SiameseModel, "DistanceLayer": DistanceLayer}
    )
    if use_embedding:
        model.set_use_embedding(True)
    return model

def find_threshold(model_path, data_path):
    """Function for find a good threshold"""
    model = load_model(model_path)
    pos, neg = pred_all(model, data_path)
    thresh, acc, pos_acc, neg_acc = find_optimal_threshold(pos, neg)

    print(f"Best accuracy: {acc:.4f} with threshold: {thresh:.4f}")
    print(f"Positive accuracy: {pos_acc:.4f}, Negative accuracy: {neg_acc:.4f}")

def save_embeddings(model_path, data_path):
    """Function to get embeddings"""
    model = load_model(model_path, True)
    data = get_people_paths(data_path)
    embeddings = {}
    paths, ank = [], []

    for image_paths in data.values():
        for anchor in image_paths:
            anchor_img = load_image(anchor)
            paths.append(anchor)
            ank.append(anchor_img)

    anchor_embedding = model.predict([np.array(ank), np.array(ank), np.array(ank)])
    for i, path in enumerate(paths):
        embeddings[path] = anchor_embedding[i].tolist()
    
    with open('data/json/db.json', 'w') as file:
        json.dump(embeddings, file, indent=4)

def add_embedding(model, data_path):
    """Function to add embeddings"""
    if type(model) == str:
        model = load_model(model, True)
    with open('data/json/db.json', 'r') as file:
        embeddings = json.load(file)
    img = load_image(data_path)
    embedding = model.predict(np.array(img))
    embeddings[data_path] = embedding.tolist()
    with open('data/json/db.json', 'w') as file:
        json.dump(embeddings, file, indent=4)

def recognize(model, img_path):
    """Function to recognize people"""
    if not os.path.exists('data/json/db.json'):
        print("No embeddings found. Please run the embedding command first.")
        return

    if type(model) == str:
        model = load_model(model, True)
    closest, furthest, same = (None, float("inf")), (None, 0), (None, 0)
    with open('data/json/db.json', 'r') as file:
        embeddings = json.load(file)
    
    img = load_image(img_path)
    embedding = model.predict(np.array([img]))
    for path, emb in embeddings.items():
        distance = calc_distance(embedding, np.array(emb))[0]

        if path == img_path:
            same = (path, distance)
        else:
            if distance < closest[1]:
                closest = (path, distance)
            if distance > furthest[1]:
                furthest = (path, distance)

    if same[0]:
        print(f"Same person: {same[0]} with distance {same[1]:.4f}")
    print(f"Closest match: {closest[0]} with distance {closest[1]:.4f}")
    print(f"Furthest match: {furthest[0]} with distance {furthest[1]:.4f}")

    name = Path(img_path).parent.name + Path(img_path).stem
    os.makedirs(f'data/compared/recognition/{name}', exist_ok=True)
    shutil.copy(img_path, f'data/compared/recognition/{name}/input.jpg')
    shutil.copy(closest[0], f'data/compared/recognition/{name}/closest_{closest[1]}.jpg')
    shutil.copy(furthest[0], f'data/compared/recognition/{name}/furthest_{furthest[1]}.jpg')
    if same[0]:
        shutil.copy(same[0], f'data/compared/recognition/{name}/same_{same[1]}.jpg')



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
    train_parser.add_argument("--learning_rate", "-lr", type=float, default=0.0001, help="Learning rate")
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

    threshold_parser = subparsers.add_parser("threshold")
    threshold_parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    threshold_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    threshold_parser.set_defaults(func=threshold_handler)

    embedding_parser = subparsers.add_parser("embedding")
    embedding_parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    embedding_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    embedding_parser.set_defaults(func=embedding_handler)

    recognize_parser = subparsers.add_parser("recognize")
    recognize_parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    recognize_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    recognize_parser.set_defaults(func=recognize_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.name, args.val_split, args.learning_rate)

def test_handler(args):
    """Handler for the test command."""
    test(args.model, args.data, args.batch_size)

def evaluate_and_predict_handler(args):
    """Handler for the evaluate command."""
    evaluate_and_predict(args.model, args.data)

def threshold_handler(args):
    """Handler for the threshold command."""
    find_threshold(args.model, args.data)

def embedding_handler(args):
    """Handler for the embedding command."""
    save_embeddings(args.model, args.data)

def recognize_handler(args):
    """Handler for the recognize command."""
    recognize(args.model, args.data)

if __name__ == '__main__':
    args = parse_args()
    args.func(args)