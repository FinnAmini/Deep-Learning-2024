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
import sys

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

def build_embedding(input_shape, base_model):
    """Builds the embedding network."""
    embedding = False
    if base_model is None:
        print("Loading ResNet50 model")
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
    else:
        print("Loading ms2 model")
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
        base_model.trainable = False
        x = BatchNormalization()(base_model.output)
        x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
        embedding = Model(base_model.input, x)
        embedding.summary()
        return embedding


def build_model(input_shape, base_model):
    """Builds the Siamese network model."""
    anchor_input = layers.Input(name="anchor", shape=input_shape)
    positive_input = layers.Input(name="positive", shape=input_shape)
    negative_input = layers.Input(name="negative", shape=input_shape)
    embedding = build_embedding(input_shape, base_model)

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

    def __init__(self, siamese_network=None, margin=0.2, base_model=None, **kwargs):
        super().__init__()
        siamese, embedding = build_model((224, 224, 3), base_model)
        self.siamese_network = siamese
        self.embedding = embedding 
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.accuracy_tracker = metrics.Mean(name="accuracy")
        self.use_embedding = False

    def set_use_embedding(self, use):
        self.use_embedding = use

    def set_model(self, model):
        self.siamese_network = model

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


def train(data_path, batch_size, epochs, margin, name, val_split, lr=0.00001, base_model=None, resume_from=None):
    # Load model from checkpoint if resume_from is specified
    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
        siamese_model = tf.keras.models.load_model(
            resume_from,
            custom_objects={"SiameseModel": SiameseModel, "DistanceLayer": DistanceLayer}
        )
    else:
        print("Starting training from scratch.")
        siamese_model = SiameseModel(margin=margin, base_model=base_model)
        siamese_model.compile(optimizer=tf.keras.optimizers.Adam(lr))

    # Adjust log directory for continuation
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = f"logs/triplet/model_{name}_{timestamp}"
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

    # Create data loaders
    dataloaders = create_dataloader(data_path, batch_size=batch_size, val_split=val_split)

    # Determine starting epoch for resuming
    initial_epoch = 0
    if resume_from:
        # Extract initial epoch from checkpoint file name
        try:
            initial_epoch = int(resume_from.split('_epoch_')[1].split('_')[0])
            print(f"Resuming from epoch {initial_epoch}")
        except IndexError:
            print("Unable to parse epoch number from checkpoint. Starting from epoch 0.")

    # Train the model
    siamese_model.fit(
        dataloaders[0],
        validation_data=dataloaders[1],
        epochs=epochs,
        steps_per_epoch=dataloaders[2],
        initial_epoch=initial_epoch,  # Start from the right epoch
        callbacks=[tensorboard_train_callback, checkpoint_callback]
    )
    siamese_model.save(f"models/triplet/{name}")

def _pred_and_visualize_triplet(model, anker_img, pos_imgs, neg_imgs, anker_path, pos_paths, neg_paths, id):
    """
    Predicts distances between the anchor, 3 positives, and 3 negatives, and visualizes the results.

    Args:
        model: The trained model.
        anker_img: The anchor image as a NumPy array.
        pos_imgs: A list of 3 positive images as NumPy arrays.
        neg_imgs: A list of 3 negative images as NumPy arrays.
        anker_path: Path to the anchor image.
        pos_paths: List of paths to the positive images.
        neg_paths: List of paths to the negative images.
        id: Unique identifier for saving the result.

    Returns:
        Tuple containing the anchor-positive distances and anchor-negative distances.
    """
    # Prepare the inputs for the model prediction
    batch_anker = np.expand_dims(anker_img, axis=0)  # Shape: (1, height, width, channels)
    
    # Predict distances for each positive image
    ap_distances = []
    for pos_img in pos_imgs:
        batch_pos = np.expand_dims(pos_img, axis=0)
        batch_neg = np.expand_dims(neg_imgs[0], axis=0)  # Dummy negative input for shape consistency
        prediction = model.predict([batch_anker, batch_pos, batch_neg])

        # Extract anchor-positive distance
        ap_distance = prediction[0][0].item()  # Anchor-Positive Distance
        ap_distances.append(ap_distance)

    # Predict distances for each negative image
    an_distances = []
    for neg_img in neg_imgs:
        batch_neg = np.expand_dims(neg_img, axis=0)
        batch_pos = np.expand_dims(pos_imgs[0], axis=0)  # Dummy positive input for shape consistency
        prediction = model.predict([batch_anker, batch_pos, batch_neg])

        # Extract anchor-negative distance
        an_distance = prediction[1][0].item()  # Anchor-Negative Distance
        an_distances.append(an_distance)

    # Create a plot to visualize the anchor, 3 positives, and 3 negatives
    fig, axes = plt.subplots(1, 7, figsize=(24, 4))  # 1 row, 7 columns (1 anchor + 3 positives + 3 negatives)
    
    # Anchor image
    axes[0].imshow(load_img(anker_path))
    axes[0].set_title("Anchor")
    axes[0].axis("off")
    
    # Positive images
    for i, (pos_path, distance) in enumerate(zip(pos_paths, ap_distances)):
        axes[i + 1].imshow(load_img(pos_path))
        axes[i + 1].set_title(f"{distance:.4f}", weight='bold', fontsize=20)
        axes[i + 1].axis("off")

    # Negative images
    for i, (neg_path, distance) in enumerate(zip(neg_paths, an_distances)):
        axes[i + 4].imshow(load_img(neg_path))
        axes[i + 4].set_title(f"{distance:.4f}", weight='bold', fontsize=20)
        axes[i + 4].axis("off")

    # Set the main title for the whole figure
    plt.suptitle(f"Triplet Evaluation", fontsize=16)
    
    # Save the figure
    plt.savefig(f"data/eval/triplet/{id}_triplet_eval.png")
    plt.close()  # Close the plot to avoid memory issues

    return ap_distances, an_distances

def evaluate_and_predict(model_path, data_path):
    """Evaluates the model and makes a prediction on a single example."""
    model = load_model(model_path)
    people_dirs = sorted(os.listdir(data_path))

    total, good, ap_dist_sum, an_dist_sum = 0, 0, 0, 0

    for i in range(1000):
        # Anchor and positive examples
        person1_dir = random.choice(people_dirs)
        person1_dir_path = os.path.join(data_path, person1_dir)
        anker_path = os.path.join(person1_dir_path, random.choice(os.listdir(person1_dir_path)))

        # Select 3 positive images from the same person
        pos_paths = random.sample(
            [os.path.join(person1_dir_path, img) for img in os.listdir(person1_dir_path) if img != anker_path],
            k=3
        )
        pos_imgs = [load_image(pos_path) for pos_path in pos_paths]

        # Select 3 negative images from other people
        neg_paths = []
        neg_imgs = []
        for _ in range(3):
            person2_dir = random.choice([p for p in people_dirs if p != person1_dir])
            person2_dir_path = os.path.join(data_path, person2_dir)
            neg_path = os.path.join(person2_dir_path, random.choice(os.listdir(person2_dir_path)))
            neg_paths.append(neg_path)
            neg_imgs.append(load_image(neg_path))

        anker_img = load_image(anker_path)

        # Perform prediction for the anchor, 3 positives, and 3 negatives
        ap_distances, an_distances = _pred_and_visualize_triplet(
            model, anker_img, pos_imgs, neg_imgs, anker_path, pos_paths, neg_paths, f"{i}_triplet"
        )

        # Evaluate accuracy
        for ap_distance in ap_distances:
            for an_distance in an_distances:
                total += 1
                if ap_distance < an_distance:
                    good += 1

        ap_dist_sum += sum(ap_distances)
        an_dist_sum += sum(an_distances)

    print(f"Predicted {good} out of {total} examples correctly. - {good/total*100:.2f}%")
    print(f"Avg. AP distance: {ap_dist_sum/total:.4f}")
    print(f"Avg. AN distance: {an_dist_sum/total:.4f}")

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
        print(label, len(image_paths))
        anchors, pos, neg = [], [], []
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

        anchors = np.array(anchors)
        pos = np.array(pos)
        neg = np.array(neg)
        pred = model.predict([anchors, pos, neg])

        if isinstance(pred[0], tf.RaggedTensor):
            pos_flattened = pred[0].flat_values.numpy().tolist()
            neg_flattened = pred[1].flat_values.numpy().tolist()
        else:
            pos_flattened = pred[0].flatten().tolist()
            neg_flattened = pred[1].flatten().tolist()
        pos_preds.extend(pos_flattened)
        neg_preds.extend(neg_flattened)

    return pos_preds, neg_preds

def find_optimal_threshold(positive_distances, negative_distances):
    """
    Find the threshold that maximizes accuracy, assuming inputs are always tf.RaggedTensor.
    """
    # Calculate the maximum values for threshold range
    max_pos_dist = np.max(positive_distances)
    max_neg_dist = np.max(negative_distances)
    
    # Generate thresholds to evaluate
    thresholds = np.linspace(0, max(max_pos_dist, max_neg_dist), 1000)
    best_threshold, best_accuracy = 0, 0
    pos_acc, neg_acc = 0, 0
    
    print(f"Evaluating {len(positive_distances)} positive and {len(negative_distances)} negative distances.")
    
    # Evaluate thresholds
    for threshold in thresholds:
        positive_correct = positive_distances < threshold
        negative_correct = negative_distances >= threshold
        
        # Compute accuracy metrics
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

def save_embeddings(model_path, data_path, output_name):
    """Function to get embeddings"""
    model = load_model(model_path, True)
    data = get_people_paths(data_path)
    embedding_map = {}
    paths, images = [], []
    embedding_map = {}
    paths, images = [], []

    for image_paths in data.values():
        for img_path in image_paths:
            img = load_image(img_path)
            paths.append(img_path)
            images.append(img)
        for img_path in image_paths:
            img = load_image(img_path)
            paths.append(img_path)
            images.append(img)

    print(img_path)
    embeddings = model.predict(np.array(images))
    for i, path in enumerate(paths):
        key = "/".join(path.split("/")[-2:])
        embedding_map[key] = embeddings[i].tolist()

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

def create_subset(data_path, output_path):
    """Function to create a subset of the data"""
    people = os.listdir(data_path)
    os.makedirs(output_path, exist_ok=True)
    for person in people:
        person_path = os.path.join(data_path, person)
        person_target_path = os.path.join(output_path, person)
        os.makedirs(person_target_path, exist_ok=True)
        for i, image in enumerate(os.scandir(person_path)):
            if i < 5:
                target_path = os.path.join(person_target_path, image.name)
                shutil.copy(image.path, target_path)

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
    train_parser.add_argument("--learning_rate", "-lr", type=float, default=0.00001, help="Learning rate")
    train_parser.add_argument("--base_model", "-bm", type=str, default=None, help="Path to the base model")
    train_parser.add_argument("-val_split", type=float, default=0.2, help="Validation split")
    train_parser.add_argument("--resume_from", "-rf", type=str, default=None, help="Path to a checkpoint to resume training")
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
    embedding_parser.add_argument("--output", "-o", type=str, default="database", help="Name of the output file")
    embedding_parser.set_defaults(func=embedding_handler)

    recognize_parser = subparsers.add_parser("recognize")
    recognize_parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    recognize_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    recognize_parser.set_defaults(func=recognize_handler)

    subset_parser = subparsers.add_parser("subset")
    subset_parser.add_argument("--data", "-d", type=str, default="data/images/test", help="Path to the dataset")
    subset_parser.add_argument("--output", "-o", type=str, default="data/subset", help="Path to the output directory")
    subset_parser.set_defaults(func=subset_handler)
    
    return parser.parse_args()

def train_handler(args):
    """Handler for the train command."""
    train(args.data, args.batch_size, args.epochs, args.margin, args.name, args.val_split, args.learning_rate, args.base_model, args.resume_from)


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
    save_embeddings(args.model, args.data, args.output)

def recognize_handler(args):
    """Handler for the recognize command."""
    recognize(args.model, args.data)

def subset_handler(args):
    """Handler for the subset command."""
    create_subset(args.data, args.output)

if __name__ == '__main__':
    args = parse_args()
    args.func(args)
