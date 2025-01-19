import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict
from typing import List, Tuple
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_image(image_path, img_width=224, img_height=224):
    """
    Load an image from the path and convert it to a tensor.
    Resize the image to the required dimensions and normalize it.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    return img

def create_triplets(root_dir):
    """
    Create a list of image pairs from the dataset. Each pair is a tuple (img1, img2, label).
    If the pair is from the same person, the label is 1 (positive), otherwise it's 0 (negative).
    """
    people = [p for p in os.scandir(root_dir) if p.is_dir()]
    data = defaultdict(list)
    triplets = []

    for label, person in enumerate(people):
        for image_file in os.scandir(person.path):
            if image_file.name.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                data[label].append(image_file.path)

    for label, image_paths in data.items():
        for anchor in image_paths:
            positive = random.choice([ip for ip in image_paths if ip != anchor])
            positive2 = random.choice([ip for ip in image_paths if ip != anchor and ip != positive])
            positive3 = random.choice([ip for ip in image_paths if ip != anchor and ip != positive and ip != positive2])
            negative_person = random.choice([i for i in range(len(people)) if i != label])
            negative_person2 = random.choice([i for i in range(len(people)) if i != label and i != negative_person])
            negative_person3 = random.choice([i for i in range(len(people)) if i != label and i != negative_person and i != negative_person2])
            negative = random.choice(data[negative_person])
            negative2 = random.choice(data[negative_person2])
            negative3 = random.choice(data[negative_person3])
            triplets.append((anchor, positive, negative))
            triplets.append((anchor, positive2, negative2))
            triplets.append((anchor, positive3, negative3))

    return triplets

def split_triplets_by_person(triplets: List[Tuple[str, str, str]], val_split: float = 0.2):
    """
    Split the triplets into training and validation sets based on distinct persons (folders).
    """
    person_to_triplets = defaultdict(list)
    for anchor, positive, negative in triplets:
        person = os.path.dirname(anchor)
        person_to_triplets[person].append((anchor, positive, negative))

    persons = list(person_to_triplets.keys())
    random.shuffle(persons)

    split_index = int(len(persons) * (1 - val_split))
    train_persons = persons[:split_index]
    val_persons = persons[split_index:]

    train_triplets = [triplet for person in train_persons for triplet in person_to_triplets[person]]
    val_triplets = [triplet for person in val_persons for triplet in person_to_triplets[person]]

    return train_triplets, val_triplets

def triplet_generator(triplets):
    """
    Generator function to yield triplets of images for training.
    """
    for anchor, positive, negative in triplets:
        yield (
            load_image(anchor),
            load_image(positive),
            load_image(negative),
        )

def create_triplet_dataloader(triplets, img_width=224, img_height=224, batch_size=32, repeat=False):
    """
    Create a tf.data.Dataset from a triplet data generator with optional repetition.
    """
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: triplet_generator(triplets),
        output_signature=(
            tf.TensorSpec(shape=(img_width, img_height, 3), dtype=tf.float32),  # Anchor
            tf.TensorSpec(shape=(img_width, img_height, 3), dtype=tf.float32),  # Positive
            tf.TensorSpec(shape=(img_width, img_height, 3), dtype=tf.float32),  # Negative
        ),
    ).shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()  # Enable infinite repetition for training

    return dataset


def create_dataloader(root_dir, img_width=224, img_height=224, batch_size=32, val_split=0.2):
    """
    Create a TensorFlow Dataset for training the Siamese network.
    """
    assert val_split >= 0 and val_split < 1, "Validation split must be between 0 and 1"

    triplets = create_triplets(root_dir)

    if val_split == 0:
        return create_triplet_dataloader(triplets, img_width, img_height, batch_size)
    else:
        train_triplets, val_triplets = split_triplets_by_person(triplets, val_split)
        steps_per_epch = len(train_triplets) // batch_size
        train_data = create_triplet_dataloader(train_triplets, img_width, img_height, batch_size, repeat=True)
        val_data = create_triplet_dataloader(val_triplets, img_width, img_height, batch_size, repeat=False)
        return train_data, val_data, steps_per_epch
