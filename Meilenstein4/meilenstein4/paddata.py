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

def create_pairs(root_dir):
    """
    Create a list of image pairs from the dataset. Each pair is a tuple (img1, img2, label).
    If the pair is from the same person, the label is 1 (positive), otherwise it's 0 (negative).
    """
    people = [p for p in os.scandir(root_dir) if p.is_dir()]
    data = defaultdict(list)
    pairs = []

    for label, person in enumerate(people):
        for image_file in os.scandir(person.path):
            if image_file.name.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                data[label].append(image_file)

    for label, image_paths in data.items():
        for img_path in image_paths:
            # positive pair
            img_path2 = random.choice([ip for ip in image_paths if ip != img_path])
            pairs.append((img_path.path, img_path2.path, 0))

            # negative pair
            label2 = random.choice([i for i in range(len(people)) if i != label])
            img_path2 = random.choice(data[label2])
            pairs.append((img_path.path, img_path2.path, 1))

    return pairs

def split_pairs_by_person(pairs: List[Tuple[str, str, int]], val_split: float = 0.2):
    """
    Split the pairs into training and validation sets based on distinct persons (folders).
    """
    person_to_pairs = defaultdict(list)
    for img1, img2, label in pairs:
        person = os.path.dirname(img1)
        person_to_pairs[person].append((img1, img2, label))

    persons = list(person_to_pairs.keys())
    random.shuffle(persons)

    split_index = int(len(persons) * (1 - val_split))
    train_persons = persons[:split_index]
    val_persons = persons[split_index:]

    train_pairs = [pair for person in train_persons for pair in person_to_pairs[person]]
    val_pairs = [pair for person in val_persons for pair in person_to_pairs[person]]

    return train_pairs, val_pairs

def generator(pairs):
    """
    Generator function to yield pairs of images for training.
    """
    for img1_path, img2_path, label in pairs:
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        yield (img1, img2), label

def _create_dataloader(pairs, img_width=224, img_height=224, batch_size=32):
    """
    Create a TensorFlow Dataset for training the Siamese network.
    """
    return tf.data.Dataset.from_generator(
        lambda: generator(pairs),
        output_signature=(
            (tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def create_dataloader(root_dir, img_width=224, img_height=224, batch_size=32, val_split=0.2):
    """
    Create a TensorFlow Dataset for training the Siamese network.
    """
    assert val_split >= 0 and val_split < 1, "Validation split must be between 0 and 1"

    pairs = create_pairs(root_dir)

    if val_split == 0:
        return _create_dataloader(pairs, img_width, img_height, batch_size)
    else:
        train_pairs, val_pairs = split_pairs_by_person(pairs, val_split)
        train_data = _create_dataloader(train_pairs, img_width, img_height, batch_size)
        val_data = _create_dataloader(val_pairs, img_width, img_height, batch_size)
        return train_data, val_data
