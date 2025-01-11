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

def create_pairs(person_dirs):
    """
    Create a list of image pairs from the dataset. Each pair is a tuple (img1, img2, label).
    If the pair is from the same person, the label is 1 (positive), otherwise it's 0 (negative).
    """
    data = defaultdict(list)
    pairs = []
    
    for label, person_dir in enumerate(person_dirs):
        for image_file in os.scandir(person_dir.path):
            if image_file.name.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                data[label].append(image_file.path)

    for label, image_paths in data.items():
        for i in range(len(image_paths) - 1):
            img_path = image_paths[i]

            # positive pair
            img_path2 = image_paths[i + 1]
            pairs.append((img_path, img_path2, 0))

            # negative pair
            label2 = random.choice(list(set(data.keys()) - {label}))
            img_path2 = random.choice(data[label2])
            pairs.append((img_path, img_path2, 1))

        # last positive pair
        img_path = image_paths[-1]
        img_path2 = image_paths[0]
        pairs.append((img_path, img_path2, 0))

        # last negative pair
        label2 = random.choice(list(set(data.keys()) - {label}))
        img_path2 = random.choice(data[label2])
        pairs.append((img_path, img_path2, 1))
            
    return pairs

def split_persons(persons: List, val_split: int = 0.2):
    """
    Split the pairs into training and validation sets.
    """
    split = int(len(persons) * val_split)
    return persons[split:], persons[:split]

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

    person_dirs = [p for p in os.scandir(root_dir) if p.is_dir()]

    if val_split == 0:
        pairs = create_pairs(person_dirs)
        return _create_dataloader(pairs, img_width, img_height, batch_size)
    else:
        train_persons, val_persons = split_persons(person_dirs, val_split)
        train_pairs = create_pairs(train_persons)
        val_pairs = create_pairs(val_persons)
        train_data = _create_dataloader(train_pairs, img_width, img_height, batch_size)
        val_data = _create_dataloader(val_pairs, img_width, img_height, batch_size)
        return train_data, val_data

def verify_dataloader(dataloader, max_batch_num = -1):
    """Verifies that a dataloader works correctly and the positive/negative split is around 50%"""
    positive_count = 0
    negative_count = 0
    total_samples = 0

    for i, ((img1, img2), label) in enumerate(dataloader):
        # Break after checking the specified number of batches
        if max_batch_num > 0 and i >= max_batch_num:
            break

        if max_batch_num > 0:
            print(f"Evaluating batch {i + 1} / {max_batch_num}...")
        else:
            print(f"Evaluating batch {i + 1}...")

        # Check the shapes and types of the inputs
        assert img1.shape == img2.shape, f"Image pair shapes don't match: {img1.shape} vs {img2.shape}"
        assert img1.shape[1:] == (224, 224, 3), f"Unexpected image shape: {img1.shape[1:]}"
        assert label.dtype == tf.int32, f"Unexpected label type: {label.dtype}"

        # Count positive and negative labels
        negative_count += tf.reduce_sum(label).numpy()
        positive_count += (label.shape[0] - tf.reduce_sum(label).numpy())
        total_samples += label.shape[0]

    # Calculate the percentage of positive and negative samples
    positive_percentage = (positive_count / total_samples) * 100
    negative_percentage = (negative_count / total_samples) * 100

    print(f"Total samples checked: {total_samples}")
    print(f"Positive pairs: {positive_count} ({positive_percentage:.2f}%)")
    print(f"Negative pairs: {negative_count} ({negative_percentage:.2f}%)")

    # Verify that the split is roughly 50%
    assert abs(positive_percentage - 50) <= 5, "Positive/negative split is not balanced (outside 45-55%)"
    print("Dataloader verification passed.")


# finn
# data/vgg_faces2_resized/train/n000397/0045_01_resized.png data/vgg_faces2_resized/train/n000280/0114_02_resized.png 1
# data/vgg_faces2_resized/train/n000397/0147_01_resized.png data/vgg_faces2_resized/train/n000397/0265_01_resized.png 0
# data/vgg_faces2_resized/train/n000397/0147_01_resized.png data/vgg_faces2_resized/train/n000157/0631_02_resized.png 1
# data/vgg_faces2_resized/train/n000397/0265_01_resized.png data/vgg_faces2_resized/train/n000397/0036_04_resized.png 0
# data/vgg_faces2_resized/train/n000397/0265_01_resized.png data/vgg_faces2_resized/train/n000190/0342_01_resized.png 1
# data/vgg_faces2_resized/train/n000397/0036_04_resized.png data/vgg_faces2_resized/train/n000397/0175_01_resized.png 0
# data/vgg_faces2_resized/train/n000397/0036_04_resized.png data/vgg_faces2_resized/train/n000299/0129_01_resized.png 1
# data/vgg_faces2_resized/train/n000397/0175_01_resized.png data/vgg_faces2_resized/train/n000397/0456_01_resized.png 0
# data/vgg_faces2_resized/train/n000397/0175_01_resized.png data/vgg_faces2_resized/train/n000130/0241_01_resized.png 1
# data/vgg_faces2_resized/train/n000397/0456_01_resized.png data/vgg_faces2_resized/train/n000397/0037_01_resized.png 0

# data/vgg_faces2_resized/train/n000094/0067_01_resized.png data/vgg_faces2_resized/train/n000277/0010_02_resized.png 1
# data/vgg_faces2_resized/train/n000094/0019_01_resized.png data/vgg_faces2_resized/train/n000094/0323_01_resized.png 0
# data/vgg_faces2_resized/train/n000094/0019_01_resized.png data/vgg_faces2_resized/train/n000011/0221_01_resized.png 1
# data/vgg_faces2_resized/train/n000094/0027_01_resized.png data/vgg_faces2_resized/train/n000094/0219_01_resized.png 0
# data/vgg_faces2_resized/train/n000094/0027_01_resized.png data/vgg_faces2_resized/train/n000305/0470_01_resized.png 1
# data/vgg_faces2_resized/train/n000094/0082_01_resized.png data/vgg_faces2_resized/train/n000094/0388_01_resized.png 0
# data/vgg_faces2_resized/train/n000094/0082_01_resized.png data/vgg_faces2_resized/train/n000482/0043_02_resized.png 1
# data/vgg_faces2_resized/train/n000094/0187_01_resized.png data/vgg_faces2_resized/train/n000094/0215_01_resized.png 0
# data/vgg_faces2_resized/train/n000094/0187_01_resized.png data/vgg_faces2_resized/train/n000146/0108_01_resized.png 1
# data/vgg_faces2_resized/train/n000094/0079_01_resized.png data/vgg_faces2_resized/train/n000094/0203_01_resized.png 0