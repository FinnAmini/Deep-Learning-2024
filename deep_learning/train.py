from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Flatten


def load_dataset_from_directory(
    directory,
    label_directory,
    image_size=(224, 224),
    batch_size=32,
    val_split=0.2,
    shuffle=True,
    random_state=42,
    multi_task=True,
):
    def parse_image(
        file_path, label, age_label=None, gender_label=None, multi_task=False
    ):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])  # Ensure the image tensor has a shape
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        if multi_task:
            return img, (label, age_label, gender_label)
        return img, label

    def load_and_preprocess_image(file_path, label):
        return parse_image(file_path, label, multi_task=False)

    def load_and_preprocess_image_multi_task(file_path, label, age_label, gender_label):
        return parse_image(file_path, label, age_label, gender_label, True)

    # List all subdirectories (class labels)
    class_names = sorted(os.listdir(directory))

    # Initialize lists to store file paths and labels
    file_paths = []
    img_labels = []

    if multi_task:
        age_labels = []
        gender_labels = []

    # Iterate over each class folder
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            # Iterate over each image file in the class folder
            for file in os.scandir(class_folder):
                file_ext = file.name.split(".")[-1]
                if file_ext == "png" or file_ext == "jpg":
                    file_paths.append(file.path)
                    img_labels.append(label)

                    if multi_task:
                        json_file_path = os.path.join(
                            label_directory, file.name.replace(file_ext, "json")
                        )
                        if os.path.exists(json_file_path) and os.path.isfile(
                            json_file_path
                        ):
                            with open(json_file_path) as json_file:
                                json_data = json.load(json_file)
                                age_labels.append(json_data[0]["faceAttributes"]["age"])
                                gender = json_data[0]["faceAttributes"]["gender"]
                                gender_labels.append(0 if gender == "male" else 1)
                        else:
                            age_labels.append(0)  # Default to 0 if no data
                            gender_labels.append(0)  # Default to 0 if no data

    # Convert labels to numpy arrays
    img_labels = np.array(img_labels)

    if multi_task:
        age_labels = np.array(age_labels)
        gender_labels = np.array(gender_labels)

        # # Normalize age labels
        # age_labels = (age_labels - age_labels.min()) / (age_labels.max() - age_labels.min())

        # Split into training and test datasets
        (
            train_file_paths,
            val_file_paths,
            train_img_labels,
            val_img_labels,
            train_age_labels,
            val_age_labels,
            train_gender_labels,
            val_gender_labels,
        ) = train_test_split(
            file_paths,
            img_labels,
            age_labels,
            gender_labels,
            test_size=val_split,
            shuffle=shuffle,
            random_state=random_state,
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_file_paths, train_img_labels, train_age_labels, train_gender_labels)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_file_paths, val_img_labels, val_age_labels, val_gender_labels)
        )

        train_dataset = train_dataset.map(
            load_and_preprocess_image_multi_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        val_dataset = val_dataset.map(
            load_and_preprocess_image_multi_task,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        # Split into training and test datasets
        train_file_paths, val_file_paths, train_img_labels, val_img_labels = (
            train_test_split(
                file_paths,
                img_labels,
                test_size=val_split,
                shuffle=shuffle,
                random_state=random_state,
            )
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_file_paths, train_img_labels)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_file_paths, val_img_labels)
        )

        train_dataset = train_dataset.map(
            load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        val_dataset = val_dataset.map(
            load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Shuffle, batch, and prefetch for performance
    train_dataset = (
        train_dataset.shuffle(buffer_size=1000, seed=random_state)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def build_model(
    model_arch, input_shape, top_layers=[], output_layers=[], freeze=True
) -> Model:
    """Builds a keras model based on the given configuration values

    Args:
        input_shape (_type_): input shape for the base model
        model_arch (_type_): base model architecture
        top_layers (list, optional): top layers for the base model, not including output layers. Defaults to [].
        output_layers (list, optional): output layers for the base model. Defaults to [].
        freeze (bool, optional): determines whether the feature extraction layers shall be freezed. Defaults to True.

    Returns:
        Model: configured keras model
    """
    include_top = len(top_layers) == 0 and len(output_layers) == 0
    base_model = model_arch(
        weights="imagenet", include_top=include_top, input_shape=input_shape
    )

    if freeze:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = Flatten()(x)

    for layer in top_layers:
        x = layer(x)

    outputs = [ol(x) for ol in output_layers]
    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def load_data(path, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

    train_ds = train_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        classes=["faces", "places"],
        shuffle=True,
        seed=42,
    )

    val_ds = train_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        classes=["faces", "places"],
        shuffle=True,
        seed=42,
    )

    return train_ds, val_ds


def multi_task_loss(y_true, y_pred):
    face_detection_true = y_true[0]  # Annahme: 1. Spalte für Gesichtserkennung
    face_detection_pred = y_pred[0]
    age_true = y_true[1]  # Annahme: 2. Spalte für Alter
    age_pred = y_pred[1]
    gender_true = y_true[2]  # Annahme: 3. Spalte für Geschlecht
    gender_pred = y_pred[2]

    face_detection_loss = tf.keras.losses.binary_crossentropy(
        face_detection_true, face_detection_pred
    )

    mask = tf.cast(tf.cast(face_detection_true, tf.float32) > 0.5, tf.float32)

    age_loss = tf.keras.losses.mean_squared_error(age_true, age_pred) * mask
    gender_loss = tf.keras.losses.binary_crossentropy(gender_true, gender_pred) * mask

    total_loss = (
        tf.reduce_mean(face_detection_loss)
        + tf.reduce_mean(age_loss)
        + tf.reduce_mean(gender_loss)
    )
    return total_loss
