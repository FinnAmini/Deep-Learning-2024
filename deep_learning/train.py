from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow.keras.preprocessing.image as keras_image
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_dataset_from_directory(
    directory,
    image_size=(224, 224),
    batch_size=32,
    val_split=0.2,
    shuffle=True,
    random_state=42,
    multi_task=False,
):
    # List all subdirectories (class labels)
    class_names = sorted(os.listdir(directory))

    # Initialize lists to store image data and labels
    images = []
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
                    img = keras_image.load_img(file.path, target_size=image_size)
                    img_array = keras_image.img_to_array(img)
                    images.append(img_array)
                    img_labels.append(label)

                    if multi_task:
                        json_file_path = file.path.replace(file_ext, "json")
                        if os.path.exists(json_file_path) and os.path.isfile(
                            json_file_path
                        ):
                            with open(json_file_path) as json_file:
                                json_data = json.load(json_file)
                                age_labels.append(json_data[0]["faceAttributes"]["age"])
                                gender = json_data[0]["faceAttributes"]["gender"]
                                gender_labels.append(0 if gender == "male" else 1)
                        else:
                            age_labels.append(-1)
                            gender_labels.append(-1)

    # Convert images and labels to numpy arrays
    images = np.array(images)
    img_labels = np.array(img_labels)

    # Normalize images to the range [0, 1]
    images = images.astype("float32") / 255.0

    if multi_task:
        age_labels = np.array(age_labels)
        gender_labels = np.array(gender_labels)

        # Split into training and test datasets
        x_train, x_val, y_train, y_val, age_train, age_val, gender_train, gender_val = (
            train_test_split(
                images,
                img_labels,
                age_labels,
                gender_labels,
                test_size=val_split,
                shuffle=shuffle,
                random_state=random_state,
            )
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((x_train,), (y_train, age_train, gender_train))
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ((x_val,), (y_val, age_val, gender_val))
        )
    else:
        # Split into training and test datasets
        x_train, x_val, y_train, y_val = train_test_split(
            images,
            img_labels,
            test_size=val_split,
            shuffle=shuffle,
            random_state=random_state,
        )

        # Create TensorFlow Datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

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

    if len(output_layers) > 1:
        x = GlobalAveragePooling2D()(x)

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
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

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
