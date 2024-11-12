from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing import image


def create_dataset(
    image_size, img_paths, labels, age_labels=None, gender_labels=None, batch_size=32
):
    def parse_image(
        file_path,
        label,
        age_label=None,
        gender_label=None,
        multi_task=False,
    ):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        if multi_task:
            return img, (label, age_label, gender_label)
        return img, label

    def load_and_preprocess_image(image_size, file_path, label):
        return parse_image(image_size, file_path, label, multi_task=False)

    def load_and_preprocess_image_multi_task(file_path, label, age_label, gender_label):
        return parse_image(file_path, label, age_label, gender_label, True)

    if age_labels is not None and gender_labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices(
            (img_paths, labels, age_labels, gender_labels)
        )
        map_fn = load_and_preprocess_image_multi_task
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        map_fn = load_and_preprocess_image

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = (
        dataset.shuffle(buffer_size=1000, seed=42)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


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
    class_names = sorted(os.listdir(directory))
    file_paths = []
    img_labels = []

    if multi_task:
        age_labels = []
        gender_labels = []

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
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
                            age_labels.append(-1)
                            gender_labels.append(-1)

    img_labels = np.array(img_labels)

    if multi_task:
        age_labels = np.array(age_labels)
        gender_labels = np.array(gender_labels)

        if val_split is not None:
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

            train_dataset = create_dataset(
                image_size,
                train_file_paths,
                train_img_labels,
                train_age_labels,
                train_gender_labels,
                batch_size,
            )

            val_dataset = create_dataset(
                image_size,
                val_file_paths,
                val_img_labels,
                val_age_labels,
                val_gender_labels,
                batch_size,
            )
            return train_dataset, val_dataset
        else:
            test_dataset = create_dataset(
                image_size,
                file_paths,
                img_labels,
                age_labels,
                gender_labels,
                batch_size,
            )
            return test_dataset
    else:
        if val_split is None:
            train_file_paths, val_file_paths, train_img_labels, val_img_labels = (
                train_test_split(
                    file_paths,
                    img_labels,
                    test_size=val_split,
                    shuffle=shuffle,
                    random_state=random_state,
                )
            )
            train_dataset = create_dataset(
                image_size, train_file_paths, train_img_labels, batch_size=batch_size
            )
            val_dataset = create_dataset(
                image_size, val_file_paths, val_img_labels, batch_size=batch_size
            )
            return train_dataset, val_dataset
        else:
            test_dataset = create_dataset(
                image_size, file_paths, img_labels, batch_size=batch_size
            )
            return test_dataset


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


def load_data(path, batch_size=32, val_split=0.2):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=val_split,
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


@register_keras_serializable()
def multi_task_loss(y_true, y_pred):
    face_detection_true = y_true[0]
    face_detection_pred = y_pred[0]
    age_true = y_true[1]
    age_pred = y_pred[1]
    gender_true = y_true[2]
    gender_pred = y_pred[2]

    face_detection_loss = tf.keras.losses.binary_crossentropy(
        face_detection_true, face_detection_pred
    )

    mask = tf.cast(tf.cast(face_detection_true, tf.float32) > 0.5, tf.float32)

    age_loss = tf.keras.losses.MeanSquaredError()(age_true, age_pred) * mask
    gender_loss = tf.keras.losses.binary_crossentropy(gender_true, gender_pred) * mask

    total_loss = (
        tf.reduce_mean(face_detection_loss)
        + tf.reduce_mean(age_loss)
        + tf.reduce_mean(gender_loss)
    )
    return total_loss


@register_keras_serializable()
def custom_loss_age(y_true, y_pred):
    face_detection_true = y_true[0]
    face_detection_pred = y_pred[0]
    age_true = y_true[1]
    age_pred = y_pred[1]
    gender_true = y_true[2]
    gender_pred = y_pred[2]

    face_detection_loss = tf.keras.losses.binary_crossentropy(
        face_detection_true, face_detection_pred
    )

    mask = tf.cast(tf.cast(face_detection_true, tf.float32) > 0.5, tf.float32)

    age_loss = tf.keras.losses.MeanSquaredError()(age_true, age_pred) * mask
    gender_loss = tf.keras.losses.binary_crossentropy(gender_true, gender_pred) * mask

    total_loss = (
        tf.reduce_mean(face_detection_loss)
        + tf.reduce_mean(age_loss)
        + tf.reduce_mean(gender_loss)
    )
    return total_loss


def eval(model: str, images: str, labels: str):
    try:
        if labels is not None:
            model = load_model(
                model, custom_objects={"multi_task_loss": multi_task_loss}
            )
        else:
            model = load_model(model)
    except Exception as e:
        raise ValueError(f"Invalid model name {model}!") from e

    try:
        if labels is not None:
            test_data = load_dataset_from_directory(
                images, labels, batch_size=64, multi_task=True, val_split=None
            )
        else:
            test_data = load_data(images, batch_size=64)
    except Exception as e:
        raise ValueError(
            f"Invalid images_path {images} or labels_path {labels}!"
        ) from e

    if labels is not None:
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=multi_task_loss,
            metrics={
                "face_detection": "accuracy",
                "age_prediction": "mae",
                "gender_classification": "accuracy",
            },
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    model.evaluate(test_data, verbose=2)


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict(model, img_array):
    return model.predict(img_array)


def predict_single_image(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = predict(model, img_array)
    gender_pred = predictions[2][0]
    return 0 if gender_pred < 0.5 else 1

    # print(f"Face Detection: {'Face' if face_detection_pred < 0.5 else 'No Face', face_detection_pred}")
    # print(f"Predicted Age: {age_pred}")
    # print(f"Predicted Gender: {gender, gender_pred}")


if __name__ == "__main__":
    model_path = "models/st_resnet50_adam_lr=0.0001_lc1_freeze=True_test_3.keras"
    img_path = "data/training/faces/00001.png"
    predict_single_image(model_path, img_path)
