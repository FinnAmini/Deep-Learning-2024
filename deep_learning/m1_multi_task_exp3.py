from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model_mt, load_dataset_from_directory
from tensorflow.keras.layers import Dense, Dropout
from numpy import exp
import os
import tensorflow as tf
import datetime
from keras.callbacks import LearningRateScheduler
from train import (
    custom_age_loss,
    custom_age_metric,
    custom_gender_loss,
    custom_gender_metric,
)

if __name__ == "__main__":
    train_ds, val_ds = load_dataset_from_directory(
        "data/training", "data/labels/train", batch_size=16, multi_task=True
    )

base_archs = {
    "resnet50": keras_app.ResNet50,
    "resnet101": keras_app.ResNet101,
    "efficientnetb0": keras_app.EfficientNetB0,
    "efficientnetb4": keras_app.EfficientNetB4,
    "mobilenetv3small": keras_app.MobileNetV3Small,
    "mobilenetv3large": keras_app.MobileNetV3Large,
}

for freeze in [True, False]:
    for archname, arch in base_archs.items():
        top_layer_conf = [
            [
                Dense(256, activation="relu", name="dense1.1"),
                tf.keras.layers.BatchNormalization(),
                Dense(256, activation="relu", name="dense1.2"),
                tf.keras.layers.BatchNormalization(),
                Dense(1, activation="sigmoid", name="face_detection"),
            ],
            [
                Dense(256, activation="relu", name="dense2.1"),
                tf.keras.layers.BatchNormalization(),
                Dense(256, activation="relu", name="dense2.2"),
                tf.keras.layers.BatchNormalization(),
                Dense(1, activation="linear", name="age_prediction"),
            ],
            [
                Dense(256, activation="relu", name="dense3.1"),
                Dense(256, activation="relu", name="dense3.2"),
                Dense(1, activation="sigmoid", name="gender_classification"),
            ],
        ]

        model_name = (f"mt_{archname}_freeze={freeze}")
        print("Training", model_name)
        model = build_model_mt(arch, (224, 224, 3), top_layer_conf, freeze)

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=[
                tf.keras.losses.BinaryCrossentropy(),
                custom_age_loss,
                custom_gender_loss,
            ],
            metrics={
                "face_detection": "accuracy",
                "age_prediction": custom_age_metric,
                "gender_classification": custom_gender_metric,
            },
        )

        train_log_dir = f"logs/mt/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_train_callback = tf.keras.callbacks.TensorBoard(
            log_dir=train_log_dir, histogram_freq=1
        )

        model.fit(
            train_ds,
            epochs=20,
            validation_data=val_ds,
            callbacks=[tensorboard_train_callback],
        )

        os.makedirs("models/mt", exist_ok=True)
        model.save(f"models/mt/{model_name}.keras")
