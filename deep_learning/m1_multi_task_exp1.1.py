from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model_mt, load_dataset_from_directory
from tensorflow.keras.layers import Dense, Dropout
import os
import tensorflow as tf
import datetime
from train import (
    custom_age_loss,
    custom_age_metric,
    custom_gender_loss,
    custom_gender_metric,
)

if __name__ == "__main__":
    # Load training and validation datasets
    train_ds, val_ds = load_dataset_from_directory(
        "data/training", "data/labels/train", batch_size=64, multi_task=True
    )

    # Iterate over different dropout rates
    for dropout in [0.1, 0.3, 0.5]:
        # Iterate over different top layer configurations
        for i, top_layer_conf in enumerate(
            [
                # Configuration 1
                [
                    [
                        Dense(256, activation="relu", name="dense1.1"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="face_detection"),
                    ],
                    [
                        Dense(256, activation="relu", name="dense2.1"),
                        Dropout(dropout),
                        Dense(1, activation="linear", name="age_prediction"),
                    ],
                    [
                        Dense(256, activation="relu", name="dense3.1"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="gender_classification"),
                    ],
                ],
                # Configuration 2
                [
                    [
                        Dense(256, activation="relu", name="dense1.1"),
                        Dropout(dropout),
                        Dense(256, activation="relu", name="dense1.2"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="face_detection"),
                    ],
                    [
                        Dense(256, activation="relu", name="dense2.1"),
                        Dropout(dropout),
                        Dense(256, activation="relu", name="dense2.2"),
                        Dropout(dropout),
                        Dense(1, activation="linear", name="age_prediction"),
                    ],
                    [
                        Dense(256, activation="relu", name="dense3.1"),
                        Dropout(dropout),
                        Dense(256, activation="relu", name="dense3.2"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="gender_classification"),
                    ],
                ],
                # Configuration 3
                [
                    [
                        Dense(128, activation="relu", name="dense1.1"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="face_detection"),
                    ],
                    [
                        Dense(128, activation="relu", name="dense2.1"),
                        Dropout(dropout),
                        Dense(1, activation="linear", name="age_prediction"),
                    ],
                    [
                        Dense(128, activation="relu", name="dense3.1"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="gender_classification"),
                    ],
                ],
                # Configuration 4
                [
                    [
                        Dense(128, activation="relu", name="dense1.1"),
                        Dropout(dropout),
                        Dense(128, activation="relu", name="dense1.2"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="face_detection"),
                    ],
                    [
                        Dense(128, activation="relu", name="dense2.1"),
                        Dropout(dropout),
                        Dense(128, activation="relu", name="dense2.2"),
                        Dropout(dropout),
                        Dense(1, activation="linear", name="age_prediction"),
                    ],
                    [
                        Dense(128, activation="relu", name="dense3.1"),
                        Dropout(dropout),
                        Dense(128, activation="relu", name="dense3.2"),
                        Dropout(dropout),
                        Dense(1, activation="sigmoid", name="gender_classification"),
                    ],
                ],
            ]
        ):
            # Define model name based on configuration and dropout rate
            model_name = (
                f"mt_resnet50_adam_lr=0.0001_lc={i}_freeze=True_dropout={dropout}"
            )
            # Build the multi-task model
            model = build_model_mt(
                keras_app.ResNet50, (224, 224, 3), top_layer_conf, True
            )

            # Compile the model with custom losses and metrics
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

            # Define TensorBoard log directory
            train_log_dir = f"logs/mt/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_train_callback = tf.keras.callbacks.TensorBoard(
                log_dir=train_log_dir, histogram_freq=1
            )

            # Train the model
            model.fit(
                train_ds,
                epochs=20,
                validation_data=val_ds,
                callbacks=[tensorboard_train_callback],
            )

            # Save the trained model
            os.makedirs("models/mt", exist_ok=True)
            model.save(f"models/mt/{model_name}.keras")
