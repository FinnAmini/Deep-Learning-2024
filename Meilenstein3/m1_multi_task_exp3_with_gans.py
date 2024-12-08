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
    # Load training and validation datasets
    train_ds, val_ds = load_dataset_from_directory(
        "../Meilenstein2/data/training_gans", "../Meilenstein2/data/labels_gans/train", batch_size=16, multi_task=True
    )

# Define base architectures to be used
base_archs = {
    "resnet50": keras_app.ResNet50
}

# Iterate over freezing options and architectures
for freeze in [False]:
    for archname, arch in base_archs.items():
        # Define top layer configurations for multi-task learning
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

        # Create model name based on architecture and freezing option
        model_name = (f"mt_{archname}_freeze={freeze}")
        print("Training", model_name)
        
        # Build the multi-task model
        model = build_model_mt(arch, (224, 224, 3), top_layer_conf, freeze)

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

        # Set up TensorBoard logging
        train_log_dir = f"logs/mt_with_gans/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        os.makedirs("models/mt_with_gans", exist_ok=True)
        model.save(f"models/mt_with_gans/{model_name}.keras")
