from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model_mt, load_dataset_from_directory
from tensorflow.keras.layers import Dense, Dropout
import os
import tensorflow as tf
import datetime
from keras.saving import register_keras_serializable
import uuid


@register_keras_serializable()
def custom_age_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    y_true_adjusted = tf.where(mask, y_true, 0.0)
    y_pred_adjusted = tf.where(mask, y_pred, 0.0)
    mse_loss = tf.square(y_pred_adjusted - y_true_adjusted)
    return tf.reduce_mean(mse_loss)


@register_keras_serializable()
def custom_gender_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    y_true_adjusted = tf.where(mask, y_true, 0.0)
    y_pred_adjusted = tf.where(mask, y_pred, 0.0)
    binary_crossentropy_loss = tf.keras.losses.binary_crossentropy(
        y_true_adjusted, y_pred_adjusted
    )
    return tf.reduce_mean(binary_crossentropy_loss)


@register_keras_serializable()
def custom_age_metric(y_true, y_pred):
    y_pred = tf.squeeze(y_pred)
    mask = tf.not_equal(y_true, -1)
    y_true_adjusted = tf.where(mask, tf.cast(y_true, tf.float32), 0.0)
    y_pred_adjusted = tf.where(mask, y_pred, 0.0)
    mae = tf.abs(y_pred_adjusted - y_true_adjusted)


    return tf.reduce_mean(mae)


@register_keras_serializable()
def custom_gender_metric(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    correct_preds = tf.equal(
        tf.cast(y_true, tf.int32), tf.cast(tf.round(y_pred), tf.int32)
    )
    correct_preds = tf.logical_or(correct_preds, tf.logical_not(mask))
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32))


if __name__ == "__main__":


    # learning_rates = [0.0001, 0.0005, 0.001]
    batch_sizes = [16, 32, 64]
    
    # for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        train_ds, val_ds = load_dataset_from_directory(
            "data/training", "data/labels/train", batch_size=batch_size, multi_task=True
        )
        for i, top_layer_conf in enumerate([
        
            # Config 1 
            [
                [
                    Dense(256, activation="relu", name="dense1.1"),
                    Dense(128, activation="relu", name="dense1.2"),
                    Dense(1, activation="sigmoid", name="face_detection"),
                ],
                [
                    Dense(256, activation="relu", name="dense2.1"),
                    Dense(128, activation="relu", name="dense2.2"),
                    Dense(1, activation="linear", name="age_prediction"),
                ],
                [
                    Dense(256, activation="relu", name="dense3.1"),
                    Dense(128, activation="relu", name="dense3.2"),
                    Dense(1, activation="sigmoid", name="gender_classification"),
                ],
            ],
            # Config 2
            [
                [
                    Dense(256, activation="relu", name="dense1.1"),
                    Dropout(0.3),
                    Dense(128, activation="relu", name="dense1.2"),
                    Dropout(0.2),
                    Dense(1, activation="sigmoid", name="face_detection"),
                ],
                [
                    Dense(256, activation="relu", name="dense2.1"),
                    Dropout(0.3),
                    Dense(128, activation="relu", name="dense2.2"),
                    Dropout(0.2),
                    Dense(1, activation="linear", name="age_prediction"),
                ],
                [
                    Dense(256, activation="relu", name="dense3.1"),
                    Dropout(0.3),
                    Dense(128, activation="relu", name="dense3.2"),
                    Dropout(0.2),
                    Dense(1, activation="sigmoid", name="gender_classification"),
                ],
            ],
            # Config 3
            [
                [
                    Dense(256, activation="relu", name="dense1.1"),
                    tf.keras.layers.BatchNormalization(),
                    Dense(128, activation="relu", name="dense1.2"),
                    tf.keras.layers.BatchNormalization(),
                    Dense(1, activation="sigmoid", name="face_detection"),
                ],
                [
                    Dense(256, activation="relu", name="dense2.1"),
                    tf.keras.layers.BatchNormalization(),
                    Dense(128, activation="relu", name="dense2.2"),
                    tf.keras.layers.BatchNormalization(),
                    Dense(1, activation="linear", name="age_prediction"),
                ],
                [
                    Dense(256, activation="relu", name="dense3.1"),
                    tf.keras.layers.BatchNormalization(),
                    Dense(128, activation="relu", name="dense3.2"),
                    tf.keras.layers.BatchNormalization(),
                    Dense(1, activation="sigmoid", name="gender_classification"),
                ],
            ],
            # Config 4
            [
                [
                    Dense(256, activation="relu", name="dense1.1"),
                    Dropout(0.3),
                    tf.keras.layers.BatchNormalization(),
                    Dense(128, activation="relu", name="dense1.2"),
                    Dropout(0.2),
                    tf.keras.layers.BatchNormalization(),
                    Dense(1, activation="sigmoid", name="face_detection"),
                ],
                [
                    Dense(256, activation="relu", name="dense2.1"),
                    Dropout(0.3),
                    tf.keras.layers.BatchNormalization(),
                    Dense(128, activation="relu", name="dense2.2"),
                    Dropout(0.2),
                    tf.keras.layers.BatchNormalization(),
                    Dense(1, activation="linear", name="age_prediction"),
                ],
                [
                    Dense(256, activation="relu", name="dense3.1"),
                    Dropout(0.3),
                    tf.keras.layers.BatchNormalization(),
                    Dense(128, activation="relu", name="dense3.2"),
                    Dropout(0.2),
                    tf.keras.layers.BatchNormalization(),
                    Dense(1, activation="sigmoid", name="gender_classification"),
                ],
            ],
        ]):

            model_name = f"mt_resnet50_adam_lr=0.0001_lc=b{i}_freeze=True_bs={batch_size}"
            model = build_model_mt(keras_app.ResNet50, (224, 224, 3), top_layer_conf, True)
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
                # run_eagerly=True,
            )

            train_log_dir = (
                f"logs/mt/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            tensorboard_train_callback = tf.keras.callbacks.TensorBoard(
                log_dir=train_log_dir, histogram_freq=1
            )

            model.fit(
                train_ds,
                epochs=20,
                validation_data=val_ds,
                callbacks=[tensorboard_train_callback],
            )

            os.makedirs("models/mt_extra", exist_ok=True)
            model.save(f"models/mt_extra/{model_name}.keras")
