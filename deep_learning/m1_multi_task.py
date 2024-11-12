from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model_mt, load_dataset_from_directory
from tensorflow.keras.layers import Dense, Dropout
import os
import tensorflow as tf
import datetime
from keras.saving import register_keras_serializable


@register_keras_serializable()
def custom_age_loss(y_true, y_pred):
    y_pred = tf.squeeze(y_pred)
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

    # print("----------age_acc----------")
    # print("Mask values (True for valid entries):", mask)
    # print("y_true_adjusted:", y_true_adjusted)
    # print("y_pred_adjusted:", y_pred_adjusted.numpy())
    # print("MAE values:", mae)
    # print("---------------------------")

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
    os.environ["TF_DUMP_GRAPH_PREFIX"] = (
        "/home/faminikaveh/projects/Deep-Learning-2024/tf_dump"
    )
    train_ds, val_ds = load_dataset_from_directory(
        "data/testing", "data/labels/test", batch_size=64, multi_task=True
    )

    top_layer_confs = [
        [
            Dense(256, activation="relu", name="dense1"),
            Dropout(0.3),
            Dense(1, activation="sigmoid", name="face_detection"),
        ],
        [
            Dense(256, activation="relu"n name="dense2"),
            Dropout(0.3),
            Dense(1, activation="linear", name="age_prediction"),
        ],
        [
            Dense(256, activation="relu", name="dense3"),
            Dropout(0.3),
            Dense(1, activation="sigmoid", name="gender_classification"),
        ],
    ]

    model_name = "mt_resnet50_adam_lr=0.0001_lc=0_freeze=True"
    model = build_model_mt(keras_app.ResNet50, (224, 224, 3), top_layer_confs, True)

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

    train_log_dir = (
        f"logs/mt/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    tensorboard_train_callback = tf.keras.callbacks.TensorBoard(
        log_dir=train_log_dir, histogram_freq=1
    )

    model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[tensorboard_train_callback],
    )

    os.makedirs("models/mt", exist_ok=True)
    model.save(f"models/mt/{model_name}.keras")
