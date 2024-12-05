import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import binary_crossentropy
import numpy as np

@register_keras_serializable()
def custom_age_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    y_true_adjusted = tf.where(mask, y_true, 0.0)
    y_pred_adjusted = tf.where(mask, y_pred, 0.0)
    mse_loss = tf.square(y_pred_adjusted - y_true_adjusted)
    loss = tf.reduce_mean(mse_loss)

    if tf.executing_eagerly():
        print("\n----------AGE LOSS----------")
        print("y_true", y_true)
        print("y_pred", y_pred)
        print("mask:", mask)
        print("y_true_adjusted:", y_true_adjusted)
        print("y_pred_adjusted:", y_pred_adjusted)
        print("mse loss values:", mse_loss)
        print("mse batch loss:", loss)
        print("---------------------------\n")
    return loss


@register_keras_serializable()
def custom_gender_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    y_true_adjusted = tf.where(mask, y_true, 0.0)
    y_pred_adjusted = tf.where(mask, y_pred, 0.0)
    bce_loss = binary_crossentropy(y_true_adjusted, y_pred_adjusted)
    loss = tf.reduce_mean(bce_loss)

    if tf.executing_eagerly():
        print("\n----------GENDER LOSS----------")
        print("y_true", y_true)
        print("y_pred", y_pred)
        print("mask:", mask)
        print("y_true_adjusted:", y_true_adjusted)
        print("y_pred_adjusted:", y_pred_adjusted)
        print("binary cross entropy:", bce_loss)
        print("bce batch loss:", loss)
        print("---------------------------\n")
    return loss


@register_keras_serializable()
def custom_age_metric(y_true, y_pred):
    y_pred = tf.squeeze(y_pred)
    mask = tf.not_equal(y_true, -1)
    y_true_adjusted = tf.where(mask, tf.cast(y_true, tf.float32), 0.0)
    y_pred_adjusted = tf.where(mask, y_pred, 0.0)
    mae = tf.abs(y_pred_adjusted - y_true_adjusted)
    loss = tf.reduce_mean(mae)

    if tf.executing_eagerly():
        print("\n----------AGE METRIC----------")
        print("y_true", y_true)
        print("y_pred", y_pred)
        print("mask:", mask)
        print("y_true_adjusted:", y_true_adjusted)
        print("y_pred_adjusted:", y_pred_adjusted)
        print("mae loss values:", mae)
        print("age batch loss:", loss)
        print("---------------------------\n")

    return loss


@register_keras_serializable()
def custom_gender_metric(y_true, y_pred):
    y_pred = tf.squeeze(y_pred)
    mask = tf.equal(y_true, -1)
    correct_preds = tf.equal(
        tf.cast(y_true, tf.int32), tf.cast(tf.round(y_pred), tf.int32)
    )
    correct_preds = tf.logical_or(correct_preds, mask)
    acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    if tf.executing_eagerly():
        print("\n----------GENDER METRIC----------")
        print("y_true", y_true)
        print("y_pred", y_pred)
        print("mask:", mask)
        print("correction preds:", correct_preds)
        print("gender batch acc:", acc)
        print("---------------------------\n")

    return acc

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array