import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from dataloading import create_dataloader

def build_siamese_model(input_shape=(224, 224, 3)):
    # ResNet50 als Basis-Modell
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    # Hinzufügen von Anpassungsschichten
    flatten = layers.Flatten()(base_model.output)
    dense = layers.Dense(128, activation="relu")(flatten)
    base_model = models.Model(inputs=base_model.input, outputs=dense)

    input_1 = layers.Input(shape=input_shape)
    input_2 = layers.Input(shape=input_shape)

    # Gemeinsames Modell für beide Eingaben
    encoded_1 = base_model(input_1)
    encoded_2 = base_model(input_2)

    # Abstand zwischen den Embeddings berechnen
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_1, encoded_2])

    # Klassifikation auf Basis der Distanz
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = models.Model(inputs=[input_1, input_2], outputs=output)
    return model

# Visualisierung des Trainingsverlaufs
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()

def contrastive_loss(margin=1.0):
    """
    Custom loss function for the Siamese network.
    Positive pairs: Mean squared distance.
    Negative pairs: Margin-based loss.
    """
    def loss(y_true, y_pred):
        print("y_true", y_true)
        print("y_pred", y_pred)
        squared_distance = tf.reduce_sum(tf.square(y_pred), axis=1)
        print("squared_distance", squared_distance)
        positive_loss = y_true * squared_distance
        print("positive_loss", positive_loss)
        # TODO: square margin
        negative_loss = (1 - y_true) * tf.maximum(0.0, margin - squared_distance)
        print("negative_loss", negative_loss)
        contr_loss = tf.reduce_mean(positive_loss + negative_loss)
        print("loss", contr_loss)
        exit()
        return contr_loss
    return loss

def train(data_dir, batch_size, epochs, output, val_split=0.2, margin=1.0):
    assert 0 < val_split < 1, "Validation split must be between 0 and 1"

    dataloaders = create_dataloader(data_dir, batch_size=batch_size, val_split=val_split)
    model = build_siamese_model()
    model.compile(
        optimizer='adam', 
        loss=contrastive_loss(margin=margin), 
        metrics=['accuracy'],
        run_eagerly=True
    )
    history = model.fit(dataloaders[0], validation_data=dataloaders[1], epochs=epochs)
    plot_history(history)
