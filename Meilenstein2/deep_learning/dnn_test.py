import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data: Resize to 32x32 and normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Resize images to 32x32x3 (3 channels)
x_train = tf.image.resize(x_train, (32, 32))  # Resize images to 32x32
x_test = tf.image.resize(x_test, (32, 32))  # Resize images to 32x32

# Convert grayscale to RGB by duplicating the channels
x_train = tf.repeat(x_train, 3, axis=-1)
x_test = tf.repeat(x_test, 3, axis=-1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model using ResNet50
model = models.Sequential()

# Add ResNet50 model as a base (use pre-trained weights, without the top layer)
model.add(ResNet50(weights="imagenet", include_top=False, input_shape=(32, 32, 3)))

# Add custom top layers for MNIST classification
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
