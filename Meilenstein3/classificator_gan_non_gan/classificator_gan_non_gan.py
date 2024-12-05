from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model, load_data
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import datetime
import os

# Create directory for saving models if it doesn't exist
os.makedirs("models/st", exist_ok=True)

# Define model architectures to be used
MODEL_ARCHS = {
    "resnet50": keras_app.ResNet50,  # 25.6M Params
}

output_layers = []
# Load training and validation datasets
train_ds, val_ds = load_data("data/training", batch_size=64)

# Iterate over each model architecture
for arch_name, arch in MODEL_ARCHS.items():
    # Iterate over freezing and non-freezing conditions
    for freeze in [False]:
        # Define different configurations for top layers
        top_layer_confs = [
            (
                [
                    Dense(256, activation="relu", name="dense2.1"),
                    Dropout(0.3),
                    Dense(256, activation="relu", name="dense2.2"),
                    Dropout(0.3),
                ],
                [Dense(1, activation="sigmoid", name="output2")],
            )
        ]

        # Iterate over each top layer configuration
        for conf_id, layer_conf in enumerate(top_layer_confs):
            top_layers, output_layers = layer_conf
            print("Training ", freeze, arch_name, conf_id)
            # Define model name based on configuration
            model_name = f"st_{arch_name}_adam_lr=0.0001_lc={conf_id}_freeze={freeze}"
            # Build the model with the specified architecture and layers
            model = build_model(arch, (224, 224, 3), top_layers, output_layers, freeze)
            # Compile the model with Adam optimizer and binary crossentropy loss
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Define log directories for TensorBoard
            train_log_dir = f"logs/st/train/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_train_callback = tf.keras.callbacks.TensorBoard(
                log_dir=train_log_dir, histogram_freq=1
            )

            val_log_dir = f"logs/st/val/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_val_callback = tf.keras.callbacks.TensorBoard(
                log_dir=val_log_dir, histogram_freq=1
            )

            # Train the model
            model.fit(
                train_ds,
                epochs=20,
                validation_data=val_ds,
                callbacks=[tensorboard_train_callback, tensorboard_val_callback],
            )

            # Save the trained model
            model.save(f"models/st/{model_name}.keras")
