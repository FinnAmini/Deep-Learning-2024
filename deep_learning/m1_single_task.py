from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model, load_data
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import datetime
import os

os.makedirs("models/st", exist_ok=True)

MODEL_ARCHS = {
    "resnet50": keras_app.ResNet50,  # 25,6M Params
    "resnet101": keras_app.ResNet101,  # 44,6M Params
}

output_layers = []
# train_ds, val_ds = load_dataset_from_directory("data_test", batch_size=64)
train_ds, val_ds = load_data("data/training", batch_size=64)

for arch_name, arch in MODEL_ARCHS.items():
    for freeze in [True, False]:
        top_layer_confs = [
            (
                [
                    Dense(256, activation="relu", name="dense1.1"),
                    Dropout(0.3),
                ],
                [Dense(1, activation="sigmoid", name="output1")],
            ),
            (
                [
                    Dense(256, activation="relu", name="dense2.1"),
                    Dropout(0.3),
                    Dense(256, activation="relu", name="dense2.2"),
                    Dropout(0.3),
                ],
                [Dense(1, activation="sigmoid", name="output2")],
            ),
            (
                [
                    Dense(128, activation="relu", name="dense3.1"),
                    Dropout(0.3),
                ],
                [Dense(1, activation="sigmoid", name="output3")],
            ),
            (
                [
                    Dense(128, activation="relu", name="dense4.1"),
                    Dropout(0.3),
                    Dense(128, activation="relu", name="dense4.2"),
                    Dropout(0.3),
                ],
                [Dense(1, activation="sigmoid", name="output4")],
            ),
        ]

        for conf_id, layer_conf in enumerate(top_layer_confs):
            top_layers, output_layers = layer_conf
            print("Training ", freeze, arch_name, conf_id)
            model_name = f"st_{arch_name}_adam_lr=0.0001_lc={conf_id}_freeze={freeze}"
            model = build_model(arch, (224, 224, 3), top_layers, output_layers, freeze)
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            train_log_dir = f"logs/st/train/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_train_callback = tf.keras.callbacks.TensorBoard(
                log_dir=train_log_dir, histogram_freq=1
            )

            val_log_dir = f"logs/st/val/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_val_callback = tf.keras.callbacks.TensorBoard(
                log_dir=val_log_dir, histogram_freq=1
            )

            model.fit(
                train_ds,
                epochs=50,
                validation_data=val_ds,
                callbacks=[tensorboard_train_callback, tensorboard_val_callback],
            )

            model.save(f"models/st/{model_name}.keras")
