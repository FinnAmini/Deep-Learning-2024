from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model, load_data
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import datetime

MODEL_ARCHS = {
    "resnet18": keras_app.ResNet18,  # 11,7M Params
    "resnet50": keras_app.ResNet50,  # 25,6M Params
    "efficientnetb0": keras_app.EfficientNetB0,  # 5,5M Params
    "efficientnetb3": keras_app.EfficientNetB3,  # 12M Params
    "inceptionv3": keras_app.inception_v3,  # 23M Params
}

top_layers = [Dense(256, activation="relu"), Dropout(0.3)]
output_layers = [Dense(1, activation="sigmoid")]
# train_ds, val_ds = load_dataset_from_directory("data_test", batch_size=64)
train_ds, val_ds = load_data("data/training", batch_size=64)

for freeze in [True, False]:
    for arch_name, arch in MODEL_ARCHS.items():
        model_name = f"st_{arch_name}_adam_lr=0.0001_lc1_freeze={freeze}"
        model = build_model(arch, (224, 224, 3), top_layers, output_layers, freeze)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[tensorboard_callback],
)

model.save(f"models/{model_name}.keras")
