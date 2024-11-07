from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import build_model, load_dataset_from_directory, multi_task_loss
from tensorflow.keras.layers import Dense, Dropout
import os
import tf
import datetime

os.makedirs("models/mt", exist_ok=True)

MODEL_ARCHS = {
    "resnet50": keras_app.ResNet50,
    # "resnet101": keras_app.ResNet101,
    # "resnet152": keras_app.ResNet152,
}


train_ds, val_ds = load_dataset_from_directory(
    "data/training", "data/labels/training", batch_size=64, multi_task=True
)

for arch_name, arch in MODEL_ARCHS.items():
    for freeze in [True, False]:
        top_layers = [Dense(256, activation="relu"), Dropout(0.3)]
        output_layers = [
            Dense(1, activation="sigmoid", name="face_detection"),
            Dense(1, activation="linear", name="age_prediction"),
            Dense(1, activation="sigmoid", name="gender_classification"),
        ]
        model_name = f"mt_{arch_name}_adam_lr=0.0001_lc=0_freeze={freeze}"
        model = build_model(arch, (224, 224, 3), top_layers, output_layers, freeze)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=multi_task_loss,
            metrics={
                "face_detection": "accuracy",
                "age_prediction": "mae",
                "gender_classification": "accuracy",
            },
        )

        train_log_dir = f"logs/mt/train/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_train_callback = tf.keras.callbacks.TensorBoard(
            log_dir=train_log_dir, histogram_freq=1
        )

        val_log_dir = f"logs/mt/val/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_val_callback = tf.keras.callbacks.TensorBoard(
            log_dir=val_log_dir, histogram_freq=1
        )

        model.fit(
            train_ds,
            epochs=10,
            validation_data=val_ds,
            callbacks=[tensorboard_train_callback, tensorboard_val_callback],
        )
        model.save(f"models/mt/{model_name}.keras")
