from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as keras_app
from train import load_dataset_from_directory, build_model
from tensorflow.keras.layers import Dense, Dropout

MODEL_ARCHS = {
    "resnet50": keras_app.ResNet50,
    "resnet101": keras_app.ResNet101,
    "resnet152": keras_app.ResNet152,
}

top_layers = [Dense(256, activation="relu"), Dropout(0.3)]
output_layers = [
    Dense(1, activation="sigmoid", name="face_detection"),
    Dense(1, activation="linear", name="age_prediction"),
    Dense(1, activation="sigmoid", name="gender_classification"),
]
train_ds, val_ds = load_dataset_from_directory(
    "data_test", batch_size=64, multi_task=True
)
# images = ...  # shape (num_samples, 224, 224, 3)
# face_labels = ...  # shape (num_samples, 1) with binary labels 0 or 1
# age_labels = ...  # shape (num_samples, 1) with ages in years
# gender_labels = ...  # shape (num_samples, 1) with binary labels 0 (female) or 1 (male)

for freeze in [True, False]:
    for arch_name, arch in MODEL_ARCHS.items():
        model_name = f"st_{arch_name}_adam_lr=0.0001_lc1_freeze={freeze}"
        model = build_model(arch, (224, 224, 3), top_layers, output_layers, freeze)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={
                "face_detection": "binary_crossentropy",
                "age_prediction": "mean_squared_error",
                "gender_classification": "binary_crossentropy",
            },
            metrics={
                "face_detection": "accuracy",
                "age_prediction": "mae",
                "gender_classification": "accuracy",
            },
        )

model.fit(train_ds, epochs=50, validation_data=val_ds)
model.save(f"models/{model_name}.keras")
