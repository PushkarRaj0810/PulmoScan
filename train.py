
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_pipeline import load_datasets

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 7

train_ds, val_ds, test_ds, class_names = load_datasets()

print("Classes:", class_names)

if class_names != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(f"Unexpected class mapping: {class_names}")

# --------------------------------------------------
# 2. Calculate class weights
# --------------------------------------------------
class_counts = Counter()

for _, labels in train_ds:
    class_counts.update(
        labels.numpy().ravel().astype(int).tolist()
    )

total_images = sum(class_counts.values())
num_classes = len(class_counts)

if num_classes != 2 or any(
    class_counts.get(class_id, 0) == 0
    for class_id in range(2)
):
    raise ValueError(f"Invalid training class counts: {class_counts}")

class_weights = {
    class_id: total_images / (
        num_classes * class_counts[class_id]
    )
    for class_id in range(num_classes)
}

print("\nTraining class counts:")
for class_id, count in sorted(class_counts.items()):
    print(f"{class_names[class_id]}: {count}")

print("\nClass weights:")
for class_id, weight in sorted(class_weights.items()):
    print(f"{class_names[class_id]}: {weight:.4f}")

# --------------------------------------------------
# 3. Load pretrained ResNet50
# --------------------------------------------------
base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3),
)

# Initially freeze the pretrained layers.
base_model.trainable = False

# ResNet50 requires its own preprocessing.
inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

x = layers.Rescaling(1.0)(inputs)  # Identity; documents input stage
x = keras.applications.resnet50.preprocess_input(x)

# Keep pretrained BatchNormalization layers in inference mode.
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name="resnet50_pneumonia")

# --------------------------------------------------
# 4. Compile classification head
# --------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

# --------------------------------------------------
# 5. Callbacks
# --------------------------------------------------
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/best_resnet50.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# --------------------------------------------------
# 6. Phase 1: Train classification head
# --------------------------------------------------
print("\nPhase 1: Training classification head")

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# --------------------------------------------------
# 7. Phase 2: Fine-tune the final ResNet layers
# --------------------------------------------------
print("\nPhase 2: Fine-tuning ResNet50")

base_model.trainable = True

# Freeze most layers; fine-tune only the final 30.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Recompile after changing trainable layers.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# --------------------------------------------------
# 8. Save final model
# --------------------------------------------------
model.save("models/final_resnet50.keras")

print("\nTraining complete.")
print("Best checkpoint: models/best_resnet50.keras")
print("Final model: models/final_resnet50.keras")