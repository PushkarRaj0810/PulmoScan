
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from data_pipeline import load_datasets

# Load validation data and best checkpoint
_, val_ds, _, class_names = load_datasets()

model = tf.keras.models.load_model("models/best_cnn.keras")

print("Class mapping:", dict(enumerate(class_names)))

# Collect true labels
y_true = np.concatenate([
    labels.numpy().ravel()
    for _, labels in val_ds
]).astype(int)

# Predict probabilities for PNEUMONIA
y_prob = model.predict(val_ds, verbose=1).ravel()

print("\nValidation ROC-AUC:", round(
    roc_auc_score(y_true, y_prob), 4
))

# Evaluate several thresholds for analysis
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    specificity = tn / (tn + fp) if tn + fp else 0

    print(f"\nThreshold: {threshold:.1f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# Detailed report at the default threshold
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification report at threshold 0.5:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0,
))

# Save confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(cmap="Blues", values_format="d")
plt.tight_layout()
plt.savefig("outputs/validation_confusion_matrix.png")
plt.show()
