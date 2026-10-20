
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

_, _, test_ds, class_names = load_datasets()

model = tf.keras.models.load_model("models/best_resnet50.keras")

y_true = np.concatenate([
    labels.numpy().ravel() for _, labels in test_ds
]).astype(int)

y_prob = model.predict(test_ds, verbose=1).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0,
))

print("ROC-AUC:", roc_auc_score(y_true, y_prob))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
display.plot(cmap="Blues", values_format="d")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()
