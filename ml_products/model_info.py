"""
model_info.py
=============
Evaluate the frozen CNN on the held-out test set and print the confusion
matrix, precision, recall, and F1 scores.

Run from the directory containing this script (test_data.csv,
test_labels.csv, and model_run_All.keras are expected next to it).

Note: the test split is balanced 777 non-dust / 777 dust (1,554 samples).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

print("1. Loading test data ...")
# Test set: 1,554 samples, 6,144 channels each (stored with per-row
# max|amplitude| = 1 normalization baked in at export time)
X_test = pd.read_csv("test_data.csv", header=None)

# Reshape to (samples, sequence_length, channels) to match the CNN input
X_test_reshaped = X_test.values.reshape(-1, 6144, 1)

# Load the true labels from the released label file
y_test = pd.read_csv("test_labels.csv", header=None).values.flatten()

print("2. Loading the trained CNN model ...")
model = tf.keras.models.load_model("model_run_All.keras")

print("3. Predicting on the test set ...")
y_pred_prob = model.predict(X_test_reshaped)
# Convert probabilities to class labels (0 or 1)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n========== Final evaluation numbers ==========\n")

# 1. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("[Confusion Matrix]:")
print(cm)
print("-" * 40)

# 2. Precision, recall, F1
print("[Classification Report]:")
print(classification_report(y_test, y_pred, target_names=["non-dust (0)", "dust (1)"], digits=4))
