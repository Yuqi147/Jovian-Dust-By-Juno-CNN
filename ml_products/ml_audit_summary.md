# CNN Classifier Audit Summary (Task 8)

## 1. Model configuration (from CNN_All.py)
| Item | Value |
|---|---|
| architecture | Conv1D(f=128,k=8) -> Conv1D(f=256,k=5) -> Conv1D(f=128,k=3) -> GlobalAveragePooling1D -> Dense(2,softmax) |
| input_shape | [6144,1] |
| optimizer | adam |
| loss | sparse_categorical_crossentropy |
| batch_size | 128 |
| epochs_set | 600 |
| validation_split | 0.2 |
| random_seed_set | False |

## 2. Data split
| Item | Value |
|---|---|
| n_train | 6216 |
| n_val_expected | 1243 |
| n_train_fit_expected | 4973 |
| n_test | 1554 |
| train_class0 | 3108 |
| train_class1 | 3108 |
| test_class0 | 777 |
| test_class1 | 777 |

## 3. Reported metrics (terminal.txt)
| Item | Value |
|---|---|
| test_accuracy | 0.9710424542427063 |
| test_loss | 0.10810913890600204 |
| test_correct | 1509 |
| test_wrong | 45 |
| last_epoch | 600 |
| val_acc_last | 0.9735 |
| val_acc_max | 0.9855 |

## 4. Reproducibility
- Random seed: **not set** -> the train/validation split is not reproducible; the offline train/test split also has no recorded seed.
- Figure 1 states 'Validation: 10 random splits'; the training script uses a single `validation_split=0.2` -> **DISCREPANCY**.
- Test set is balanced 777/777; training set 3108/3108.
- Per-class test metrics require the frozen model (TensorFlow not installed here) -> AUTHOR VERIFICATION REQUIRED to reproduce predictions and the confusion matrix.
