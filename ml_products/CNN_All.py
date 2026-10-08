import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
# import torch as th

# Import training data (6216 observatios, each with dimension 6144)
X_train = pd.read_csv("train_data.csv", header=None)
# Import testing data (1554 observatios, each with dimension 6144)
X_test = pd.read_csv("test_data.csv", header=None)

# Import 16000 training labels and 4000 testing labels
Y_train = pd.read_csv("train_labels.csv", header=None)
Y_test = pd.read_csv("test_labels.csv", header=None)
print("csv read successfully")

# Define the CNN arcitecture
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8,strides=1, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5,strides=1, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3,strides=1, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(2, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

# (modified)
model = make_model(input_shape=[6144,1])

# Plot a diagram of the model arcitecture
keras.utils.plot_model(model, show_shapes=True)

# Number of times to run through the training data\
# (modified)
# epochs = 225 Guangzhou 600
epochs = 600

# Number of samples to run through before updating the network parameters
batch_size = 128

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "run_GitHub.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=100, min_lr=0.000001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=300, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

# model = keras.models.load_model("run_GitHub.h5")
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig("sparse_categorical_accuracy_All_trans.png",
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            edgecolor ='w',
            orientation ='landscape')
plt.savefig("sparse_categorical_accuracy_All.png",
            bbox_inches ="tight",
            pad_inches = 1,
            edgecolor ='w',
            orientation ='landscape')
plt.show()

Y_pred = model.predict(X_test)
Y_pred = [0 if y[0]>=0.5 else 1 for y in Y_pred]

total = 0
correct = 0
wrong = 0
for i in range(len(Y_pred)):
    total=total+1
    if(Y_test.at[i,0] == Y_pred[i]):
        correct=correct+1
    else:
        wrong=wrong+1

print("Total " + str(total))
print("Correct " + str(correct))
print("Wrong " + str(wrong))

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)

model.summary()
# model.save("model_run_All") ##old version
model.save("model_run_All.keras")
model.export("model_run_All")
