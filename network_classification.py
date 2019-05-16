import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_image, train_labels), (test_image, test_labels) = data.load_data()

class_names = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

train_image = train_image/255.0
test_image = test_image/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_image, train_labels, epochs=8)

# test_loss, test_acc = model.evaluate(test_image, test_labels)

# print("tested accuracy: ", test_acc)

predicted = model.predict(test_image)

# To ensure that it doesn't run through everything.
# Could put into a function later.
# for i in range(5):
#     plt.grid(False)
#     plt.imshow(test_image[i], cmap=plt.cm.binary)
#     plt.xlabel("Actual: ", class_names[test_labels[i]])
#     plt.title("Prediction: ", class_names[np.argmax(predicted[i])])

model.save("image_model.h5")
