import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_label), (test_data, test_label) = data.load_data(num_words=10000)

vocabulary_index = data.get_word_index()

vocabulary_index = {k:(v + 3) for k, v in vocabulary_index.items()}
vocabulary_index["<PAD>"] = 0
vocabulary_index["<START>"] = 1
vocabulary_index["<UNK>"] = 2
vocabulary_index["<UNUSED>"] = 3

reverse_vocabulary_index = dict([(value, key) for (key, value) in vocabulary_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=vocabulary_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=vocabulary_index["<PAD>"], padding="post", maxlen=256)

def decode_review(text):
    return " ".join([reverse_vocabulary_index.get(i, "?") for i in text])

# for i in range(5):
#     print(decode_review(test_data[i]))

model = keras.Sequential()
model.add(keras.layers.Embedding(100000, 16)) 
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_validation = train_data[:10000]
x_train = train_data[10000:]
y_validation = train_label[:10000]
y_train = train_label[10000:]

fit_model = model.fit(x_train, y_train, epochs=50, batch_size=512, validation_data=(x_validation, y_validation), verbose=1)

result = model.evaluate(test_data, test_label)

model.save("text_model.h5")



# print(result)

# prediction = model.predict([test_data])
