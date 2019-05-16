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


text_model = keras.models.load_model("text_model.h5")

def review_encoder(s):
    encoded = [1]
    for word in s:
        if word.lower() in vocabulary_index:
            encoded.append(vocabulary_index[word])
        else:
            encoded.append(2)
    return encoded

# coming from a text file
with open(" ", encoding="utf-8") as f:
    for line in f.readlines():
        r_line = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("\"", "").replace(":", "").replace(";", "").strip().split(" ")
        encode = review_encoder(r_line)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=vocabulary_index["<PAD>"], padding="post", maxlen=256)
        prediction = keras.Sequential.predict(encode)
        print(prediction[0])
