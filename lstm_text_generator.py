import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import np_utils

# from tensorflow.python.keras import backend as k

def char_to_int(text_list):
    return dict((c, i) for i, c in enumerate(text_list))

alice_file = "alice.txt"
dracula_file = "dracula.txt"
jekyll_hyde_file = "jekyll_and_hyde.txt"

raw_text_alice = open(alice_file).read()
raw_text_dracula = open(dracula_file).read()
raw_text_jekyll = open(jekyll_hyde_file).read()

raw_text_alice = raw_text_alice.lower()
raw_text_dracula = raw_text_dracula.lower()
raw_text_jekyll = raw_text_jekyll.lower()

# Can be refactored into a function
alice = sorted(list(set(raw_text_alice)))
alice_to_int = char_to_int(alice)
dracula = sorted(list(set(raw_text_dracula)))
dracula_to_int = char_to_int(dracula)
jekyll_hyde = sorted(list(raw_text_jekyll))
jekyll_hyde_to_int = char_to_int(jekyll_hyde)

# Can be refactored into a function
alice_chars = len(raw_text_alice)
alice_vocab = len(alice)
dracula_chars = len(raw_text_dracula)
dracula_vocab = len(dracula)
jekyll_hyde_chars = len(raw_text_jekyll)
jekyll_hyde_vocab = len(jekyll_hyde)

# Can be refactored into a function
sequence_length = 10
aliceX = []
aliceY = []
for i in range(0, alice_chars - sequence_length, 1):
    sequence_in = raw_text_alice[i: i+sequence_length]
    sequence_out = raw_text_alice[i + sequence_length]
    aliceX.append([alice_to_int[char] for char in sequence_in])
    aliceY.append(alice_to_int[sequence_out])

number_of_patterns = len(aliceX)

wonderlandX = np.reshape(aliceX, (number_of_patterns, sequence_length, 1))
wonderlandX = wonderlandX / float(alice_vocab)
wonderlandy = np_utils.to_categorical(aliceY)

model = Sequential()
model.add(LSTM(256, input_shape = (wonderlandX.shape[1], wonderlandX.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(wonderlandy.shape[1], activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

path_to_file = "weights-improvement-{epoch:02d}-{loss:0.4f}.hdf5"
checkpoint = ModelCheckpoint(path_to_file, monitor='loss', verbose=1, save_best_only = True, mode='min')
callbacks_list = [checkpoint]

alice_model = model.fit(wonderlandX, wonderlandy, epochs=50, batch_size=64, callbacks=callbacks_list)
alice_model_file = "weights-improvement-50-1.4290.hdf5"
alice_model_load = model.load_weights(alice_model_file)

alice_compile = model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
alice_int_to_char = dict((i, c) for i, c in enumerate(alice_chars))

start = np.random.randint(0, len(aliceX)-1)
# pattern = aliceX