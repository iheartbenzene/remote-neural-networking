import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LTSM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

alice_file = "alice.txt"
dracula_file = "dracula.txt"
jekyll_hyde_file = "jekyll_and_hyde.txt"

raw_text_alice = open(alice_file).read()
raw_text_dracula = open(dracula_file).read()
raw_text_jekyll = open(jekyll_hyde_file).read()

raw_text_alice = raw_text_alice.lower()
raw_text_dracula = raw_text_dracula.lower()
raw_text_jekyll = raw_text_jekyll.lower()

alice = sorted(list(set(raw_text_alice)))
dracula = sorted(list(set(raw_text_dracula)))
jekyll_hyde = sorted(list(raw_text_jekyll))

