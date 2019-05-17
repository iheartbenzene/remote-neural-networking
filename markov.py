import markov
import nltk
import re
import pprint
import random

import wolframclient

class Markov(object):
    def __init__(self, order = 2, dictFile = '', max_words = 100):
        self.table = {}
        self.inputLineCount = 0
        self.inputWordCount = 0
        self.setOrder(order)
        self.setMaxWords(max_words)
        if dictFile == False:
            pass
