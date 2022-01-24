import numpy as np
import nltk
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import billboard
#from nltk_utils import bag_of_words, tokenize, stem
#from model import NeuralNet
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from imdb import IMDb
from nltk.metrics.distance  import edit_distance
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
nltk.download('words')
from spellchecker import SpellChecker

spell = SpellChecker()
from nltk.corpus import words
nltk.download('punkt')
nltk.download('popular')
nltk.download('stopwords')
correct_words = words.words()
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def spellcheck(sentence):

    unknwn = spell.unknown(list(sentence))

    

    corr = spell.correction(words)

    return corr


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag