import nltk
from imdb import IMDb
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import random
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spellchecker import SpellChecker
from nltk_utils import bag_of_words, spellcheck, tokenize, stem
from model import NeuralNet
from imdb import IMDb
from flask import Flask, render_template, request
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
nltk.download('words')
from nltk.corpus import words
spell = SpellChecker()
  
  
correct_words = words.words()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'rb') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]



model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()



def chatbot_response(msg):
    sentence = tokenize(msg)

    # #unkwn = spell.unknown(sentence)
    # corr = spell.correction(str(sentence))

    # print(corr)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]



    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
            if tag == "movie":
                movies = IMDb()
                search = movies.get_top250_movies()
                mov = []
                for i in range(10):
                    res = search[i]['title']
                    mov.append(res)
                response = str(mov)
    else:
        response = "sORRY!"

    return response


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")

def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()
