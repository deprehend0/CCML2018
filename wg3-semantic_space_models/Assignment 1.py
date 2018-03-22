#!/usr/bin/env python
from tokenize import String
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import *

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

stemmer = PorterStemmer()

def send_word(password):
    similar_words = model.most_similar(password)
    unique_words = []
    stemmed = []
    for word, score in similar_words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in model.vocab:
            similarity_score = model.similarity(password, stemmed_word)
            if stemmed_word not in unique_words and stemmed_word != stemmer.stem(password):
                stemmed.append((stemmed_word, similarity_score))
                unique_words.append(stemmed_word)

    return max(stemmed, key=lambda word: stemmed[1])[0]


def receive_word(hint):
    similar_words = model.most_similar(hint)
    unique_words = []
    possibilities = []
    for word, score in similar_words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in model.vocab:
            similarity_score = model.similarity(hint, stemmed_word)
            if stemmed_word not in unique_words and stemmed_word != stemmer.stem(hint):
                possibilities.append((stemmed_word, similarity_score))
                unique_words.append(stemmed_word)

    return max(possibilities, key=lambda guess: possibilities[1])[0]

def score(secret, hints):
    score = model.similarity(secret, hints)
    return score 

### Test
bank_hint = send_word("sofa")
print(bank_hint)
bank_guess = receive_word(bank_hint)
print(bank_guess)

### Evaluation
A = send_word("lost")
B = receive_word(A)
C = score("lost", B)
print(C)
