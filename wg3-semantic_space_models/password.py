#!/usr/bin/env python
from tokenize import String

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import *

# from gensim.scripts.glove2word2vec import glove2word2vec

# glove_input_file = './glove.6B.50d.txt'
# word2vec_output_file = './glove.6B.50d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)


# Load Google's pre-trained Word2Vec model
model = KeyedVectors.load_word2vec_format('./glove.6B.50d.txt.word2vec', binary=False)
# has plural and upper/lower case, and even bigrams (e.g., taxpayer_dollars; vast_sums)
# stemmer = SnowballStemmer("english")
stemmer = PorterStemmer()

# flex word2vec's muscles
model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())
model.most_similar("amsterdam")

# Consider a two-person task with a signaler and a receiver (similar to the TV gameshow 'Password'):
# The signalers were told that they would be playing a word-guessing game in which 
# they would have to think of one-word signals that would help someone guess their items. 
# They were talked through an example: if the item was 'dog', then a good signal would be 
# 'puppy' since most people given 'puppy' would probably guess 'dog'.

# sender thinks bank, says money
# receiver think cash
# print(model.most_similar("bank"))  # .69 robber, .67 robbery, robbers, security, agency ..
# print(model.most_similar("money"))  # .55 dollars, .55 profit, .54 cash
# print(model.most_similar("cash"))  # .69 capitalize, .54 money, sell, debt, tax

model['money']

model.similarity("hot", "cold")  # .20
model.similarity("hot", "warm")  # .14

# print(model.similarity("hot", "cold"))


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

bank_hint = send_word("sofa")
print(bank_hint)
bank_guess = receive_word(bank_hint)
print(bank_guess)
