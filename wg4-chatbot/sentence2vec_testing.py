from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from nltk import word_tokenize
import numpy as np


# open quotes as array
raw_quotes = []
with open('./quotes/quotes.txt', mode='r') as file:
    raw_quotes = file.readlines()

gen_quotes = [[w.lower() for w in word_tokenize(quote)] for quote in raw_quotes]
dictionary = Dictionary(gen_quotes)
corpus = [dictionary.doc2bow(gen_quote) for gen_quote in gen_quotes]
tf_idf = TfidfModel(corpus)
similarities = Similarity('./quotes/quote_similarities/', tf_idf[corpus], num_features=len(dictionary))

query_doc = [w.lower() for w in word_tokenize('Sisters are for life')]
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(similarities[query_doc_tf_idf])
ms_idx = np.argmax(similarities[query_doc_tf_idf])
print(ms_idx)
print(len(raw_quotes))
print(raw_quotes[ms_idx])