# Assignment 3: Chatbot
-----
### Description

The goal of the assignment is to write a working chatbot by applying techniques learned throughout the CCML course or outside of the subject material. This meant that the chatbot should not only look for predefined keywords and get their answers from databases. Techniques such as synonym detection, text generation and/or sentiment analysis are techniques that could be used to make the chatbot more intelligent. 

The requirements for the chatbot are as followed:

- Do some basic chitchat
- Answering at least two types of domain specific questions?
- Incorporating any NLP technique to extend the functionality of the chatbot

### Idea
The idea for this assignment was to create a chatbot that is able to use movie quotes as responses to the Users messages. The corpus used for this assignment was a self-created quotes list. 

The original NLP technique would be to employ a Sequence To Sequence Model. However, this proved to be too difficult within this short a timeframe to incorporate it into a Telegram Chatbot succesfully. Nevertheless, because the model is very interesting a small paragraph will be used to explain the method. 

The eventual experiment used the tf-idf model. TF-IDF stands for term frequency-inverse document frequency. Term frequency is how often the word shows up in the document and inverse document frequency scales the value by how rare the word is in the corpus. 

### Sequence to Sequence
Sequence to Sequence model is a neural network consisted of two recurrent neural networks(RNN). One RNN encodes a sequence of symbols into a fixed-length vector representations, and the other decodes the representation into another sequence of symbols, The encoder and decoder of the proposed model are jointly trained to maximize the conditional probality of a target sequence given a source sequence[1]. 

An example of how this could look is shown below. 

![alt text](http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png) "Encoder Decoder")

Each hidden state influences the next hidden state and the final hidden state can be seen as the summary of the sequence. This state is called the context or thought vector, as it represents the intention of the sequence[2]. From the context, the decoder generates another sequence, one symbol(word) at a time. Here, at each time step, the decoder is influenced by the context and the previously generated symbols[1].

### Execution
The first step consisted of generating a corpus of, in this case, quotes and preprocessing them. So, that these can be trained in the tf-idf model(preprocess_quotes.py)[3]. 

```sh
quote_lines = []
with open('./quotes/memorable_quotes.txt', mode='r') as file:
    quote_lines = file.readlines()

with open('./quotes/non_memorable_quotes.txt', mode='r') as file:
    lines = file.readlines()
    quote_lines = quote_lines + lines
```

Afterwards NLTK was used to tokenize the quotes and to create a dictionary to maps every word to a number. This dictionary is turned into a corpus. This serves as a bag-of-words representations for the number of times each word occurs in the document(sentence2vec_testing.py)[3]. 

```sh
gen_quotes = [[w.lower() for w in word_tokenize(quote)] for quote in raw_quotes]
dictionary = Dictionary(gen_quotes)
corpus = [dictionary.doc2bow(gen_quote) for gen_quote in gen_quotes]
```

This corpus gets turned into a tf-idf model. This tf-idf model is turned into a similarity measure object. Now queries can be used to find the most similar document based on the corpus. 

```sh
similarities = Similarity('./quotes/quote_similarities/', tf_idf[corpus], num_features=len(dictionary))

query_doc = [w.lower() for w in word_tokenize('Sisters are for life')]
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(similarities[query_doc_tf_idf])
ms_idx = np.argmax(similarities[query_doc_tf_idf])
print(ms_idx)
print(len(raw_quotes))
print(raw_quotes[ms_idx])
```

This was integrated in to the Telegram chatbot. In which, the bot responds with the most similar quotes based on the user sent message. 

### Demo

An demo based on the source code are as followed: 

```sh
query_doc = [w.lower() for w in word_tokenize('Sisters are for life')]
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(similarities[query_doc_tf_idf])
ms_idx = np.argmax(similarities[query_doc_tf_idf])
print(ms_idx)
print(len(raw_quotes))
print(raw_quotes[ms_idx])
```

![alt text](https://github.com/deprehend0/CCML2018/blob/master/wg4-chatbot/images/life%20quote.png?raw=true)

Other examples that could be generated are as such.

![alt text](https://github.com/deprehend0/CCML2018/blob/master/wg4-chatbot/images/python%20quote.png?raw=true)

![alt text](https://github.com/deprehend0/CCML2018/blob/master/wg4-chatbot/images/hello%20quote.png?raw=true)

Incorporating them in the Telegram chatbot (telegram.py). Results in the following interactions

![alt text](https://github.com/deprehend0/CCML2018/blob/master/wg4-chatbot/images/Telegram%20chat.png?raw=true)

With this, we have created a chatbot that uses quotes to answer specific domain questions. 

### Drawbacks

Normally, only the drawbacks will be mentioned of the used model. However, because Seq2Seq was the original idea. The drawbacks for this method will still be mentioned. 

##### Seq2Seq

There are a few challenges in using this model. The most disturbing one is that the model cannot handle variable length sequences. It is disturbing because almost all the sequence-to-sequence applications, involve variable length sequences[1]. 

The next one is the vocabulary size. The decoder has to run softmax over a large vocabulary of say 20,000 words, for each word in the output. That is going to slow down the training process, even if your hardware is capable of handling it[1]. 

These problems could be solved through padding and bucketing[1].

##### TF-IDF
Drawbacks of TF-IDF mostly consists of the size and content of the used corpus. Because it does not distinguish between synonyms it cannot make the connections between words. In a experiment done by Ramos. J. TF-IDF could not equate the word drug with its plural drugs categorizing each instead as separate words and slightly decreasing the words wd value. For large document
collections, this could present an escalating problem[4]. 

#### Do it Yourself 
The environment and version of the package used for this experiment were as such:

```sh
Python 3.6.4
Gensim = 3.4.0
NLTK = 3.2.5
Requests = 2.18.4
Urllib3 = 1.22
```

The setup can be downloaded interactively from the Github by selecting and downloading Chatbot Telegram TF-IDF.7z from the Master Branch.

#### References
1. Ram, S. (2016, June 28). Chatbots with Seq2Seq. Accessed on March 20, 2018, from Learn to build a chatbot using TensorFlow: http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
2. Cho, K., Merrienboer, B., Gulchere, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. Universite de Montreal. Accessed On March 20, 2018, from https://www.aclweb.org/anthology/D14-1179
3. Mugan, J. (2017, April 18). How do I compare document similarity using Python? Accessed on March 19, 2018, from Oreilly: https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python
4. Ramos, J. (2003). Using TF-IDF to Determine Word Relevance in Document Queries. Piscataway: Department of Computer Science, Rutgers University. Accessed on March 20, 2018, from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.1424&rep=rep1&type=pdf


