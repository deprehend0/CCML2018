# Assignment 3: Chatbot
-----
# Description

The goal of the assignment is to write a working chatbot by applying techniques learned throughout the CCML course or outside of the subject material. This meant that the chatbot should not only look for predefined keywords and get their answers from databases. Techniques such as synonym detection, text generation and/or sentiment analysis are techniques that could be used to make the chatbot more intelligent. 

The requirements for the chatbot are as followed:

- Do some basic chitchat
- Answering at least two types of domain specific questions?
- Incorporating any NLP technique to extend the functionality of the chatbot

# Idea
The idea for this assignment was to create a chatbot that is able to use movie quotes as responses to the Users messages. The corpus used for this assignment was the Cornell Movie Dialog corpus. 

The original NLP technique would be to employ a Sequence To Sequence Model. However, this proved to be too difficult within this short a timeframe to incorporate it into a Telegram Chatbot succesfully. Nevertheless, because the model is very interesting a small paragraph will be used to explain the method. 

# Sequence to Sequence
Sequence to Sequence model is a neural network consisted of two recurrent neural networks(RNN). One RNN encodes a sequence of symbols into a fixed-length vector representations, and the other decodes the representation into another sequence of symbols, The encoder and decoder of the proposed model are jointly trained to maximize the conditional probality of a target sequence given a source sequence. 

An example of how this could look is shown below. 

Each hidden state influences the next hidden state and the final hidden state can be seen as the summary of the sequence. This state is called the context or thought vector, as it represents the intention of the sequence. From the context, the decoder generates another sequence, one symbol(word) at a time. Here, at each time step, the decoder is influenced by the context and the previously generated symbols.

# Execution

```sh
Test
```

# Demo

```sh
Test
```


# Drawbacks

There are a few challenges in using this model. The most disturbing one is that the model cannot handle variable length sequences. It is disturbing because almost all the sequence-to-sequence applications, involve variable length sequences. 

The next one is the vocabulary size. The decoder has to run softmax over a large vocabulary of say 20,000 words, for each word in the output. That is going to slow down the training process, even if your hardware is capable of handling it. 

Representation of words is of great importance. How do you represent the words in the sequence? Use of one-hot vectors means we need to deal with large sparse vectors due to large vocabulary and there is no semantic meaning to words encoded into one-hot vectors.

These problems could be solved through padding and bucketing

# References


