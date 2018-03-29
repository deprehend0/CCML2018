# basic telegram bot
# https://www.codementor.io/garethdwyer/building-a-telegram-bot-using-python-part-1-goi5fncay
# https://github.com/sixhobbits/python-telegram-tutorial/blob/master/part1/echobot.py

import json
import requests
import time
import urllib
import os
import logging
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from nltk import word_tokenize
import numpy as np
import numbers
import random

# python3: urllib.parse.quote_plus
# python2: urllib.pathname2url

TOKEN = os.environ['CCML_BOT']  # don't put this in your repo! (put in config, then import config)
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

with open('./quotes/quotes.txt', mode='r') as file:
    raw_quotes = file.readlines()
gen_quotes = [[w.lower() for w in word_tokenize(quote)] for quote in raw_quotes]
dictionary = Dictionary(gen_quotes)
corpus = [dictionary.doc2bow(gen_quote) for gen_quote in gen_quotes]
tf_idf = TfidfModel(corpus)
similarities = Similarity('./quotes/quote_similarities/', tf_idf[corpus], num_features=len(dictionary))

# gen_quotes = None
# dictionary = None
# corpus = None
# tf_idf = None
# similarities = None
# raw_quotes = None
#
# def initialize_similairty_model():


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates"
    if offset:
        url += "?offset={}".format(offset)
    js = get_json_from_url(url)
    return js


def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


def echo_message(update):
    msg = update.get('message')
    if msg:
        text = msg.get('text')
        chat = msg.get('chat').get('id')
        send_message(text, chat)



def echo_all(updates):
    for update in updates.get('result'):
        echo_message(update)


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def send_message(text, chat_id):
    text = urllib.pathname2url(text)  # urllib.parse.quote_plus(text) # (python3)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)


def send_speech_based_text(updates, text):
    for update in updates["result"]:
        # text = urllib.pathname2url(text) # urllib.parse.quote_plus(text) # (python3)
        chat_id = update["message"]["chat"]["id"]
        url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
        get_url(url)


def is_welcome_message(text):
    greetings = ['hello', 'good morning', 'good evening']
    return any(g in text.lower() for g in greetings)


def send_welcome_message(update):
    greetings = ['Hello', 'Good day', 'Welcome', 'Bon jour']
    text = random.choice(greetings)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, update["message"]["chat"]["id"])
    get_url(url)


def is_about_message(text):
    about_messages = ['Who are you', 'what are you']
    return any(a.lower() in text.lower() for a in about_messages)


def send_about_message(update):
    text = 'Well... I\'m an aritfical person who responds to you. When you ask a quote for or about something, ' \
           'then I\'ll try to respond with an appropriate quote.'
    url = URL + "sendMessage?text={}&chat_id={}".format(text, update["message"]["chat"]["id"])
    get_url(url)


def is_goodbye_message(text):
    greetings = ['bye', 'cya', 'see you later', 'good bye']
    return any(g.lower() in text.lower() for g in greetings)


def send_goodbye_message(update):
    greetings = ['Cya later', 'Thanks for passing by', 'Have a good day', 'Until next time']
    text = random.choice(greetings)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, update["message"]["chat"]["id"])
    get_url(url)


def is_quote_request(text):
    quote_requests = ['quote for', 'quote about']
    return any(q.lower() in text.lower() for q in quote_requests)


def send_quote(update, user_text):
    if 'quote for' in user_text:
        relevant_part = user_text.split('quote for')[1]
    elif 'quote about' in user_text:
        relevant_part = user_text.split('quote about')[1]
    query_doc = [w.lower() for w in word_tokenize(relevant_part)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    ms_idx = np.argmax(similarities[query_doc_tf_idf])
    if isinstance(ms_idx, numbers.Number):
        text = raw_quotes[ms_idx]
    else:
        text = raw_quotes[random.choice(ms_idx)]
    url = URL + "sendMessage?text={}&chat_id={}".format(text, update["message"]["chat"]["id"])
    get_url(url)


def send_greeting(text, update):
    if is_goodbye_message(text):
        send_goodbye_message(update)
    elif is_welcome_message(text):
        send_welcome_message(update)
    elif is_about_message(text):
        send_about_message(update)
    else:
        echo_message(update)


def process_input(updates):
    for update in updates.get('result'):
        msg = update.get('message')
        if msg:
            text = msg.get('text').encode('utf8')
            if is_quote_request(text):
                send_quote(update, text)
            else:
                send_greeting(text, update)


def main():
    logging.basicConfig(filename='example.log', level=logging.INFO)
    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        if updates.get('result') and len(updates.get('result')) > 0:
            last_update_id = get_last_update_id(updates) + 1
            process_input(updates)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
