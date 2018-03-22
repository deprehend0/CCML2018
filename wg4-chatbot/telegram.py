# basic telegram bot
# https://www.codementor.io/garethdwyer/building-a-telegram-bot-using-python-part-1-goi5fncay
# https://github.com/sixhobbits/python-telegram-tutorial/blob/master/part1/echobot.py

import json
import requests
import time
import urllib
import os
import logging
from markov_norder import Markov

# python3: urllib.parse.quote_plus
# python2: urllib.pathname2url

TOKEN = os.environ['CCML_BOT']  # don't put this in your repo! (put in config, then import config)
URL = "https://api.telegram.org/bot{}/".format(TOKEN)


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


def echo_all(updates):
    for update in updates.get('result'):
        msg = update.get('message')
        if msg:
            text = msg.get('text')
            chat = msg.get('chat').get('id')
            send_message(text, chat)


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
    greetings = ['hi', 'hello', 'ola', 'good morning', 'good evening', 'good day']
    print(any([g for g in greetings if g in urllib.pathname2url(text)]))
    return any([urllib.pathname2url(text).find(g) for g in greetings])


def send_welcome_message(update):
    text = 'Hi there good one! I\'m working on some more greetings'
    url = URL + "sendMessage?text={}&chat_id={}".format(text, update["message"]["chat"]["id"])
    get_url(url)


def is_goodbye_message(text):
    greetings = ['see you later', 'good bye', 'bye', 'cya', 'later', 'until next time']
    return any([urllib.pathname2url(text).find(g) for g in greetings])


def send_goodbye_message(update):
    text = 'Cya later aligator! I\'m working on some more greetings'
    url = URL + "sendMessage?text={}&chat_id={}".format(text, update["message"]["chat"]["id"])
    get_url(url)


def is_quote_request(text):
    return True


def send_about_quote(update):
    text = "I'm thinking about some good quotes, well actually it's still work in progress..."
    url = URL + "sendMessage?text={}&chat_id={}".format(text, update["message"]["chat"]["id"])
    get_url(url)


def process_input(updates):
    for update in updates.get('result'):
        msg = update.get('message')
        if msg:
            text = msg.get('text')
            if is_welcome_message(text):
                send_welcome_message(update)


def main():
    logging.basicConfig(filename='example.log', level=logging.INFO)
    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        if len(updates.get('result')) > 0:
            last_update_id = get_last_update_id(updates) + 1
            echo_all(updates)
            process_input(updates)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
