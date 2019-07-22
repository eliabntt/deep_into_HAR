import json
import os
import telepot
from tensorflow.keras.callbacks import Callback


class TelegramCallback(Callback):

    def __init__(self, name=None, verbose=False):
        super(TelegramCallback, self).__init__()
        if not os.path.exists('tg_config.json'):
            self.disabled = True
            return
        self.disabled = False
        with open('tg_config.json','r') as confFile:
            config = json.load(confFile)
        self.users = config['users']
        self.bot = telepot.Bot(config['token'])
        if name != None:
            self.name = name
        else:
            self.name = self.model.name
        self.verbose = verbose

    def send_message(self, text):
        if self.disabled: return
        for user in self.users:
            try:
                self.bot.sendMessage(user, text)
            except Exception as e:
                print('ERROR sending message: {}'.format(e))

    def on_train_begin(self, logs={}):
        if not self.verbose: return
        text = '{}: Training started'.format(self.name)
        self.send_message(text)

    def on_train_end(self, logs={}):
        text = '{}: Training ended'.format(self.name)
        self.send_message(text)

    def on_epoch_end(self, epoch, logs={}):
        if not self.verbose: return
        text = '{}: Epoch {}.\n'.format(self.name, epoch)
        for k, v in logs.items():
            if k != "lr":
                text += '{}: {:.4f}; '.format(k, v)
            else:
                text += '{}: {:.6f}; '.format(k, v) #4 decimal places too short for learning rate
        self.send_message(text)
