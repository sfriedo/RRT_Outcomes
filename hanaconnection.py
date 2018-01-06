import pyhdb
import socket
import json

HOST = ''
PORT = 0
USER = ''
PW = ''

try:
    with open('config.txt', 'rb') as cfg:
        cfg = json.load(cfg)
        HOST = cfg['host']
        PORT = cfg['port']
        USER = cfg['user']
        PW = cfg['password']
        print('Config loaded')
except Exception:
    print('Could not find config.txt, Database connection won\'t work!')


class HanaConnection(object):
    def __init__(self):
        try:
            self.connection = pyhdb.connect(
                host=HOST,
                port=PORT,
                user=USER,
                password=PW,
                autocommit=True
            )
            self.cursor = self.connection.cursor()
        except socket.gaierror as e:
            print('Database instance is not available! \n\n')
            raise e

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
        self.cursor.close()
        self.connection.close()
