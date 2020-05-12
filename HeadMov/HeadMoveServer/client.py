import socket
import random
import numpy as np
import json
import time
from multiprocessing import Process, Queue
class RandomPosGenerator(object):
    def __init__(self, seed):
        self.seed = seed
        random.seed(self.seed)
    @classmethod
    def read(self):
        x = (random.random() - 0.5) * 2
        y = (random.random() - 0.5) * 2
        z = (random.random() - 0.5) * 2
        k = (random.random() - 0.5) * 2
        return {"time":time.time(), "x":x,"y":y,"z":z,"k":k}

def json_str_to_dict(json_str:str):
    return json.loads(json_str)

def send_msg(host:str, port:int, signal:Queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        data_dict = RandomPosGenerator.read()
        data_str = json.dumps(data_dict)
        # print('[ send ] ', data_str)
        s.sendto(data_str.encode('utf-8'), (host, port))
        if signal.empty():
            pass
        else:
            if signal.get(timeout=1e-3) != 0:
                break
        time.sleep(1 / 30)
    s.close()

def recv_msg(host:str, port:int, signal:Queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    while True:
        headpos_prediction_str = s.recv(1024).decode('utf-8')
        headpos_prediction_dict = json_str_to_dict(headpos_prediction_str)
        print('[ recv ] ', headpos_prediction_dict)
        if signal.empty():
            pass
        else:
            if signal.get(timeout=1e-3) != 0:
                break
    s.close()

if __name__ == '__main__':
    signal = Queue()
    p_send = Process(target=send_msg, args=('127.0.0.1', 19998, signal,))
    p_recv = Process(target=recv_msg, args=('127.0.0.1', 19997, signal,))
    p_send.start()
    p_recv.start()
    p_send.join()
    p_recv.join()

