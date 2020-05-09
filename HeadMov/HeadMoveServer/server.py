import socket
import random
import numpy as np
import json
import time
from multiprocessing import Process, Queue

class PseudoPredict():
    def __init__(self):
        pass

    @classmethod
    def predict(self, headpos_dict:dict):
        _t = headpos_dict["time"]
        _x = headpos_dict["x"]
        _y = headpos_dict["y"]
        _z = headpos_dict["z"]
        _k = headpos_dict["k"]
        return {"time":_t + 2, "x":_x, "y":_y, "z":_z, "k":_k}

def json_str_to_dict(json_str:str):
    return json.loads(json_str)

def dict_to_json_str(dict:str):
    return json.dumps(dict)

def send_msg(host:str, port:int, msg_queue:Queue, signal_queue:Queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        if msg_queue.empty():
            continue
        else:
            headpos_str = msg_queue.get()
        headpos_dict = json_str_to_dict(headpos_str)
        headpos_prediction_dict = PseudoPredict.predict(headpos_dict)
        headpos_prediction_str = dict_to_json_str(headpos_prediction_dict)
        print('[ send ] ', headpos_str)
        s.sendto(headpos_prediction_str.encode('utf-8'), (host, port))
        if signal.empty():
            pass
        else:
            if signal.get(timeout=1e-3) != 0:
                break
    s.close()

def recv_msg(host:str, port:int, msg_queue:Queue, signal_queue:Queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    while True:
        headpos_str = s.recv(1024).decode('utf-8')
        msg_queue.put(headpos_str)
        print('[ recv ] ', headpos_str)
        if signal.empty():
            pass
        else:
            if signal.get(timeout=1e-3) != 0:
                break
    s.close()

if __name__ == '__main__':
    signal = Queue()
    msg_queue = Queue()
    p_send = Process(target=send_msg, args=('127.0.0.1', 19997, msg_queue, signal,))
    p_recv = Process(target=recv_msg, args=('127.0.0.1', 19998, msg_queue, signal,))
    p_send.start()
    p_recv.start()
    p_send.join()
    p_recv.join()
