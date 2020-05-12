import socket
import random
import numpy as np
import json
import time
from multiprocessing import Process, Queue
import requests


class LSTMPredict():

    def __init__(self):
        self.api = 'http://127.0.0.1:5000/predict'
        self.headpos_array: np.array = np.zeros(shape=(60, 4))
        self.index = 0

    def predict(self, headpos_dict: dict):
        _t = headpos_dict["time"]
        _x = headpos_dict["x"]
        _y = headpos_dict["y"]
        _z = headpos_dict["z"]
        _k = headpos_dict["k"]
        if self.index < 60:
            self.headpos_array[self.index] = [_x, _y, _z, _k]
            self.index += 1

        if self.index >= 60:
            self.index = 0
            package = {"data": self.headpos_array.tolist(), "time": _t}
            try:
                response = requests.post(self.api, json=json.dumps(package))
            except:
                return None

            print("[ info ] Calling api")
            if response.ok:
                return response.text
            else:
                return None
        else:
            return None


def json_str_to_dict(json_str: str):
    return json.loads(json_str)


def dict_to_json_str(dict: str):
    return json.dumps(dict)


def send_msg(host: str, port: int, msg_queue: Queue, signal_queue: Queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    P = LSTMPredict()
    while True:
        if msg_queue.empty():
            continue
        else:
            headpos_str = msg_queue.get()
        headpos_dict = json_str_to_dict(headpos_str)
        headpos_prediction_str = P.predict(headpos_dict)
        if headpos_prediction_str == None:
            continue
        headpos_prediction_dict = json_str_to_dict(headpos_prediction_str)
        for i in range(60):
            try:
                headpos_prediction_slice = {"time": headpos_prediction_dict["time"] + i / 30,
                                            "x": headpos_prediction_dict["data"][i][0],
                                            "y": headpos_prediction_dict["data"][i][1],
                                            "z": headpos_prediction_dict["data"][i][2],
                                            "k": headpos_prediction_dict["data"][i][3]}
            except IndexError:
                break
            headpos_prediction_slice_str = dict_to_json_str(headpos_prediction_slice)
            print('[ send ] ', headpos_prediction_slice_str)
            s.sendto(headpos_prediction_slice_str.encode('utf-8'), (host, port))
        for i in range(msg_queue.qsize()):
            msg_queue.get_nowait()
        if signal.empty():
            pass
        else:
            if signal.get(timeout=1e-3) != 0:
                break
    s.close()


def recv_msg(host: str, port: int, msg_queue: Queue, signal_queue: Queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    while True:
        headpos_str = s.recv(1024).decode('utf-8')
        msg_queue.put(headpos_str)
        # print('[ recv ] ', headpos_str)
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
