from sklearn.metrics import mean_absolute_error
import requests
from PIL import Image
import numpy as np
import cv2
import base64
import json
import matplotlib.pyplot as plt
import pandas as pd

def image_to_base64(image_np: np.ndarray):
    image_cv2_encoded = cv2.imencode('.png', image_np)[1]
    image_base64: str = str(base64.b64encode(image_cv2_encoded))[2:-1]
    return image_base64


def base64_to_image(image_base64: str):
    image_cv2_encoded: str = base64.b64decode(image_base64)
    nparry: np.array = np.frombuffer(image_cv2_encoded, np.uint8)
    image_np: np.array = cv2.imdecode(nparry, cv2.IMREAD_COLOR)
    return image_np


def show_image_np(image_np: np.array, h, w):
    pil_image = Image.fromarray(np.uint8(image_np))
    pil_image = pil_image.resize((w, h))
    # pil_image.show()
    return pil_image


def predict(frame_np: np.array, url):
    h, w = frame_np.shape[0], frame_np.shape[1]
    frame_base64: str = image_to_base64(frame_np)

    payload: dict = {"image": frame_base64}
    response = requests.get(url, data=json.dumps(payload))

    response_json: dict = json.loads(response.text)
    mask_base64: str

    mask_base64: str = response_json["mask"]
    mask = base64_to_image(mask_base64)

    return show_image_np(mask, h, w)


def cv2_load_img(path):
    frame_np_BGR: np.ndarray = cv2.imread(path)
    frame_np_RGB: np.ndarray = cv2.cvtColor(frame_np_BGR, cv2.COLOR_BGR2RGB)
    h, w = frame_np_RGB.shape[0], frame_np_RGB.shape[1]
    return frame_np_RGB, h, w


if __name__ == "__main__":
    best_image_idx = 104
    test_img_path = "./stimulis/" + "{:0>3d}".format(best_image_idx) + ".png"
    test_label_path = "./objects/" + "{:0>3d}".format(best_image_idx) + ".png"
    predict_url = "http://liyutong-ubuntu.local:5000/predict"

    # read, predict
    frame_np, _, _ = cv2_load_img(test_img_path)
    pr = predict(frame_np, predict_url)
    frame_np, h, w = cv2_load_img(test_label_path)
    lbl = show_image_np(frame_np, h, w)
    pr_ary = np.array(pr.convert('L')) / 255
    lbl_ary = np.array(lbl.convert('L')) / 255

    print(mean_absolute_error(pr_ary, lbl_ary))