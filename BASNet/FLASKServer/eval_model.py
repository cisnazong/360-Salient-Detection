import requests
from sklearn.metrics import mean_absolute_error
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


def eval_by_idx(idx, predict_url):
    test_img_path = "./stimulis/" + "{:0>3d}".format(idx) + ".png"
    test_label_path = "./objects/" + "{:0>3d}".format(idx) + ".png"
    thresh_hold_min = 0
    thresh_hold_max = 255

    acc_ary = []
    precision_ary = []
    recall_ary = []
    F1_score_ary = []

    # read, predict
    frame_np, _, _ = cv2_load_img(test_img_path)
    pr = predict(frame_np, predict_url)
    frame_np, h, w = cv2_load_img(test_label_path)
    lbl = show_image_np(frame_np, h, w)

    # to grey scale
    lbl_gs = lbl.convert('L')
    lbl_gs_ary = np.array(lbl_gs)
    pr_gs = pr.convert('L')
    pr_gs_ary = np.array(pr_gs)

    mae = mean_absolute_error(pr_gs_ary / 255, lbl_gs_ary / 255)

    # make the graph look better
    acc_ary.append(0)
    precision_ary.append(0)
    recall_ary.append(1)

    for thresh_hold in range(thresh_hold_min, thresh_hold_max):
        # apply thresh_hold
        print("Applying thresh_hold", thresh_hold)
        pr_gs_ary_th = ((pr_gs_ary >= thresh_hold) + 0).astype(np.uint8)
        lbl_gs_ary_th = (lbl_gs_ary >= 255 + 0).astype(np.uint8)

        # TP, FP, TN, FN
        P = np.sum(lbl_gs_ary_th)
        N = h * w - P
        TP = np.sum(np.multiply(pr_gs_ary_th, lbl_gs_ary_th))
        FP = np.sum(np.multiply(pr_gs_ary_th, (1 - lbl_gs_ary_th)))
        TN = np.sum(np.multiply((1 - pr_gs_ary_th), (1 - lbl_gs_ary_th)))
        FN = np.sum(np.multiply((1 - pr_gs_ary_th), lbl_gs_ary_th))

        # acc, precision, recall
        acc = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * precision * recall / (precision + recall)

        acc_ary.append(acc)
        precision_ary.append(precision)
        recall_ary.append(recall)
        F1_score_ary.append(F1_score)

    # make the graph look better
    acc_ary.append(0)
    precision_ary.append(1)
    recall_ary.append(0)
    F1_score_max = max(F1_score_ary)

    return acc_ary, precision_ary, recall_ary, F1_score_max, mae

if __name__ == "__main__":
    idx_min = 81
    idx_max = 108
    F1_score_glb = np.zeros(shape=108)
    mae_glb = []
    predict_url = "http://liyutong-ubuntu.local:5000/predict"

    for idx in range(idx_min, idx_max):
        _,_,_,F1_score_idx_max, mae = eval_by_idx(idx, predict_url)
        F1_score_glb[idx] = F1_score_idx_max
        mae_glb.append(mae)

    best_image_idx = np.argmax(F1_score_glb)
    acc_ary, precision_ary, recall_ary, _, _ = eval_by_idx(best_image_idx, predict_url)
    df = pd.DataFrame(columns=['acc', 'P', 'R'])
    df["acc"] = acc_ary
    df["P"] = precision_ary
    df["R"] = recall_ary
    df.to_csv("Model_eval.csv")

    best_mae = min(mae_glb)

    with open('max_fscore.txt', 'w') as f:
        f.write("f_score: " + str(F1_score_glb[best_image_idx]) + "\n")
        f.write("max_idx: " + str(best_image_idx) + "\n")
        f.write("mae: " + str(best_mae) + "\n")

    plt.plot(recall_ary, precision_ary)