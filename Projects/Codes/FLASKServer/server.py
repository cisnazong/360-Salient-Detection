from flask import Flask, request
from tools import Detector
import json, time
import cv2
app = Flask(__name__)
frameDetector = Detector('./saved_models/basnet_bsi/basnet.pth')
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['GET'])
def predict_image():
    global frameDetactor
    start_time = time.time()
    res:dict = json.loads(request.data)
    try:
        img_base64_png:str = res["image"]
    except:
        print("ERROR1")
        return

    msg: dict = {'mask': frameDetector.detect(img_base64_png)}
    duration = time.time() - start_time
    print ('duration:[%.0fms]' % (duration * 1000))
    return msg