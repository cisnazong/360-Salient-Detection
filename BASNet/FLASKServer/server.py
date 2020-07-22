from flask import Flask, request
from core.tools import Detector
import json, time

app = Flask(__name__)
frameDetector = Detector('core/saved_models/basnet_bsi/basnet_self_trained.pth')
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

if __name__ == '__main__':
    app.run(host='liyutong-ubuntu.local', port=5000)