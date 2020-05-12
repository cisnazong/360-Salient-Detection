from flask import Flask
from flask import request
import json
import numpy as np
import time
from model import LSTM
import torch
import torch.nn as nn
from torch.autograd import Variable
app = Flask(__name__)

model_dropout = 0.2
model_path = 'saved_model/model.pth'
model = LSTM(4, 128, 4, 2, dropout=model_dropout).cuda()
model.load_state_dict(torch.load(model_path))
criterion = nn.MSELoss()
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if request.method == 'POST':
        res:dict = json.loads(json.loads(request.data.decode('utf-8')))
        headmove_array = np.array(res["data"])
        _t = res["time"]
        headmove_predict_array = do_lstm_predict(headmove_array)
        headmove_predict_dict = {"data": headmove_predict_array.tolist(), "time": _t + 2}

        duration = time.time() - start_time
        print('duration:[%.0fms]' % (duration * 1000))
        return headmove_predict_dict

def do_lstm_predict(headmove_array):
    headmove_array = np.expand_dims(headmove_array, 0)
    headmove_array = headmove_array.astype(np.float32)
    test_x = torch.from_numpy(headmove_array.reshape(-1, 60, 4))
    var_x = Variable(test_x).cuda()
    pred_test = model(var_x)
    pred_test = pred_test.cpu().data.numpy()[:, :, :]

    return pred_test[0]

if __name__ == '__main__':
    app.run()