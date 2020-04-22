# Evaluation
import torch
from model import LSTM
from utils import create_dataset
from utils import load_data_from_txt
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch import nn

model_path = './saved_models/lstm_itr_14000_train_0.006450.pth'

model = LSTM (4, 128, 4, 2).cuda ()
model.load_state_dict (torch.load (model_path))
model.eval ()
step_look_back = 30
step_look_forward = 30

dataset = load_data_from_txt (
	'./ML/dataset/uid-00d6d7f2-23df-4062-84dd-d5e99183dae1/test0/Diving-2OzlksZBTiA/Diving-2OzlksZBTiA_0.txt')
train_X, train_Y, test_X, test_Y = create_dataset (dataset,
                                                   look_back=step_look_back,
                                                   look_forward=step_look_forward,
                                                   divide_ratio=0)

test_x = torch.from_numpy (test_X.reshape (-1, step_look_forward, 4))
test_y = torch.from_numpy (test_X.reshape (-1, step_look_forward, 4))
model = model.eval ()
criterion = nn.MSELoss ()
var_x= Variable (test_x).cuda ()
var_y= Variable (test_y).cuda ()
pred_test = model (var_x)

loss = criterion (pred_test, var_y)
print('Loss:{:.5f}'.format(loss.data.item()))
# pred_test = np.squeeze (pred_test.cpu ().data.numpy (), axis=1)

pred_test = pred_test.cpu().data.numpy()[:,0,:]
ticks = np.linspace (0, 1, pred_test.shape[0])

# plt.title('test')
# plt.plot (ticks, test_Y[:, 0, 0], color='b')
# plt.plot (ticks, test_Y[:, 0, 1], color='g')
# plt.plot (ticks, test_Y[:, 0, 2], color='y')
# plt.plot (ticks, test_Y[:, 0, 3], color='r')
# plt.show ()

plt.title('prediction')
plt.ylim(-1.1,1.1)
plt.plot (ticks, pred_test[:, 0], color='b')
plt.plot (ticks, pred_test[:, 1], color='g')
plt.plot (ticks, pred_test[:, 2], color='y')
plt.plot (ticks, pred_test[:, 3], color='r')
plt.show ()
