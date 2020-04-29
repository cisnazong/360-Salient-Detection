# Evaluation
import torch
from model import LSTM
from utils import create_dataset
from utils import load_data_from_txt
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch import nn
from tensorboardX import SummaryWriter
from torchviz import make_dot

log_dir = './log'
dataset_path = './ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Rhino-training-7IWp875pCxQ/Rhino-training-7IWp875pCxQ_0.txt'
writer = SummaryWriter(log_dir)
model_path = './saved_models/Archive/class_paris_dropout_step120/Tue Apr 28 02:27:05 2020_lstm_itr_100_train_0.079369.pth'
model_dropout = 0.2
model = LSTM(4, 128, 4, 2, dropout=model_dropout).cuda()

model.load_state_dict(torch.load(model_path))
model.eval()

step_look_back = 120
step_look_forward = 120

dataset = load_data_from_txt(dataset_path, show=False)

train_X, train_Y, test_X, test_Y = create_dataset(dataset,
                                                  look_back=step_look_back,
                                                  look_forward=step_look_forward,
                                                  divide_ratio=0)

test_x = torch.from_numpy(test_X.reshape(-1, step_look_forward, 4))
test_y = torch.from_numpy(test_X.reshape(-1, step_look_forward, 4))
model = model.eval()
criterion = nn.MSELoss()
var_x = Variable(test_x).cuda()
var_y = Variable(test_y).cuda()

# writer.add_graph(model, (var_x,))
pred_test = model(var_x)

loss = criterion(pred_test, var_y)
print('Loss:{:.5f}'.format(loss.data.item()))
# pred_test = np.squeeze (pred_test.cpu ().data.numpy (), axis=1)

pred_test = pred_test.cpu().data.numpy()[:, 0, :]
ticks1 = np.linspace(0, 1, dataset.shape[0])
ticks2 = np.linspace(0, 1, pred_test.shape[0])

# plt.title('test')
# plt.plot (ticks, test_Y[:, 0, 0], color='b')
# plt.plot (ticks, test_Y[:, 0, 1], color='g')
# plt.plot (ticks, test_Y[:, 0, 2], color='y')
# plt.plot (ticks, test_Y[:, 0, 3], color='r')
# plt.show ()

fig1 = plt.figure(1, figsize=(8, 6))
plt.title('original')
plt.ylim(-1.1, 1.1)
plt.plot(ticks1, dataset[:, 0], color='b')
plt.plot(ticks1, dataset[:, 1], color='g')
plt.plot(ticks1, dataset[:, 2], color='y')
plt.plot(ticks1, dataset[:, 3], color='r')
plt.show()
writer.add_figure('dataset: '+ dataset_path, figure=fig1)

fig2 = plt.figure(2, figsize=(8, 6))
plt.title('prediction')
plt.ylim(-1.1, 1.1)
plt.plot(ticks2, pred_test[:, 0], color='b')
plt.plot(ticks2, pred_test[:, 1], color='g')
plt.plot(ticks2, pred_test[:, 2], color='y')
plt.plot(ticks2, pred_test[:, 3], color='r')
plt.show()
writer.add_figure('dataset: '+ dataset_path + ', model: ' + model_path.split('/')[-2], figure=fig2)
writer.close()