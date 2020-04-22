import torch
from torch import nn
from torch.autograd import Variable
from model import LSTM
from utils import HeadPosDataset
from torch.utils.data import DataLoader
import os

# Define some parameters
step_look_back = 120
step_look_forward = 120
save_interval = 2000
print_interval = 50
num_epoch = 20000
model_dir = './saved_models'
model_path = './saved_models/lstm_itr_14000_train_0.006450.pth'
refine = False
# Abandoned functions
# from utils import create_dataset
# from utils import load_data_from_txt
# dataset = load_data_from_txt('/home/liyutong/Documents/LSTM-HeadMovePredict/ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Diving-2OzlksZBTiA/Diving-2OzlksZBTiA_0.txt')
# train_X, train_Y, test_X, test_Y = create_dataset(dataset,
#                                 look_back=step_look_back,
#                                 look_forward=step_look_forward,
#                                 divide_ratio=0.7)


head_pos_dataset = HeadPosDataset (
	'./ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Diving-2OzlksZBTiA/Diving-2OzlksZBTiA_0.txt',
	look_back=step_look_back,
	look_forward=step_look_forward,
	preprocess=True,
	column_names=['T', 'F', 'x', 'y', 'z', 'k'],
	index=['x', 'y', 'z', 'k'])

model = LSTM (4, 25, 4, 2).cuda()
if refine:
	model.load_state_dict(torch.load(model_path))
	criterion = nn.MSELoss ()
	optimizer = torch.optim.Adam (model.parameters (), lr=1e-4)

else:
	criterion = nn.MSELoss ()
	optimizer = torch.optim.Adam (model.parameters (), lr=1e-2)

head_pos_dataloader = DataLoader (head_pos_dataset, batch_size=3000, num_workers=1)

for ite_num in range (num_epoch):
	for var_x, var_y in head_pos_dataloader:
		var_x = Variable (var_x).cuda ()
		var_y = Variable (var_y).cuda ()
		# forward
		out = model (var_x)
		loss = criterion (out, var_y)

		# backward
		optimizer.zero_grad ()
		loss.backward ()
		optimizer.step ()

	if (ite_num + 1) % print_interval == 0:
		print ('Epoch: {}, Loss: {:.5f}'.format (ite_num + 1, loss.data.item ()))

	if (ite_num + 1) % save_interval == 0:
		print ('saving the model')
		torch.save (model.state_dict (),
		            str (os.path.join (model_dir, "lstm_itr_%d_train_%3f.pth" % (ite_num + 1, loss.data.item ()))))
