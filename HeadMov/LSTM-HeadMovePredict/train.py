import torch
from torch import nn
from torch.autograd import Variable
from model import LSTM
from utils import HeadPosDataset
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter

from utils import RecordLoaderAll, RecordLoaderClass
import time

# Define some hyper parameters
step_look_back = 60
step_look_forward = 60
save_interval = 50
log_interval = 1
num_epoch = 2000
ratio = 0.8
model_dir = './saved_models'
log_dir = './log'
model_path = './saved_models/Mon Apr 27 19:51:34 2020_lstm_itr_100_train_0.024368.pth'
refine = False
display_current_record = False
train_class = 'Paris-sJxiPiAaB4k'

# TensorBoard
writer = SummaryWriter(log_dir)
writer.add_text('project', "LSTM-HeadMovePredict")
writer.add_text('dataset', train_class)

# Abandoned functions
from utils import create_dataset
from utils import load_data_from_txt

# dataset = load_data_from_txt('/home/liyutong/Documents/LSTM-HeadMovePredict/ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Diving-2OzlksZBTiA/Diving-2OzlksZBTiA_0.txt')
# train_X, train_Y, test_X, test_Y = create_dataset(dataset,
#                                 look_back=step_look_back,
#                                 look_forward=step_look_forward,
#                                 divide_ratio=0.7)
# train_x = torch.from_numpy(train_X.reshape(-1, step_look_back, 4))
# train_y = torch.from_numpy(train_Y.reshape(-1, step_look_forward, 4))
# test_x = torch.from_numpy(test_X.reshape(-1, step_look_forward, 4))

# Define model
num_layers = 2
hidden_size = 128
model_dropout = 0.2
model = LSTM(4, hidden_size, 1, num_layers, dropout=model_dropout).cuda()

if refine:
	model.load_state_dict(torch.load(model_path))
	criterion = nn.MSELoss()  # Propose a new loss function, maybe...
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

else:
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print('num_layers: ', num_layers)
writer.add_text('num_layers: ', str(num_layers))
print('dropout: ', model_dropout)
writer.add_text('num_layers: ', str(model_dropout))
print('hidden_size: ', str(hidden_size))
writer.add_text('hidden_size: ', str(hidden_size))
print('refine: ', str(refine))
writer.add_text('refine', str(refine))
num_of_parameters = 0
for param in model.parameters():
	num_of_parameters += param.numel()
print('num of parameters: ', num_of_parameters)
writer.add_scalar('refine', num_of_parameters)

# Load records, log train_records and test_records
head_pos_records = RecordLoaderClass(r'./ML/dataset/*/*/*/*.txt', ratio=0.7, class_name=train_class)
train_records, test_records = head_pos_records.load()
writer.add_text("train_class", train_class)
writer.add_text("train_records", str(train_records))
writer.add_text("test_records", str(test_records))
print('num_of_train_records: ', len(train_records))
writer.add_text('num_of_train_records', str(len(train_records)))
if len(train_records) < 1:
	exit(-1)

for ite_num in range(num_epoch):
	time_prev = time.time()
	count = 0
	for record in train_records:
		count += 1
		if display_current_record:
			print('current record: ', record)

		# Method 1: use torch.Dataloader
		head_pos_dataset = HeadPosDataset(
			record,
			look_back=step_look_back,
			look_forward=step_look_forward,
			preprocess=True,
			column_names=['T', 'F', 'x', 'y', 'z', 'k'],
			index=['x', 'y', 'z', 'k'])
		head_pos_dataloader = DataLoader(head_pos_dataset, batch_size=100, num_workers=1)
		for var_x, var_y in head_pos_dataloader:
			var_x = Variable(var_x).cuda()
			var_y = Variable(var_y).cuda()
			# forward
			out = model(var_x)
			loss = criterion(out, var_y)

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	# Method 2: use custom functions
	# dataset = load_data_from_txt(record, show=False)
	# train_X, train_Y, test_X, test_Y = create_dataset(dataset,
	#                                                   look_back=step_look_back,
	#                                                   look_forward=step_look_forward,
	#                                                   divide_ratio=1)
	#
	# train_x = torch.from_numpy(train_X.reshape(-1, step_look_back, 4))
	# train_y = torch.from_numpy(train_Y.reshape(-1, step_look_forward, 4))
	#
	# var_x = Variable(train_x).cuda()
	# var_y = Variable(train_y).cuda()
	# # forward
	# out = model(var_x)
	# loss = criterion(out, var_y)
	#
	# # backward
	# optimizer.zero_grad()
	# loss.backward()
	# optimizer.step()
	#
	# del var_x, var_y, train_x, train_y
	# torch.cuda.empty_cache()


	writer.add_scalar('train loss', loss.data.item(), ite_num + 1)

	if (ite_num + 1) % log_interval == 0:
		print('Epoch: {}, Train Loss: {:.5f}'.format(ite_num + 1, loss.data.item()))

	# Evaluate
	model = model.eval()

	for record in test_records[:5]:
		head_pos_dataset = HeadPosDataset(
			record,
			look_back=step_look_back,
			look_forward=step_look_forward,
			preprocess=True,
			column_names=['T', 'F', 'x', 'y', 'z', 'k'],
			index=['x', 'y', 'z', 'k'])
		head_pos_dataloader = DataLoader(head_pos_dataset, batch_size=100, num_workers=1)
		for var_x, var_y in head_pos_dataloader:
			var_x = Variable(var_x).cuda()
			var_y = Variable(var_y).cuda()
			# forward
			out = model(var_x)
			loss = criterion(out, var_y)

	writer.add_scalar('eval loss', loss.data.item(), ite_num + 1)
	if (ite_num + 1) % log_interval == 0:
		print('Epoch: {}, Eval Loss: {:.5f}'.format(ite_num + 1, loss.data.item()))
	# continue to train
	model.train()

	writer.add_scalar('epoch', ite_num + 1, ite_num + 1)
	writer.add_scalar('GPU memory', torch.cuda.memory_allocated() / 1e6, ite_num + 1)
	writer.add_scalar('speed', count / (time.time() - time_prev), ite_num + 1)


	if (ite_num + 1) % save_interval == 0:
		print('saving the model')
		torch.save(model.state_dict(),
		           str(os.path.join(model_dir, "%s_lstm_itr_%d_train_%3f.pth" % (
			           time.asctime(time.localtime(time.time())), ite_num + 1, loss.data.item()))))

writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
writer.close()
