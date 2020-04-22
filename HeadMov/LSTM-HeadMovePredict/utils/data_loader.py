import torch


class data_loader (torch.utils.data.DataLoader):
	def __init__(self,
	             dataset,
	             batch_size=1,
	             shuffle=False,
	             sampler=None,
	             num_workers=0,
	             pin_memory=False,
	             drop_last=False):
		pass
