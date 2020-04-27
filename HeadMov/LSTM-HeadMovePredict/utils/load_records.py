# records_user = [
# 	'./ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Diving-2OzlksZBTiA/Diving-2OzlksZBTiA_0.txt',
# 	'./ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Paris-sJxiPiAaB4k/Paris-sJxiPiAaB4k_0.txt',
# 	'./ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Rhino-training-7IWp875pCxQ/Rhino-training-7IWp875pCxQ_0.txt',
# 	'./ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Rollercoaster-8lsB-P8nGSM/Rollercoaster-8lsB-P8nGSM_0.txt',
# 	'./ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Timelapse-CIw8R8thnm8/Timelapse-CIw8R8thnm8_0.txt',
# 	'./ML/dataset/uid-0f33f0d6-6c9a-4641-b685-76397da22681/test0/Venise-s-AJRFQuAtE/Venise-s-AJRFQuAtE_0.txt',
# ]
#
# dataset_root_path = './ML/dataset/'
#
# video_list = ['Diving-2OzlksZBTiA',
#               'Paris-sJxiPiAaB4k',
#               'Rhino-training-7IWp875pCxQ',
#               'Rollercoaster-8lsB-P8nGSM',
#               'Timelapse-CIw8R8thnm8',
#               'Venise-s-AJRFQuAtE']


class RecordLoaderAll(object):
	def __init__(self, glob_pattern, ratio):
		self.train_records = []
		self.test_records = []
		self.glob_pattern = glob_pattern
		self.ratio = ratio

	def load(self):
		import glob, random
		path_list = glob.glob(self.glob_pattern)
		random.shuffle(path_list)
		train_size = int(self.ratio * len(path_list))
		self.train_records = path_list[0:train_size]
		self.test_records = path_list[train_size:]
		return self.train_records, self.test_records


class RecordLoaderClass(object):
	def __init__(self, glob_pattern, ratio, class_name):
		self.train_records = []
		self.test_records = []
		self.glob_pattern = glob_pattern
		self.class_name = class_name
		self.ratio = ratio

	def load(self):
		import glob, random
		path_list = []
		if self.class_name == 'all':
			path_list = glob.glob(self.glob_pattern)
		else:
			for path in glob.glob(self.glob_pattern):
				if path.find(self.class_name) >= 0:
					path_list.append(path)
		random.shuffle(path_list)
		train_size = int(self.ratio * len(path_list))
		self.train_records = path_list[0:train_size]
		self.test_records = path_list[train_size:]
		return self.train_records, self.test_records
# r'./ML/dataset/*/*/*/*.txt'
