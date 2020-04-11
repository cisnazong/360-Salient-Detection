import numpy as np
from PIL import Image
import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
import pycuda.autoinit  # very important import

from pycuda.compiler import SourceModule
import multiprocessing


class ImageProjectorTarget (object):
	def __init__(self, h, w, d=3, block=(16, 16, 3)):
		self.height = h  # heightß
		self.width = w  # width
		e = int (self.width / 4)
		self.depth = d
		self.edge = np.array ([int (self.width / 4)])  # edge
		self.block_dim = block
		self.grid_dim = (int (self.edge / self.block_dim[0]), int (self.edge / self.block_dim[1]), 1)
		self.Output = [np.zeros (shape=[e, e, d]).astype (np.uint8) for i in range(6)]

# Usage

# img_path = '../basnet-master/train_data/F-360SOD/stimulis/001.png'
# imgIn = np.array (Image.open (img_path)).astype (np.int8)  # (h,w,c)
#
# imgIn_edge = int (imgIn.shape[1] / 4)
# imgIn_color_depth = imgIn.shape[-1]
# imgOut = np.zeros (shape=[imgIn_edge, imgIn_edge, imgIn_color_depth]).astype (np.int8)
#
#
# projector[0](
#          drv.Out(imgOut), drv.In(imgIn), drv.In(np.array([imgIn_edge])),
#          block=(16,16,3),grid=(32,32,1))
#
# imgOut = Image.fromarray(np.uint8(imgOut)).convert('RGB')
# imgOut.save('out.jpeg')


class ImageCorpCUDA:
	def __init__(self):
		pass

	def resize(self, size: tuple, imgIn: np.array):
		pass


# def process(projector, Output, imgIn, edge, block_dim, grid_dim):
# 	projector (Output, imgIn, edge, block_dim=block_dim, grid_dim=grid_dim)
