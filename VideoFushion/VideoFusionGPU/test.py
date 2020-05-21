import py360convert
import glob
import os
import PIL
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import numba.cuda as cuda
import numba
from utils import project, copy

sample = './sample/001.png'
imgIn = np.array(Image.open(sample))
img_flat = py360convert.e2c(imgIn, face_w=512, mode='bilinear', cube_format='horizon')

print(img_flat.shape)
plt.imshow(img_flat)
plt.show()
imgIn_gpu = cuda.to_device (img_flat)
imgOut_gpu = cuda.device_array(shape=(1024, 2048, 3), dtype=np.uint8)
imgOut_cpu = np.zeros(shape=(1024, 2048, 3), dtype=np.uint8)

project[(64, 128), (16, 16)](imgOut_gpu, imgIn_gpu, 1024, 2048, 512)
imgOut_gpu.copy_to_host(imgOut_cpu)
cuda.synchronize()

plt.imshow(imgOut_cpu)
plt.show()