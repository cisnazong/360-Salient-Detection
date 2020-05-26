from VideoProjectorGPU.frame import frame_to_horizon
from frame import frame_from_horizon
from PIL import Image
import numpy as np
import time
import numba.cuda as cuda
from utils import copy

sample = './sample/001.png'
imgIn = np.array(Image.open(sample))
imgIn_gpu = cuda.to_device(imgIn)
imgOut_gpu = cuda.to_device(np.zeros(shape=imgIn.shape))
cnt = 0
start_time = time.time()

while True:
    copy[(64,128,1),(16,16)](imgOut_gpu, imgIn_gpu)
    cnt += 1
    if time.time() - start_time > 1:
        print("FPS: ", cnt / (time.time() - start_time))
        cnt = 0
        start_time = time.time()

# 理论最大FPS