from VideoProjectorGPU.frame import frame_to_horizon
from frame import frame_from_horizon
from PIL import Image
import numpy as np
import time

sample = './sample/001.png'
imgIn = np.array(Image.open(sample))
f2h = frame_to_horizon(imgIn.shape)
imgO1 = f2h.render(imgIn)

ffh = frame_from_horizon(imgO1.shape)


cnt = 0
start_time = time.time()

while True:
    imgO2 = ffh.render(imgO1)
    cnt += 1
    if time.time() - start_time > 1:
        print("FPS: ", cnt / (time.time() - start_time))
        cnt = 0
        start_time = time.time()

