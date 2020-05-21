from VideoProjectorGPU.frame import frame_to_horizon
from frame import frame_from_horizon
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sample = './sample/001.png'
imgIn = np.array(Image.open(sample))
plt.imshow(imgIn)
plt.show()

f2h = frame_to_horizon(imgIn.shape)
imgO1 = f2h.render(imgIn)
plt.imshow(imgO1)
plt.show()


ffh = frame_from_horizon(imgO1.shape)
imgO2 = ffh.render(imgO1)
plt.imshow(imgO2)
plt.show()