import numba.cuda as cuda
import numpy as np
import math
import cv2
from utils.ImageProjectorCubicBilinear import project_front, project_right, project_back, project_left, project_up, \
	project_down
from utils import ProjectConfig
import time

# filename
video_file_name = 'test_video.mp4'

# generate config with the shape of frame
config = ProjectConfig ((3840, 1920))

# define project functions, CUDA streams, input and output placeholders
function_list = [project_front, project_right, project_back, project_left, project_up, project_down]
stream_list = [cuda.stream () for i in range (6)]
stream0 = cuda.stream ()
imgOut_gpu_list = [cuda.device_array (shape=config.view_shape, dtype=np.uint8, stream=stream_list[i]) for i in
                   range (6)]
imgOut_cpu_list = [np.zeros (shape=config.view_shape, dtype=np.uint8) for i in range (6)]
title_list: list = ['front', 'right', 'back', 'left', 'up', 'down']

# Capture the image
cap = cv2.VideoCapture (video_file_name)
start_time = time.time ()
counter = 0
while (cap.isOpened ()):
	ret, frame = cap.read ()
	frame = cv2.resize (frame, (3840, 1920))
	imgIn_gpu = cuda.to_device (frame, stream=stream0)
	# stream0.synchronize()
	for i in range (6):
		function_list[i][config.grid_dim, config.block_dim, stream_list[i]] (imgOut_gpu_list[i], imgIn_gpu)
		imgOut_gpu_list[i].copy_to_host (imgOut_cpu_list[i], stream=stream_list[i])
	cuda.synchronize ()
	# for i in range(6):
	# 	stream_list[i].synchronize()

	# cv2.imshow('front', imgOut_cpu_list[0])
	# for i in range (6):
	# 	cv2.imshow (title_list[i], imgOut_gpu_list[i].copy_to_host (stream=stream_list[i]))
	if cv2.waitKey (1) & 0xFF == ord ('q'):
		break
	counter += 1
	if (time.time () - start_time) > 1:
		print ("FPS: ", counter / (time.time () - start_time))
		counter = 0
		start_time = time.time ()
cap.release ()
cv2.destroyAllWindows ()
