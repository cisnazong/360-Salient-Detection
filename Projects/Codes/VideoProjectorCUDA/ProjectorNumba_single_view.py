import numba.cuda as cuda
import numpy as np
import math
import cv2
from utils.ImageProjectorCubicBilinear import project_front, project_right, project_back, project_left, project_up, project_down
from utils.ImageProjectorCubicBilinear import ProjectConfig
import time

video_file_name = 'test_video.mp4'
config = ProjectConfig ((3712, 1920))
function_list = [project_front, project_right, project_back, project_left, project_up, project_down]
stream_list = [cuda.stream () for i in range (6)]
stream0 = cuda.stream ()

imgOut_gpu_list = [cuda.device_array (shape=config.view_shape, dtype=np.uint8, stream=stream_list[i]) for i in
                   range(6)]
imgOut_cpu_list = [np.zeros(shape=config.view_shape, dtype=np.uint8) for i in range(6)]
title_list = ['front', 'right', 'back', 'left', 'up', 'down']

cap = cv2.VideoCapture (video_file_name)
start_time = time.time ()
counter = 0
while (cap.isOpened ()):
	ret, frame = cap.read ()

	# frame = cv2.resize (frame, (3840, 1920))
	imgIn_gpu = cuda.to_device (frame, stream=stream0)

	function_list[0][config.grid_dim, config.block_dim, stream_list[0]] (imgOut_gpu_list[0], imgIn_gpu)
	imgOut_gpu_list[0].copy_to_host(imgOut_cpu_list[0], stream=stream_list[0])
	cuda.synchronize ()

	cv2.imshow('test', imgOut_cpu_list[0])
	if cv2.waitKey (1) & 0xFF == ord ('q'):
		break
	counter += 1
	if (time.time () - start_time) > 1:
		print ("FPS: ", counter / (time.time () - start_time))
		counter = 0
		start_time = time.time ()
cap.release ()
cv2.destroyAllWindows ()
# 优化思路：
# 1. 弃用python这种低效率语言
# 2. 显示和计算异步进行