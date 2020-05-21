import numba.cuda as cuda
import numpy as np
import math
import cv2
from frame import frame_to_horizon, frame_to_list
import time
import matplotlib.pyplot as plt

video_file_name = 'test_video.mp4'
f2h = frame_to_horizon((1920, 3720, 3))

cap = cv2.VideoCapture(video_file_name)
start_time = time.time()
counter = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize (frame, (3840, 1920))
    f2h.render(frame)

    # cv2.imshow('test', imgOut_cpu_list[0])
    counter += 1
    if (time.time() - start_time) > 1:
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
cap.release()
cv2.destroyAllWindows()

