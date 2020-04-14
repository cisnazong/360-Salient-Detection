import cv2
from utils import *
import numpy as np
import multiprocessing


def Convert_video_cubicmap(file_name: str, T: ImageProjectorTarget, f, title='untitled'):
	cap = cv2.VideoCapture (file_name)
	while (cap.isOpened ()):
		ret, frame = cap.read ()
		# rgb = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
		# output = image_projector.project (rgb)
		frame = cv2.resize (frame, (3840, 1920))
		f (T, frame)
		cv2.imshow (title, T.Output)
		if cv2.waitKey (1) & 0xFF == ord ('q'):
			break
	cap.release ()
	cv2.destroyAllWindows ()

if __name__ == '__main__':
	file_name = '../test_video.mp4'
	image_projectors: list = [ImageProjectorTarget ((3840, 1920)) for i in range (6)]
	titles: list = ['front', 'right', 'back', 'left', 'up', 'down']
	project_functions: list = [project_front, project_right, project_back, project_left, project_up, project_down]
	Convert_video_cubicmap (file_name, image_projectors[1], project_functions[1], titles[1])