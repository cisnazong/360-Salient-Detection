import cv2
from utils import ImageProjectorTarget
from utils import project
import numpy as np
import multiprocessing
image_projector = ImageProjectorTarget (1920,3840)
output: list = []
cap = cv2.VideoCapture ('test_video.mp4')
while (cap.isOpened ()):
	ret, frame = cap.read ()
	# rgb = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
	# output = image_projector.project (rgb)
	frame = cv2.resize(frame, (3840, 1920))
	output = project (image_projector, frame)
	cv2.imshow ('front', output)
	# cv2.imshow ('right', output[1])
	# cv2.imshow ('back', output[2])
	# cv2.imshow ('left', output[3])
	# cv2.imshow ('up', output[4])
	# cv2.imshow ('down', output[5])
	if cv2.waitKey (1) & 0xFF == ord ('q'):
		break
cap.release ()
cv2.destroyAllWindows ()
