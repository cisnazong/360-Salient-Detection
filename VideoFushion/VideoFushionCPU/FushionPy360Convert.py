import py360convert
import glob
import os
import PIL
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tensorboardX import SummaryWriter
import tensorflow as tf

sample_dir = './sample'
sample_ext = '*.png'
sample_list = glob.glob(os.path.join(sample_dir, sample_ext))
output_dir = './output'
log_dir = './log'
writer = SummaryWriter(log_dir)

count = 0
for sample in sample_list:
	count += 1
	file_name = sample.split('/')[-1]

	# read input image with PIL
	imgIn = np.array(Image.open(sample))

	# read input image with Tensorflow
	img_file_1 = tf.io.read_file(sample)
	im1 = tf.io.decode_image(img_file_1, channels=3)

	# Plot and log input image
	plt.title(file_name + '_original')
	plt.imshow(imgIn)
	plt.show()
	writer.add_image(file_name + '_original', np.transpose(imgIn, (2, 0, 1)))

	# convert, compress, fushion
	h, w = imgIn.shape[0:2]
	img_dict = py360convert.e2c(imgIn, face_w=512, mode='bilinear', cube_format='dict')
	img_dict['U'] = np.array(Image.fromarray(img_dict['U']).filter(ImageFilter.BLUR))
	img_dict['D'] = np.array(Image.fromarray(img_dict['D']).filter(ImageFilter.BLUR))
	img_h = py360convert.cube_dict2h(img_dict)
	imgOut = py360convert.c2e(img_h, 1024, 2048, cube_format='horizon')
	imgOut = imgOut.astype(np.uint8)

	# Plot output image
	plt.title(file_name + '_compressed')
	plt.imshow(imgOut)
	plt.show()

	# save output image to disk with PIL
	imgOut = Image.fromarray(imgOut)
	imgOut.save(os.path.join(output_dir, file_name), 'png')

	# read output image with TensorFlow
	img_file_2 = tf.io.read_file(os.path.join(output_dir, file_name))
	im2 = tf.io.decode_image(img_file_2, channels=3)

	# calculate SSIM with TensorFLow
	ssim_loss = 1 - float(tf.image.ssim(im1, im2, max_val=255, filter_size=11,
	                                    filter_sigma=1.5, k1=0.01, k2=0.03))

	writer.add_scalar('ssim_loss', ssim_loss, count)
	writer.add_image(file_name + '_compressed_' + str(float(ssim_loss)), np.transpose(imgOut, (2, 0, 1)))

	writer.add_scalar('space saved',
	                  os.path.getsize(sample) - os.path.getsize(os.path.join(output_dir, file_name)),
	                  count)

	writer.add_scalar('space saved ratio',
	                  (os.path.getsize(sample) - os.path.getsize(os.path.join(output_dir, file_name))) / os.path.getsize(sample),
	                  count)

	del ssim_loss, im1, im2, img_file_1, img_file_2
writer.close()
