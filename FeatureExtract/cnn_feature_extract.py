#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function
import os
import sys
import copy
sys.path.append('/root/caffe/python')
sys.path.append('/home/kenneth/ckwork/caffe-master/python')
import time
import cv2
import caffe
import numpy as np

from IOoperate import rwOperate

caffe.set_mode_gpu()
caffe.set_device(0)

def load_net():
	weight = '../datafolder/resnet/ResNet-152-model.caffemodel'
	deploy_file = '../datafolder/resnet/ResNet_152_deploy.prototxt'

	net = caffe.Net( deploy_file, weight, caffe.TEST )
	return net

def extract_feature(net, image_path):
	net = load_net()
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	net.blobs['data'].reshape(1,3,224,224)
	
	features_dict = {} 
	image_list = os.listdir(image_path)
	
	show_flag = 0
	for img in image_list:
		image_full_path = os.path.join( image_path, img )
		try:
			image = caffe.io.load_image( image_full_path )
		except:
			print('read image error, error path:', image_full_path)
			continue
		transformered_image = transformer.preprocess( 'data', image )
		net.blobs['data'].data[...] = transformered_image 
		st0 = time.time()
		net.forward()
		if not (show_flag % 10):
			print( time.time() - st0 )
			show_flag += 1		
		show_flag += 1		

		tmp_feature = copy.deepcopy(np.squeeze( net.blobs['pool5'].data))
#		print( 'tmp_feature:', tmp_feature )
		features_dict[img] = tmp_feature
	rwOperate.save_dict_des( features_dict, './test/com/image_cnn_dict.feature')
	return features_dict 
		
