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
import ConfigParser
from IOoperate import rwOperate

config = ConfigParser.ConfigParser()
current_path = os.path.dirname(os.path.abspath(__file__))
config.read(os.path.join(current_path,'cnn_config.ini'))

caffe.set_mode_gpu()
caffe.set_device(0)

def load_net():
	weight = config.get('cnn', 'weight')
	deploy_file = config.get('cnn', 'deploy_file')

	net = caffe.Net( deploy_file, weight, caffe.TEST )
	return net

def one_extract_feature(net, image_path):
	#抽取一张图像的特征
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	net.blobs['data'].reshape(1,3,224,224)
	
	try:
		simage = caffe.io.load_image( image_path )
	except:
		print('read image error, error path:', image_path)
		return -1

	transformered_image = transformer.preprocess( 'data', simage )
	net.blobs['data'].data[...] = transformered_image 
	st0 = time.time()
	net.forward()
	print( time.time() - st0 )
	one_feature = np.squeeze( net.blobs['pool5'].data)
	return one_feature 

def extract_feature( image_path):
	#抽取文件夹下所有图像的特征 
	net = load_net()
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	net.blobs['data'].reshape(1,3,224,224)
	
	features_dict = {} 
	image_list = os.listdir(image_path)
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
		print('extract_feature time:', time.time() - st0 )
		features_dict[img] = copy.deepcopy( np.squeeze( net.blobs['pool5'].data))
	return features_dict 

def load_feature( feature_proto_path ):
	fd = rwOperate.read_dict_des(feature_proto_path )
	feature_list = []
	for key in fd.keys():
		feature_list.append( fd[key] )
	feature_train = np.array( feature_list, dtype=np.float32 )
	return fd.keys(), feature_train

def search_demo(feature, feature_train):
	st_time = time.time()
	sf = cv2.BFMatcher()
	res = sf.knnMatch( feature, feature_train, k = 1000 )
	print( 'search time:', time.time() - st_time )
	return res 


