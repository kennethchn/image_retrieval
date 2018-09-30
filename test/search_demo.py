#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import copy
import numpy as np
import sys
sys.path.append('..')
from Search import base_search
from IOoperate import rwOperate
from FeatureExtract import cnn_feature_extract as cfe
def demo1():
	image_path = './test/com'
	image_path = './image'
	net = base_search.load_net()
	feature_dict = cfe.extract_feature(net, image_path )	
	rwOperate.save_dict_des( feature_dict, 'image_cnn_dict.feature')

def demo2():
#	fd = rwOperate.read_dict_des( os.path.join('/home/kenneth/gitstore/datafolder/com', 'image_cnn_dict.feature') )
	fd = rwOperate.read_dict_des( os.path.join('/home/kenneth/gitstore/datafolder/com', 'descriptors_dict.vlad') )
	print( fd.keys() )
	for key in fd.keys():
		print( fd[key][0] )
		print( len(fd[key]) )
		print( type( fd[key]))	

def demo3():
	#cnn search
	net = base_search.load_net()
	image_path = '../datafolder/demo.jpg'
	afeature = base_search.one_extract_cnn_feature(net, image_path)
	afeature = afeature[np.newaxis, :]		
	print( afeature.shape, afeature.dtype )
	
	name_list, feature_train = base_search.load_feature('../datafolder/com/image_cnn_dict.feature')
	print(feature_train.shape, feature_train.dtype )
	match_result = base_search.search( afeature, feature_train )
	for mr in match_result:
		print( name_list[mr[0].trainIdx])
	
def demo4():
#	image_path = '/root/caffework/TestLabelImage'
#	image_path = './image'
	image_path = '/home/kenneth/gitstore/datafolder/original_wine_images'
	search_feature = base_search.extract_cnn_feature( image_path )
	name_list, feature_train = base_search.load_feature('/home/kenneth/gitstore/datafolder/com/image_cnn_dict.feature')
	
	print( name_list[0:3])
	print( feature_train[0:3])	

	record_result = []
	for key in search_feature.keys():
		afeature = search_feature[key]
		afeature = afeature[np.newaxis, :]
#		print( afeature )
		match_result = base_search.search( afeature, feature_train )

		temp_record = []
		temp_record.append( key )
		for mr in  match_result[0] :
			temp_record.append( name_list[mr.trainIdx] )
		record_result.append( copy.deepcopy(temp_record) )	
#	print( record_result )
	np.save( '../../datafolder/record.npy', record_result )
def demo5():
	v = base_search.one_extract_vlad_feature('/home/kenneth/gitstore/datafolder/demo.jpg')
	print( v )

def demo6():
	#vlad search
	one_feature = base_search.one_extract_vlad_feature('/home/kenneth/gitstore/datafolder/demo.jpg')
	name_list, feature_train = base_search.load_feature( os.path.join('/home/kenneth/gitstore/datafolder/com', 'descriptors_dict.vlad'))
	match_result = base_search.search(one_feature, feature_train)
	for mr in match_result:
		print( name_list[mr[0].trainIdx])


if __name__ == '__main__':
    demo6()
