#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function
import time
import sys
sys.path.append('/home/kenneth/gitstore/image_retrieval')
from IOoperate import rwOperate
from FeatureExtract import image_path, feature_extract, cnn_feature_extract


def image_path_demo():
    imageDir = '/home/kenneth/gitstore/datafolder/img2000'
    save_path = '/home/kenneth/gitstore/datafolder/img2000_feature/src_image_path.path'
    image_path.save_image_path( imageDir, save_path)

    image_path_dict = image_path.read_image_path( save_path )
    print( image_path_dict)

def feature_extract_demo():
    proto_path = '/home/kenneth/gitstore/datafolder/img2000_feature/src_image_path.path'
    save_root_path = '/home/kenneth/gitstore/datafolder/img2000_feature'
    feature_extract.save_feature( proto_path, save_root_path )
def cnn_feature_extract_demo():

#	image_path = '../datafolder/CompanyStandardLabel'
#	image_path = './TestLabelImage'
	image_path = '/home/kenneth/gitstore/datafolder/img2000'
	net = cnn_feature_extract.load_net()
 	feature_dict = cnn_feature_extract.extract_feature(net, image_path )	
#	print( feature_dict )
	rwOperate.save_dict_des( feature_dict, '/home/kenneth/gitstore/datafolder/img2000_feature/image_cnn_dict.feature')
	fd = rwOperate.read_dict_des( '/home/kenneth/gitstore/datafolder/img2000_feature/image_cnn_dict.feature')
	print( fd )
#	print( fd.keys() )
#	for key in fd.keys():
#		print( fd[key][0] )
#		print( len(fd[key]) )
#		print( type( fd[key]))	

def test_speed():
    image_path = '/home/kenneth/gitstore/datafolder/img1000'
    net = cnn_feature_extract.load_net()
    feature_dict = cnn_feature_extract.extract_feature(net, image_path )
    a = []
    st_time = time.time()
    for key in feature_dict.keys():
        a.append(feature_dict[key])
    print( '-----------------------------------------------------------------------------------')
    print('spend time:', time.time() - st_time )
    return a

if __name__ == '__main__':
    image_path_demo()
    feature_extract_demo()
    cnn_feature_extract_demo()
#    a = test_speed()
#    print(a)