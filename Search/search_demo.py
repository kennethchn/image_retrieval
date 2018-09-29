#!/usr/bin/env python
#-*- coding:utf-8 -*-

import time
import numpy as np
import cv2

import cnn_search
import feature_extract
import feature_match
from RWoperation import rwOperation 

name_list , feature_train = cnn_search.load_feature()

image_path = '../datafolder/demo.jpg'

net = cnn_search.load_net()

sf = cv2.BFMatcher()
svm_clf = feature_match.ana_svm()

image_feature_path = '../datafolder/test/src_image_feature_path.path'
feature_path_dict = rwOperation.read_dict( image_feature_path )

surf = feature_extract.create_detector('surf')

st_time = time.time()
one_feature = cnn_search.one_extract_feature( net, image_path )
print( one_feature )
print( type( one_feature ))
one_feature = one_feature[np.newaxis, :]

res = sf.knnMatch( one_feature, feature_train, k = 10 )
print( 'spend time :', time.time() - st_time )
#print(res )
dest_name_list = []
for idx in res[0]:
	dest_name_list.append( name_list[idx.trainIdx] )
	
read_image_st = time.time()
img = cv2.imread(image_path )
img = cv2.resize(img, (360,360))
print('read image time:', time.time() - read_image_st )

kp, des = feature_extract.detect( surf, img)

f_train_path = []
for nameIdx in dest_name_list:
	f_train_path.append(feature_path_dict[nameIdx])

f_train_des = []

for pathIdx in f_train_path:
	img_name, img_kp, img_des = rwOperation.read_feature( pathIdx )
	 
	r_good_match, good_match =feature_match.match(kp, des.astype(np.float32),img_kp, img_des.astype(np.float32), 0.75, 'bf')
	if svm_clf.predict([[float( len(r_good_match)), float( len(good_match))]]):
		print( img_name )

print( 'all time:', time.time() - st_time )




