#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import numpy as np
from IOoperate import rwOperate

def demo1():
    #save and load feature, save_feature function and read_feature function 
    kp_aa = cv2.KeyPoint()
    kps_d = [kp_aa,kp_aa]
    des_a = np.array( [[1,2,3], [2,3,4], [5,6,7]])
    rwOperate.save_feature( 'image1.jpg', kps_d, des_a, './test/data.surf' ) 
    img_name, img_kp, img_des = rwOperate.read_feature('./test/data.surf')

    print( img_name )
    print( img_kp, type( img_kp ))
    print( img_des, type( img_des ))

def demo2():
    # test for save_dict and read_dict function 
    pdict = dict()
    pdict['a'] = 'b'
    pdict[''] = 'd'

    save_dict_path = './test/key_value.path'
    rwOperate.save_dict( pdict, save_dict_path )
    dd = rwOperate.read_dict( save_dict_path )
    print( dd )

def demo3():
    #test for save_dict_des and read_dict_des function which only save the des without keypoint
    #these two fucntion can use to save deeplearn feature or vald feature
    dicte = {}
    dicte['123'] = [1,2,3,4]
    dicte['345'] = [12,3,4,56,7,9]

    rwOperate.save_dict_des(dicte, './test/feature.cnn_feature')
    a = rwOperate.read_dict_des('./test/feature.cnn_feature')
    for key in a.keys():
        print(key)
        print( type( a[key]))
    print( a[key] )

if __name__ == '__main__':
    demo1()
    demo2()
    demo3()
