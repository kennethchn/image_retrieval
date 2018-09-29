# coding: utf-8
from __future__ import print_function
import os
import cv2
import time
import ConfigParser

import image_path
from IOoperate import rwOperate as rw 

print('cv version: ', cv2.__version__)
config = ConfigParser.ConfigParser()
config.read('./FeatureExtract/fe_config.ini')

def create_detector( detector='surf' ):
    if detector.startswith('si'):
        print( "sift detector......")
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        print( "surf detector......")
        sift = cv2.xfeatures2d.SURF_create()
    return sift

def detect(sift, image, mask = None):
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(image, mask )
    return kp, des

def save_feature( proto_path, save_root_path ):
    # save feature proto and feature path proto
    #proto_path : image path dict, proto file
    # save_root_path : save root folder, the subfolder will be create 

    if not os.path.exists(proto_path):
        print('ErrorMessage:', proto_path, ' is not exisit!!!')
        return -1

    if not os.path.isdir( save_root_path ):
        os.makedirs( save_root_path)

    image_feature_folder = os.path.join( save_root_path, 'image_feature_folder')
    if os.path.isdir( image_feature_folder ):
        print('Warnning:', image_feature_folder, ' is exisit, please change a new path')
    else:
        os.makedirs(image_feature_folder)
    
    image_path_dict = image_path.read_image_path( proto_path )

    surf_detector = create_detector()

    
    image_feature_path_file = os.path.join( save_root_path, 'src_image_feature_path.path')

    if not os.path.exists(image_feature_path_file):
        image_feature_path_dict = dict()
    else:
        image_feature_path_dict = rw.read_dict(image_feature_path_file)
    
    image_idx = int( config.get('MetaData', 'image_idx') ) #从中断处加载,继续处理
    save_interval = int( config.get('MetaData', 'save_interval') )

    for k, img_path_key in enumerate(image_path_dict.keys()[image_idx:]):
       
        try:
            image_idx += 1
            img_path = image_path_dict[img_path_key]
            img = cv2.imread( img_path ) 
            img = cv2.resize(img, (360,360))

        except:
            print('wrong read!')
            continue
        kp, des = detect( surf_detector, img)
        if not kp:
            continue
        _, tmpfilename = os.path.split(img_path)
        filename, _ = os.path.splitext( tmpfilename )

        one_image_feature_path = os.path.join( image_feature_folder, filename + '.surf')
        rw.save_feature( tmpfilename, kp, des, one_image_feature_path )
        image_feature_path_dict[img_path_key] =  one_image_feature_path
        if (not k%save_interval ) or image_idx == len( image_path_dict.keys()):
            rw.save_dict(image_feature_path_dict, image_feature_path_file)
        
            config.set('MetaData', 'image_idx', str(image_idx - 1))
            with open('config.ini','wb') as configfile:
                config.write(configfile)



     



