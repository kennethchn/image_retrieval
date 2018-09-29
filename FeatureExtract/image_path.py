#-*- coding:utf-8 -*-
from __future__ import print_function

import os
#from image_feature import rw
from IOoperate import rwOperate as rw 

#遍历文件夹   
def get_path_dict(imageDir):
    #遍历根目录
    path_dict = dict()
    for root,_,files in os.walk(imageDir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif']: #bmp、gif、jpg、pic、png、tif               
                filepath = os.path.join(root,file)
                path_dict[file] = filepath
    return path_dict 

def save_image_path(imageDir, save_path): 
    #imageDir : src image path
    #save_path : the proto file save path
    if not os.path.exists(imageDir):
        print('ErrorMessage:', imageDir, ' is not exisit!')
        return -1
    
    path_dict = get_path_dict( imageDir )
    rw.save_dict( path_dict, save_path )
    

def read_image_path( protofile_path ):
    #read data from the protofile
    #protofile_path: the protofile path
    
    if not os.path.exists( protofile_path):
        print('ErrorMessage:', protofile_path, ' is not exisit !!!')
        return -1
    image_path_dict = rw.read_dict( protofile_path )

    return image_path_dict


