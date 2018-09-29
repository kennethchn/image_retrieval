#!/usr/bin/env python
#-*-coding:utf-8 -*-

import os
import numpy as np 

from VLAD import VlADlib as VLAD
from IOoperate import rwOperate

import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.externals import joblib
import pickle
import glob
import cv2


def getDescriptors(src_image_feature_path):
    #read image feature for 
    #path = '../datafolder/test'
    image_feature_path = os.path.join(src_image_feature_path, 'src_image_feature_path.path')
    if os.path.exists( image_feature_path ):
        path_dict = rwOperate.read_dict(image_feature_path)
    else:
        print('have no src_image_feature_path.path to read!')

    descriptors=list()

    for key in path_dict.keys():
        one_image_feature_path = path_dict[key]
        _, _, img_des = rwOperate.read_feature(one_image_feature_path)
        descriptors.extend(img_des.tolist())

    descriptors = np.asarray(descriptors)
    return descriptors

    # for imagePath in glob.glob(path+"/*.jpg"):
    #     print(imagePath)
    #     im=cv2.imread(imagePath)
    #     kp,des = functionHandleDescriptor(im)
    #     if len(des)!=0:
    #         descriptors.append(des)
    #         print(len(kp))
        
    # #flatten list       
    # descriptors = list(itertools.chain.from_iterable(descriptors))
    #list to array

# training = a set of descriptors
def  kMeansDictionary(training, k, save_path):

    #K-means algorithm
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    joblib.dump( est, os.path.join( save_path, 'surf_cluster.pkl'))#save cluster result

def read_kmean_result(surf_cluster_path):
    est = joblib.load(os.path.join( surf_cluster_path, 'surf_cluster.pkl') )
    return est

def save_VLAD_to_proto(src_image_feature_path, visualDictionary):
    image_feature_path = os.path.join(src_image_feature_path, 'src_image_feature_path.path')
    if os.path.exists( image_feature_path ):
        path_dict = rwOperate.read_dict(image_feature_path)
    else:
        print('have no src_image_feature_path.path to read!')

    descriptors_dict=dict()

    for key in path_dict.keys():
        try:
            one_image_feature_path = path_dict[key]
            _, _, img_des = rwOperate.read_feature(one_image_feature_path)
            v=VLAD.VLAD(img_des,visualDictionary)
            descriptors_dict[key] = v
        except:
            print('extract vlad feature failure')
            continue
    rwOperate.save_dict_des(descriptors_dict, os.path.join(src_image_feature_path, 'descriptors_dict.vlad'))

def load_VLAD_from_proto(descriptor_dict_path):
    if not os.path.exists( descriptor_dict_path ):
        print('descriptor_dict_path is not exist!')
    descriptors_dict = rwOperate.read_dict_des( descriptor_dict_path)
    return descriptors_dict

if __name__ == '__main__':

    #test1
    path = './test/com'
    train_feature = getDescriptors(path)
    kMeansDictionary(train_feature, 100, path)
    print( 'cluster finished!')
    res = read_kmean_result(path)

    save_VLAD_to_proto(path, res)
    print( 'save vlad feature finished!')
    des_dict = load_VLAD_from_proto(os.path.join(path, 'descriptors_dict.vlad'))


    ##cluster result save and load example
    #joblib.dump( res, 'surf_cluster.pkl')
    #km = joblib.load('surf_cluster.pkl')

