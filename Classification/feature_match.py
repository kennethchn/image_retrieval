# coding:utf-8
from __future__ import print_function
import os
import cv2
import time
import numpy as np
import shutil
import ConfigParser
from IOoperate import rwOperate as  rw

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import tensorflow as tf

current_path = os.path.dirname(__file__)
config = ConfigParser.ConfigParser()
config_path = os.path.join(current_path, 'cls_config.ini')
config.read(config_path)


#是否从断点开始
test_flag = True


def get_ana_data():
    #分析数据得到最优阈值
    with open(os.path.join(current_path, 'data_ana_foranaly.txt'), 'r') as f:
        data = f.readlines()

    ana_data = []
    for one_data in data:
        if not one_data:
            continue
        a = one_data.strip().split('\t')

        ana_data.append( [float(a[0]), float(a[1]), int(a[5])] )
    return ana_data

def ana_svm():

    data = np.array( get_ana_data())
    data_2d = data[:,0:2]
    label = data[:,2].astype(np.float64)
    
    svm_clf = Pipeline(( ("scaler", StandardScaler()),
                        ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))

    svm_clf.fit( data_2d, label )
    return svm_clf

def create_matcher():
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5 )
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher( index_params, search_params)
    return flann

def match(kp1, des1, kp2, des2, th = 0.75, method = 'bfmatcher'):
    good_matchs = []
    if method.startswith('bf'):
        #交叉匹配
        bfm = cv2.BFMatcher_create( cv2.NORM_L2, True )   
        good_matchs = bfm.knnMatch( des1, des2, 1 )
        good_matchs = [m for m in good_matchs if m != []]

    else:
        flann = create_matcher() 
        matches = flann.knnMatch( des1, des2, k = 2)
        good_matchs = [[m] for m, n in matches if m.distance < th * n.distance]

    MIN_MATCH_COUNT = 10
    ransanc_good_match = []
    if len(good_matchs) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good_matchs ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good_matchs ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    else:
        print('good_matchs is less than MIN_MATCH_COUNT!')
        return ransanc_good_match, good_matchs
        
    for i in range(len(mask)):
        if mask[i]:
            ransanc_good_match.append(good_matchs[i])
    return ransanc_good_match, good_matchs

def read_des_feature_by_path(feature_path):
    '''
        通过key读取特征数据，该方式占用内存少，但是耗时长
    '''
    img_name, img_kp, img_des = rw.read_feature( feature_path )
    return img_name, img_kp, img_des
	
def read_des_feature(feature_path_dict):
    #from feature path dict to get des dict
    des_dict = dict()
    kp_dict = dict()
    for key in feature_path_dict.keys():
        img_name, img_kp, img_des = rw.read_feature(feature_path_dict[key])
        kp_dict[img_name] = img_kp
        des_dict[img_name] = img_des
    return kp_dict, des_dict

# def class_image( imageDir ):
#     #读取原始图像路径， image_path_dict记录原始图像路径，如 ‘a.jpg', '/root/kenneth/a.jpg'格式 
#     src_image_path = os.path.join( imageDir, 'src_image_path.path')
#     if not os.path.exists(src_image_path):
#         print('ErrorMessage:', src_image_path, ' is not exisit!')
#     image_path_dict = rw.read_dict(src_image_path)

#     #获取原始图像特征路径， feature_path_dict记录图像特征路径， 如 ’a.jpg'：‘/root/kenneth/a.surf'
#     src_image_feature_path = os.path.join( imageDir, 'src_image_feature_path.path')
#     if not os.path.exists(src_image_feature_path):
#         print('ErrorMessage:', src_image_feature_path, ' is not exisit!')
#     feature_path_dict = rw.read_dict(src_image_feature_path)

#     #查看image_class_floder 文件夹是否存在，如果不存在创建一个
#     image_class_floder = os.path.join( imageDir, 'image_class_folder' )
#     if not os.path.exists( image_class_floder ):
#         os.mkdir( image_class_floder )
    
# #   获取索引特征字典，如果存在读取，如果不存在创建一个dict，用于后面存储特征索引，
#     iif_save_path = os.path.join( imageDir, 'index_image_feature.path' )
#     if os.path.exists(iif_save_path):
#         index_image_feature_path_dict = rw.read_dict(iif_save_path)
#     else:
#         index_image_feature_path_dict = dict()
    
#     #从特征文件夹中获取所以的kp和des数据，并以字典方式保存，key是图像名称
#     kp_dict, des_dict = read_des_feature(feature_path_dict)

#     #获取所以特征的key，即所以特征的图像名称list
#     key_list = des_dict.keys()

#     #测试
#     if test_flag:
#         print(key_list)
#         data_ana = []  #测试时候使用 

#     svm_clf = ana_svm()
# #    res = svm_clf.predict( [[5.5, 1.7]] )

#     #将图像分类、并拷贝一份到image_class_floder文件夹中，其中的子文件夹是以first_key的名称命名，即用来比对的图像名称
#     while key_list:
#         first_key = key_list.pop(0)
#         first_kp = kp_dict[first_key]
#         first_des = des_dict[first_key].astype(np.float32)

#         #将索引特征路径加入index_image_feature.path中
#         index_image_feature_path_dict[first_key] = feature_path_dict[first_key]

#         #创建分类图像文件夹，以第一图像名称命名文件夹，并将第一图像拷贝到该文件夹中，如果已经存在则报错
#         img_name, _ = os.path.splitext(first_key)
#         cfd_folder_name = os.path.join( image_class_floder, img_name )
#         if not os.path.exists(cfd_folder_name):
#             os.mkdir(cfd_folder_name)
#             shutil.copyfile(image_path_dict[first_key], os.path.join(cfd_folder_name, first_key))
#         else:
#             print('Warnning:', cfd_folder_name, 'is already exisit!')
        
#         for i, key in enumerate(key_list):

#             ransanc_gmatch, gmatch = match(first_kp, first_des,kp_dict[key], des_dict[key].astype(np.float32))

# #            if len(ransanc_gmatch) > 30 and len(ransanc_gmatch)/float(len(gmatch)) > 0.4:
#             if svm_clf.predict([[float( len(ransanc_gmatch)), float( len(gmatch))]]):
#                 shutil.copyfile( image_path_dict[key], os.path.join(cfd_folder_name, key) )

#                 #测试
#                 if test_flag:
#                     data_ana.append([len(ransanc_gmatch), len(gmatch), len(ransanc_gmatch)/float(len(gmatch)), first_key, key_list[i], 1])
                    
#                 key_list[i] = 0
#             else:
#                 #测试
#                 if test_flag:
#                     data_ana.append([len(ransanc_gmatch), len(gmatch), len(ransanc_gmatch)/float(len(gmatch)), first_key, key_list[i], 0])
#         key_list = [elem for elem in key_list if elem is not 0]
    
#     #将索引特征路径dict保存
#     rw.save_dict(index_image_feature_path_dict, iif_save_path)

#     #测试时候保存数据，用于分析
#     if test_flag:
        
#         with open('data_ana.txt', 'w') as f:
#             for one_data in data_ana:
#                 txt = ''
#                 for elem in one_data:
#                     if isinstance(elem, int) or isinstance( elem, float):
#                         elem = str(elem)
#                     txt += elem +'\t'
#                 f.writelines(txt+'\n')

def class_image_slow( imageDir ):
    #特征读取一个一个读取，运行速度相对慢些，但是占用电脑资源更少
    #读取原始图像路径， image_path_dict记录原始图像路径，如 ‘a.jpg', '/root/kenneth/a.jpg'格式 
    src_image_path = os.path.join( imageDir, 'src_image_path.path')
    if not os.path.exists(src_image_path):
        print('ErrorMessage:', src_image_path, 'the image path protofile is not exisit!')
        return -1
    image_path_dict = rw.read_dict(src_image_path)

    #获取原始图像特征路径， feature_path_dict记录图像特征路径， 如 ’a.jpg'：‘/root/kenneth/a.surf'
    src_image_feature_path = os.path.join( imageDir, 'src_image_feature_path.path')
    if not os.path.exists(src_image_feature_path):
        print('ErrorMessage:', src_image_feature_path, 'the image feature save path protofile is not exisit!')
        return -1
    feature_path_dict = rw.read_dict(src_image_feature_path)

    #查看image_class_floder 文件夹是否存在，如果不存在创建一个
    image_class_floder = os.path.join( imageDir, 'image_class_folder' )
    if not os.path.exists( image_class_floder ):
        os.mkdir( image_class_floder )
    
#   获取索引特征字典，如果存在读取，如果不存在创建一个dict，用于后面存储特征索引，
    iif_save_path = os.path.join( imageDir, 'index_image_feature.path' )

    index_image_feature_path_dict = dict()
    
    #从特征文件夹中获取所以的kp和des数据，并以字典方式保存，key是图像名称
#    kp_dict, des_dict = read_des_feature(feature_path_dict)

    key_list = list()
    
    break_from_flag = int( config.get('MetaData', 'break_from_flag'))
    if break_from_flag: 
        key_list_path = config.get('MetaData', 'key_list_path')
        index_image_feature_path_dict = rw.read_dict(iif_save_path)

        with open(key_list_path, 'r') as f:
            temp_key = f.readline().strip()
            while temp_key:
                key_list.append(temp_key)
                temp_key = f.readline().strip()
        break_from_flag = 0
    else:
        #获取所以特征的key，即所以特征的图像名称list
        key_list = feature_path_dict.keys()
        key_list_path = os.path.join( imageDir, 'key_list.txt' )

    #测试
    if test_flag:
        print(key_list)
        data_ana = []  #测试时候使用 

    svm_clf = ana_svm()
#    res = svm_clf.predict( [[5.5, 1.7]] )


    #将图像分类、并拷贝一份到image_class_floder文件夹中，其中的子文件夹是以first_key的名称命名，即用来比对的图像名称
    while key_list:

        first_key = key_list.pop(0)
        _, first_kp, first_des = read_des_feature_by_path(feature_path_dict[first_key])
#        first_kp = kp_dict[first_key]
        first_des = first_des.astype(np.float32)

        #将索引特征路径加入index_image_feature.path中
        index_image_feature_path_dict[first_key] = feature_path_dict[first_key]

        #创建分类图像文件夹，以第一图像名称命名文件夹，并将第一图像拷贝到该文件夹中，如果已经存在则报错
        img_name, _ = os.path.splitext(first_key)
        cfd_folder_name = os.path.join( image_class_floder, img_name )
        if not os.path.exists(cfd_folder_name):
            os.mkdir(cfd_folder_name)
            shutil.copyfile(image_path_dict[first_key], os.path.join(cfd_folder_name, first_key))
        else:
            print('warnning:', cfd_folder_name, 'is already exisit!')
        
        for i, key in enumerate(key_list):
            _, one_kp, one_des = read_des_feature_by_path(feature_path_dict[key])
            ransanc_gmatch, gmatch = match(first_kp, first_des,one_kp, one_des.astype(np.float32))

#            if len(ransanc_gmatch) > 30 and len(ransanc_gmatch)/float(len(gmatch)) > 0.4:
            if svm_clf.predict([[float( len(ransanc_gmatch)), float( len(gmatch))]]):
                shutil.copyfile( image_path_dict[key], os.path.join(cfd_folder_name, key) )

                #测试
                if test_flag:
                    data_ana.append([len(ransanc_gmatch), len(gmatch), len(ransanc_gmatch)/float(len(gmatch)), first_key, key_list[i], 1])
                    
                key_list[i] = 0
            else:
                #测试
                if test_flag:
                    data_ana.append([len(ransanc_gmatch), len(gmatch), len(ransanc_gmatch)/float(len(gmatch)), first_key, key_list[i], 0])
        key_list = [elem for elem in key_list if elem is not 0]


        rw.save_dict(index_image_feature_path_dict, iif_save_path)
            
        with open(key_list_path, 'w') as f:
            for one_key in key_list:
                f.writelines(one_key+'\n')
            

    #测试时候保存数据，用于分析
    if test_flag:
        
        with open(os.path.join(current_path,'data_ana.txt'), 'w') as f:
            for one_data in data_ana:
                txt = ''
                for elem in one_data:
                    if isinstance(elem, int) or isinstance( elem, float):
                        elem = str(elem)
                    txt += elem +'\t'
                f.writelines(txt+'\n')

def demo():
    src_image_feature_path = 'wine_images/src_image_feature_path.path'

    feature_path_dict = rw.read_dict(src_image_feature_path)

    key_list = feature_path_dict.keys()


    img_name, img_kp, img_des = rw.read_feature(feature_path_dict[key_list[0]].strip())
    print( img_name )
    img_name2, img_kp2, img_des2 = rw.read_feature(feature_path_dict[key_list[100]].strip())
    print( img_name2)
    r_good_match, good_match = match(img_kp, img_des.astype(np.float32),img_kp2, img_des2.astype(np.float32), 0.75, 'bf')
    print(len(good_match))

    # MIN_MATCH_COUNT = 10
    # ransanc_good_match = []
    # if len(good_match) > MIN_MATCH_COUNT:
    #     # 获取关键点的坐标
    #     src_pts = np.float32([ img_kp[m[0].queryIdx].pt for m in good_match ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ img_kp2[m[0].trainIdx].pt for m in good_match ]).reshape(-1,1,2)

    #     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    # for i in range(len(mask)):
    #     if mask[i]:
    #         ransanc_good_match.append(good_match[i])
    # print( 'ransanc_good_match:', len( ransanc_good_match))

    print( len( r_good_match ) )

    print( img_des.shape )


if __name__ == '__main__':
    import time
    st = time.time()
    class_image_slow('test')
    print('class_image_slow spendtime:', time.time() - st )

#    class_image('test')
#    demo()
