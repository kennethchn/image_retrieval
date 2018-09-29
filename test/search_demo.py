#!/usr/bin/env python
#-*-coding:utf-8-*-
from Search import 

def demo1():
	image_path = './test/com'
	image_path = './image'
	net = load_net()
	feature_dict = extract_feature(net, image_path )	
	rwOperate.save_dict_des( feature_dict, 'image_cnn_dict.feature')

def demo2():
	fd = rwOperate.read_dict_des( 'image_cnn_dict.feature')
	print( fd.keys() )
	for key in fd.keys():
		print( fd[key][0] )
		print( len(fd[key]) )
		print( type( fd[key]))	

def demo3():
	image_path = './image/demo.jpg'
	afeature = one_extract_feature( image_path)
	afeature = afeature[np.newaxis, :]		
	print( afeature.shape, afeature.dtype )
	
	name_list, feature_train = load_feature()
	print(feature_train.shape, feature_train.dtype )
	match_result = search_demo( afeature, feature_train )
	for mr in match_result:
		print( name_list[mr[0].trainIdx])
	
def demo4():
#	image_path = '/root/caffework/TestLabelImage'
#	image_path = './image'
	image_path = '../datafolder/ceshi0109'
	search_feature = extract_feature( image_path )
	name_list, feature_train = load_feature()
	
	print( name_list[0:3])
	print( feature_train[0:3])	

	record_result = []
	for key in search_feature.keys():
		afeature = search_feature[key]
		afeature = afeature[np.newaxis, :]
#		print( afeature )
		match_result = search_demo( afeature, feature_train )

		temp_record = []
		temp_record.append( key )
		for mr in  match_result[0] :
			temp_record.append( name_list[mr.trainIdx] )
		record_result.append( copy.deepcopy(temp_record) )	
#	print( record_result )
	np.save( '../datafolder/record.npy', record_result )


if __name__ == '__main__':
    demo1()