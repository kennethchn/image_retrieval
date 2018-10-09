import os
from VLAD import VLADtoproto
#test1
path = '/home/kenneth/gitstore/datafolder/img1000_feature'
train_feature = VLADtoproto.getDescriptors(path)
VLADtoproto.kMeansDictionary(train_feature, 100, path)
print( 'cluster finished!')
res = VLADtoproto.read_kmean_result(path)

VLADtoproto.save_VLAD_to_proto(path, res)
print( 'save vlad feature finished!')
des_dict = VLADtoproto.load_VLAD_from_proto(os.path.join(path, 'descriptors_dict.vlad'))


##cluster result save and load example
#joblib.dump( res, 'surf_cluster.pkl')
#km = joblib.load('surf_cluster.pkl')
