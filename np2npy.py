#!/usr/bin/python
import numpy as np
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2

filename = '/home/cari/caffe_150323/CarStyle_test/car_mean.binaryproto'
blob = caffe_pb2.BlobProto()

data = open(filename, "rb").read() 
blob.ParseFromString(data) 

#nparray =blobproto_to_array(blob) 
arr = np.array(blobproto_to_array(blob) )
out = arr[0]
#f = file("/home/cari/caffe_150323/CarStyle_test/car_mean_npy.npy","wb") 
#np.save(f,nparray) 
outpath = '/home/cari/caffe_150323/CarStyle_test/car_mean_npy.npy'
np.save( outpath , out )

print "OK,Done!"
