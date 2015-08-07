#!/usr/bin/env python
import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2
import pylab
import os
import os.path
import time

caffe_root = '/home/cari/caffe/'
image_root_dir = caffe_root + 'CarStyle_test/Done/yage/yage_val/'

def endWith(s,*endString):
    array = map(s.endswith,endString)
    if True in array:
        return True
    else:
        return False



MODEL_FILE = caffe_root+'CarStyle_test/model_files/car_deploy.prototxt'
PRETRAINED = caffe_root+'CarStyle_test/trained_model/408_crop_iter_2400.caffemodel'
np_mean_file = '/home/cari/caffe/CarStyle_test/car_mean.binaryproto'
npy_mean_file = '/home/cari/caffe/CarStyle_test/car_mean_npy.npy'

blob = caffe_pb2.BlobProto()
blob.ParseFromString(open(np_mean_file, "rb").read())

print blob.num, blob.channels, blob.height, blob.width
means = np.array(blobproto_to_array(blob) )
shape_means = means.reshape(3,150,150)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(npy_mean_file).mean(1).mean(1), channel_swap=(2,1,0),raw_scale=255,
                       image_dims=(133,133))


start_time = time.clock()

for filenames in os.listdir(image_root_dir):
    #print filenames
    if endWith(filenames,'.jpg'):
        prediction = net.predict([caffe.io.load_image(image_root_dir + filenames)])
        print 'ok'
end_time = time.clock()
spend_time = (end_time - start_time)

print("The function run time is : %.03f seconds" %(spend_time))



