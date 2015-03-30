import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2
import pylab

caffe_root = '/home/cari/caffe_150323/'

MODEL_FILE = caffe_root+'CarStyle_test/model_files/car_deploy.prototxt'
PRETRAINED = caffe_root+'CarStyle_test/trained_model/327_iter_1600.caffemodel'
#IMAGE_FILE = caffe_root +  'CarStyle_test/Images/Fox/Fox_val/Fox0069.jpg'
#outpath = '/home/cari/caffe_150323/CarStyle_test/car_mean_npy.npy'
np_mean_file = '/home/cari/caffe_150323/CarStyle_test/car_mean.binaryproto'
npy_mean_file = '/home/cari/caffe_150323/CarStyle_test/car_mean_npy.npy'
Chhe_file = caffe_root + 'CarStyle_test/Images/Chhe/Chhe_val/Chhe0109.jpg'
cat_file = caffe_root + 'examples/images/cat.jpg'
Fox_file = caffe_root +  'CarStyle_test/Images/Fox/Fox_val/Fox0099.jpg'
Alto_file = caffe_root +  'CarStyle_test/Images/Alto/Alto_val/Alto0116.jpg'
Alto_file_fo = caffe_root +  'CarStyle_test/Images/Alto/Alto_val/Alto'
other_test_file = '/home/cari/Downloads/Alto0861.jpg'
Alto_file_t = '.jpg'
blob = caffe_pb2.BlobProto()
blob.ParseFromString(open(np_mean_file, "rb").read())

print blob.num, blob.channels, blob.height, blob.width
means = np.array(blobproto_to_array(blob) )
shape_means = means.reshape(3,100,100)
caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(npy_mean_file).mean(1).mean(1), channel_swap=(2,1,0),raw_scale=255,
                       image_dims=(100,100))
input_image  = caffe.io.load_image(other_test_file)
#input_image1= caffe.io.load_image(Alto_file_fo + '0019'+Alto_file_t)
#input_image2= caffe.io.load_image(Alto_file_fo + '0015'+Alto_file_t)
#input_image3= caffe.io.load_image(Alto_file_fo + '0010'+Alto_file_t)
#input_image4= caffe.io.load_image(Alto_file_fo + '0005'+Alto_file_t)
#input_image5= caffe.io.load_image(Alto_file_fo + '0020'+Alto_file_t)
#pylab.ion()
plt.imshow(input_image)
plt.show()
prediction = net.predict([input_image]) 
#print 'prediction shape:', prediction.shape
plt.plot(prediction[0])
#plt.plot(prediction[4])
plt.show()

for x in prediction:
    print x.argmax()
