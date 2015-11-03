import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2
import pylab

caffe_root = '/home/cari/caffe/'

MODEL_FILE = caffe_root+'CarStyle_test/model_files/car_deploy.prototxt'
PRETRAINED = caffe_root+'CarStyle_test/trained_model/104kinds_8841p.caffemodel'

npy_mean_file = caffe_root + 'CarStyle_test/carMeanNpyFile_100kinds.npy'
test_file = caffe_root + 'CarStyle_test/0018_65.jpg'


caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(npy_mean_file).mean(1).mean(1), channel_swap=(2,1,0),raw_scale=255,
                       image_dims=(180,180))

input_image  = caffe.io.load_image(test_file)

#pylab.ion()
plt.imshow(input_image)
plt.show()
prediction = net.predict([input_image]) 
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-4:-1]
#print 'prediction shape:', prediction.shape
print prediction[0][65],prediction[0][63]
plt.plot(prediction[0])

plt.show()

for x in prediction:
    print x.argmax()
#label_file_name =   caffe_root + 'CarStyle_test/synsets/carSynset_words.txt'

#try:
    #labels = np.loadtxt(label_file_name, str, delimiter='\t')
#except:
    #print "wrong"

#print labels[top_k]
