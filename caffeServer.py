#!/usr/bin/env python 
import socket
import caffe
import numpy as np
import json
import time
import threading
import carPredictRun

    
#establish the net
caffe_root = '/home/cari/caffe/'
image_root_dir = caffe_root + 'CarStyle_test/'
MODEL_FILE = caffe_root+'CarStyle_test/model_files/car_deploy.prototxt'
PRETRAINED = caffe_root+'CarStyle_test/trained_model/104Kinds_8952p.caffemodel'
npy_mean_file = caffe_root + 'CarStyle_test/carMeanNpyFile_100kinds.npy'
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=np.load(npy_mean_file).mean(1).mean(1),
                       channel_swap=(2,1,0),raw_scale=255,image_dims=(180,180))
label_file_name = caffe_root + 'CarStyle_test/synsets/carSynset_words.txt'
    
try:
    labels = np.loadtxt(label_file_name, str, delimiter='\t')
except:
    print "labels wrong"

print 'Net has been established'

#establish the socket server
hostName = '211.87.232.79'
portNum = 23333

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((hostName,portNum))
s.listen(50)
print 'the server has been established'
image_name_list = []
predict_result = {}
check = False
def tcplink(client,addr):
#while(True):
 #   client , addr = s.accept()
    caffe.set_mode_gpu()
    print 'Got connected from', addr
    #receive execode
    execode = client.recv(1024).decode('utf-8').strip('\r\n')
    print execode
    code = execode[:2]
    addrString = execode[2:]
    if(code == 'a|'):
        check,image_name_list,predict_result = carPredictRun.carPredictRun().singleCarPRun(caffe_root,addrString,net,labels)
    elif(code == 'b|'):
        check,image_name_list,predict_result = carPredictRun.carPredictRun().MultiCarsPRun(caffe_root,addrString,net,labels)
    elif(code == 'c|'):
        check,image_name_list,predict_result = carPredictRun.carPredictRun().dirPRun(caffe_root,addrString,net,labels)
    if check == "True":
        print('prepare to transform check')
        client.send(check + '\n')
        print('prepare to transform list and dict')
        #print client.recv(1024).decode('utf-8').strip('\n')
        
        inl = json.dumps(image_name_list)
        pr = json.dumps(predict_result)
        #send image_name_list
        client.send(inl + '\n')
        #receive image_name_list feedback
        print client.recv(1024).decode('utf-8').strip('\n')
        #send predict_result
        client.send(pr + '\n')
        #receuve predict_result feedback
        print client.recv(1024).decode('utf-8').strip('\n')
    else:
        print 'wth worong'
    client.close()

while(True):
    client , addr = s.accept()
   # print 'here'
    p = threading.Thread(target=tcplink,args=(client,addr))
    p.start()
    
