#!/usr/bin/env python 
__metaclass__ = type
import caffe
import os
import time


#assist function
def endWith(s,*endString):
    """find the jpg files"""
    array = map(s.endswith,endString)
    if True in array:
        return True
    else:
        return False
#--endWith--END---#
class carPredictRun:
    def dirPRun(self,caffe_root,image_root_dir,net,labels):
        #create a list to cotain the whole jpg files  arrays
        image_process_list = []
        #cotains the every pic kind num
        check = "False"
        kind_num = []
        image_name_list = []
        predict_result = {}   
        #walk through the image-root-dir to find jpg files
        if os.path.exists(image_root_dir):
            for filename in os.listdir(image_root_dir):  
                if endWith(filename,'.jpg'):
                    kind_num.append(filename[5:7])
                    image_process_list.append(caffe.io.load_image(image_root_dir + filename))       
                    image_name_list.append(filename)
        
            image_nums = len(image_process_list)
            start_time = time.clock()
            prediction = net.predict(image_process_list)  
            end_time = time.clock()
            spend_time = (end_time - start_time)
            print("The dirPRun function run time is : %.03f seconds" %(spend_time))
            
            for p,n in zip(prediction,image_name_list):
                top_k = p.flatten().argsort()[-1:-6:-1]
                #print x.argmax()
                su =  labels[top_k]
                temp = []
                predict_result[n] = temp
                #print 'The kind of:',n,'is :'
                for i in range(0,len(su)):
                    if(i == 4):
                        #print su[i][11:],';'
                        temp.append(su[i][11:])
                    else:
                        #print su[i][11:],',',
                        temp.append(su[i][11:])
            check = "True"
        
        return check,image_name_list,predict_result
    #--dirPRun--END---#
    def singleCarPRun(self,caffe_root,image_path,net,labels):
         #create a list to cotain the whole jpg files  arrays
        image_process_list = []
        image_name_list = []
        predict_result = {}
        check = 'False'
        if os.path.isfile(image_path):
            image_process_list.append(caffe.io.load_image(image_path))
            image_name_list.append(image_path[-11:])
            start_time = time.clock()
            prediction = net.predict(image_process_list)  
            end_time = time.clock()
            spend_time = (end_time - start_time)
            print("The  singleCarPRun function run time is : %.03f seconds" %(spend_time))
            
            top_k = prediction.flatten().argsort()[-1:-6:-1]
            #print x.argmax()
            su =  labels[top_k]
            temp = []
            predict_result[image_name_list[0]] = temp
            #print 'The kind of:',n,'is :'
            for i in range(0,len(su)):
                if(i == 4):
                    #print su[i][11:],';'
                    temp.append(su[i][11:])
                else:
                    #print su[i][11:],',',
                    temp.append(su[i][11:])
            check = 'True'
        
        return check,image_name_list,predict_result
    #--singleCarPRun--END---#
    def MultiCarsPRun(self,caffe_root,image_dir,net,labels):
        dirPRun(caffe_root,image_dir,net,labels)
    #--MultiCarsPRun--END---#
    if __name__ == '__main__':
        print 'this is a single module'
