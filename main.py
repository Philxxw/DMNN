# %load main.py
"""
Created on Wed Aug 19 12:01:28 2020

@author: xxw
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from train_conv import *
from train import *
from forecast_conv import *
from forecast import *
import torch
#%%
def data_select (data):
    '''
    data: type of dataset 
    return:
        file : path of data file  
        fileorder, dataorder:sub data order
        train_count: size of training set
    '''
    if data == "China Mobile":
        file = ["dataA_fill","dataB_fill","dataC_fill"]
        fileorder = [0,1,2,4]
        dataorder = [0,1]
        train_count = 384
        print("Now we use the data of China Mobile")      
    elif data =="data of DataSet":
        file = ['DataSet/train13519', 'DataSet/train13619', 'DataSet/train13720', 
                'DataSet/train13820', 'DataSet/train13919', 'DataSet/train14019', 
                'DataSet/train25902', 'DataSet/train26717', 'DataSet/train26718', 
                'DataSet/train26803', 'DataSet/train26804', 'DataSet/train27303']
        fileorder = [0]
        dataorder = [0]
        train_count = 304
        print("Now we use the data of DataSet")
        
    return file,fileorder,dataorder,train_count
#%%
def model_data_select_train(Net_name,data,model,lr =0.0001):
    '''
    Net_name: list of neural network 
    data: datasets (AIIA or Kaggle)
    model: baseline model or DMNN  
    lr : learning rate, default = 0.0001
    '''
    file,fileorder,dataorder,train_count = data_select(data)    
    if model == "single_model":
        for net_name in  Net_name:
            print("Network:",net_name)
            for file_ in file:
                for file_or in fileorder:
                    for data_or in dataorder:
                        print("The %i part of the %i file in %s"%(data_or+1,file_or+1,file_))
                        train_single_model(lr,net_name,origin_data ="data/hour/"+file_+".csv",filename = "data/seg/"+str(file_or)+".npy",data_order = data_or,
                                           num_epochs =200,Select_num =918,batch_size =16,l_forward = 72,l_back =24,
                                           n = 14,train_count =train_count,parapath="paramater/single_model/"+net_name+'/'+file_+"/"+str(file_or)+str(data_or)+"/")
    elif model =="DMNN":
        #loss_type = ["MAPE"]
        loss_type = ["k=0.5","k=0.75","k=1","k=1.25","k=1.5"]
        for net_name in  Net_name:
            print("Network:",net_name)
            for file_ in file:
                for loss_t in loss_type:
                    for file_or in fileorder:
                        for data_or in dataorder:
                            if loss_t =="MAPE":
                                print("The %i part of the %i file in %s, which use %s loss function"%(data_or+1,file_or+1,file_,loss_t))
                            else:
                                print("The %i part of the %i file in %s, which use hybrid loss function with parameter %s"%(data_or+1,file_or+1,file_,loss_t))
                            train_DMNNM(lr,net_name,loss_t,origin_data ="data/hour/"+file_+".csv",filename = "data/seg/"+str(file_or)+".npy",data_order = data_or,
                                num_epochs =150,Select_num =918,batch_size =16,l_forward = 72,l_back =24,n = 14,
                                train_count =train_count,parapath="paramater/DMNNM/"+net_name+'/'+file_+"/"+loss_t+"/"+str(file_or)+str(data_or)+"/")
#%%
def model_data_select_test(Net_name,data,model):
    '''
    Net_name: list of neural network 
    data: datasets (AIIA or Kaggle)
    model: baseline model or DMNN  
    '''
    file,fileorder,dataorder,train_count = data_select(data)
    if model == "single_model":
        for net_name in  Net_name:
            for file_ in file:
                loss_list = []
                for file_or in fileorder:
                    for data_or in dataorder:
                        print("The part | the file   |  network ")
                        print("    %i%i   | %s | %s"%(data_or+1,file_or+1,file_,net_name))
                        loss_mean = test_single_model(net_name,origin_data ="data/hour/"+file_+".csv",filename = "data/seg/"+str(file_or)+".npy",data_order = data_or,
                                           Select_num =918,batch_size =1,l_forward = 72,l_back =24,n = 14,
                                           train_count =384,parapath="paramater/single_model/"+net_name+'/'+file_+"/"+str(file_or)+str(data_or)+"/"
                                           ,forepath="prediction/single_model/"+net_name+'/'+file_+"/"+str(file_or)+str(data_or)+"/")
                        loss_list.append(loss_mean.float())
                loss_list.append(sum(loss_list)/ len(loss_list))
                #print("Loss of the file %s: %s"%(file_,(sum(loss_list)/ len(loss_list))))
                pd.DataFrame(loss_list).to_csv("prediction/single_model/"+net_name+'/'+file_+"/loss.csv")
  
    elif model =="DMNN":
        loss_type = ["MAPE","k=0.5","k=0.75","k=1","k=1.25","k=1.5"]
        for net_name in  Net_name:
            for file_ in file:
                for loss_t in loss_type:
                    loss_list = []
                    for file_or in fileorder:
                        for data_or in dataorder:
                            if loss_t =="MAPE":
                                print("The part | the file    | network | loss Function")
                                print("    %i%i   | %s |%s| %s  "%(data_or+1,file_or+1,file_,net_name,loss_t))
                            else:
                                print("The part | the file    | network |       loss Function    ")
                                print("    %i%i   | %s |%s|hybrid loss function with %s"%(data_or+1,file_or+1,file_,net_name,loss_t))
                            loss_mean = test_DMNNM(net_name,loss_t,origin_data ="data/hour/"+file_+".csv",filename = "data/seg/"+str(file_or)+".npy",data_order = data_or,
                                Select_num =918,batch_size = 1,l_forward = 72,l_back =24,n = 14,
                                train_count =train_count,parapath="paramater/DMNNM/"+net_name+'/'+file_+"/"+loss_t+"/"+str(file_or)+str(data_or)+"/",
                                forepath="prediction/DMNNM/"+net_name+'/'+file_+"/"+loss_t+"/"+str(file_or)+str(data_or)+"/")
                            loss_list.append(loss_mean.float())
                    loss_list.append(sum(loss_list)/len(loss_list))
                    print("Loss of the file %s in %s: %s"%(file_,loss_t,(sum(loss_list)/ len(loss_list))))
                    pd.DataFrame(loss_list).to_csv("prediction/DMNNM/"+net_name+'/'+file_+"/"+loss_t+"/loss.csv")
#%%
if __name__ =='__main__':
    
    Net_name  =['conv_LSTM','conv_BiLSTM','BiLSTM','short_LSTM']
    Data = ["China Mobile","data of DataSet"]
    Model = ["single_model","DMNN"]
    
    ''' Train Model '''
    model_data_select_train(Net_name,Data[0],Model[1],lr = 0.0001)
    ''' Test Model '''
    model_data_select_test(Net_name,Data[0],Model[1])

    
