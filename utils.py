# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:01:32 2019

@author: xxw
"""

import numpy as np
import pandas as pd

#%%

def minmaxscaler(data):
    '''
    Normalized function
    data: input data
    '''
    min = np.amin(data)
    max = np.amax(data)
    return 10*(data-min)/(max-min)


#%%
def seg_new_data(datapath,storepath,week_num):
    '''
    databath: path of original data 
    storepath: path of data after segmented 
    week_num: week size of each subdataset
    '''
    data = pd.read_csv(datapath)
    columns = data.columns
    data = np.array(data)
    lenth = week_num*7*24
    total_len = data.shape[0]
    rest_len = total_len//lenth
    
    if (total_len % lenth) ==0:
        if rest_len ==1:
           seg_data =[]
           for n in range(0,lenth):
               seg_data.append(data[n])
           seg_data = pd.DataFrame(seg_data,columns = columns)

           np.save(storepath+'/'+"0.npy",seg_data) 
        else:
            for m in range(0,total_len-lenth,lenth):
                seg_data =[]
                for n in range(m,m+lenth):
                    seg_data.append(data[n])
                seg_data = pd.DataFrame(seg_data,columns = columns)
                np.save(storepath+'/'+"%i.npy"%(m/(lenth-1)),seg_data)
    else:
        seg_last =[]
        for i in range(0,total_len-rest_len-lenth,lenth):
            seg_data =[]
            for j in range(i,i+lenth):
                seg_data.append(data[j])

            np.save(storepath+'/'+"%i.npy"%(i/lenth),seg_data)
        for l in range(total_len-lenth,total_len):
            seg_last.append(data[l])
        np.save(storepath+'/'+"%i.npy"%((l/lenth)+1),seg_last)
        
#%%
def data_split(filename,random_num, T ,n ,l_back,m ,l_forward,test_T):
    '''
    Input parameters:
        filename: path of segmention data 
        l_forward: Input size of predition module, default = 72
        T: periodic length，default = 24
        n: length of periodic after sampled, default = 14
        l_back: length of predicted sequence，default = 24
        m: size of periodic data ，default = 1
        random_num: start point, the model will predict m*l_back size of predited sequence. n*T <= random_num <= (data)-m*l_back
    Return:
        input data 
    
    '''
    data_nT0,data_nT1,data_nT2 =[],[],[]
    data_p,data_lf = [],[]
    data = np.load(filename,allow_pickle=True)
    
    if data.shape[1] ==4:
        data = pd.DataFrame(np.array(data),columns = ["Num","Time","City","Value"])
        data = data.drop(["Time","City"],axis = 1)
        data = np.array(data)
    #else:
    #    data = minmaxscaler(data)

    start = random_num - (n*T)
    end = random_num + m*l_back
    for j in range(random_num-l_forward,random_num):
        data_lf.append(np.array(data[j][1]))
    
    for i in range(start,random_num,T):
        data_T0,data_T1,data_T2 = [],[],[]
        for l in range(0,int(l_back)): 
            a = data[i-1+l]
            b = data[i+l]
            c = data[i+1+l]
            
            data_T0.append(np.array(a[1]))
            data_T1.append(np.array(b[1]))
            data_T2.append(np.array(c[1]))
#        print(len(data_T0))
        data_nT0.append(data_T0)
        data_nT1.append(data_T1)
        data_nT2.append(data_T2)
                
    for k in range(random_num,end):
        data_p.append(np.array(data[k][1]))
        
    data_lf = np.reshape(data_lf,(l_forward,1)).T
    data_p = np.reshape(data_p,(m*l_back,1))   
    data_nT0 = np.reshape(data_nT0,(n,l_back)).T     
    data_nT1 = np.reshape(data_nT1,(n,l_back)).T    
    data_nT2 = np.reshape(data_nT2,(n,l_back)).T 
    return data_nT0,data_nT1,data_nT2,data_lf,data_p
    
    
#%%
def create_multi_data(filename,store_path,select_num,step =4,split_rate = 1/2,train_count =384,T = 24 ,n =14,l_back = 24,m =1,l_forward =72):
    '''
    Input parameter:
        select_num: number of short-term data in each subdataset 
        periodic length，default = 24
        n: length of periodic after sampled, default = 14
        l_back: length of predicted sequence，default = 24
        m: size of periodic data ，default = 1
    '''
    ls_num = []
    
    data = np.load(filename,allow_pickle=True)
    lendata = np.array(data).shape[0]
#    if n*T > l_forward:
#        ls_num = list(range(n*T,lendata-m*l_back,step))
#    else:
#        ls_num = list(range(l_forward,lendata-m*l_back,step))
    ls_num = list(range(n*T,lendata-m*l_back,step))
    end_num = select_num*split_rate
    split_num = int(1/split_rate)
    for i in range(0,split_num):
        train_num,test_num = ls_num[int(i*end_num) : int((i*end_num)+train_count)], ls_num[(int((i*end_num)+train_count)):(int((i+1)*end_num))]
        s_T0,s_T1,s_T2,Forw,Pred = [],[],[],[],[]
        s_T0_test,s_T1_test,s_T2_test,Forw_test,Pred_test = [],[],[],[],[]
        if select_num > ((lendata-m*l_back-n*T)/3):
            return print("Warning: select_num %i is out of range"%select_num )
        
        for train_number in train_num:
            T0,T1,T2,forward,pred=data_split(filename,train_number,T,n,l_back,m,l_forward,test_T = False)
            s_T0.append({"T0":(T0)})
            s_T1.append({"T1":(T1)})
            s_T2.append({"T2":(T2)})
            Forw.append({"Forw":(forward)})
            Pred.append({"Pred":(pred)})
        for test_number in (test_num):
            T0,T1,T2,forward,pred=data_split(filename,test_number,T,n,l_back,m,l_forward,test_T = True)
            s_T0_test.append({"T0":(T0)})
            s_T1_test.append({"T1":(T1)})
            s_T2_test.append({"T2":(T2)})
            Forw_test.append({"Forw":(forward)})
            Pred_test.append({"Pred":(pred)})    

        np.save(store_path+'/'+'T0_train%i.npy'%i,s_T0)
        np.save(store_path+'/'+'T1_train%i.npy'%i,s_T1)
        np.save(store_path+'/'+'T2_train%i.npy'%i,s_T2)
        np.save(store_path+'/'+'Forw_train%i.npy'%i,Forw)
        np.save(store_path+'/'+'Pred_train%i.npy'%i,Pred)
        np.save(store_path+'/'+'T0_test%i.npy'%i,s_T0_test)
        np.save(store_path+'/'+'T1_test%i.npy'%i,s_T1_test)
        np.save(store_path+'/'+'T2_test%i.npy'%i,s_T2_test)
        np.save(store_path+'/'+'Forw_test%i.npy'%i,Forw_test)
        np.save(store_path+'/'+'Pred_test%i.npy'%i,Pred_test)

    
    
#%%
def load_data(select_num ,step ,split_rate,train_count ,T ,n ,l_back ,m ,l_forward,data_order,city_path ="data/hour/A.csv",store_seg ="data/seg", week_num = 24,filename = "data/seg/0.npy",store_path = "data/train", ):
    seg_new_data(city_path,store_seg,week_num =24)
    '''
    All input parameter from other difined functions of utils.py
    
    Return:
        traing data and forecasting data 
        
    '''
    create_multi_data(filename,store_path,select_num =select_num,step = step,split_rate = split_rate ,train_count = train_count,T  = T,n = n,l_back = l_back  ,m = m,l_forward = l_forward)
    s_T0 = np.load(store_path + '/'+'T0_train%i.npy'%data_order,allow_pickle=True)
    s_T1 = np.load(store_path + '/'+'T1_train%i.npy'%data_order,allow_pickle=True)
    s_T2 = np.load(store_path + '/'+'T2_train%i.npy'%data_order,allow_pickle=True)
    Forw = np.load(store_path + '/'+'Forw_train%i.npy'%data_order,allow_pickle=True)
    Pred = np.load(store_path + '/'+'Pred_train%i.npy'%data_order,allow_pickle=True)
    s_T0_test = np.load(store_path + '/'+'T0_test%i.npy'%data_order,allow_pickle=True)
    s_T1_test = np.load(store_path + '/'+'T1_test%i.npy'%data_order,allow_pickle=True)
    s_T2_test = np.load(store_path + '/'+'T2_test%i.npy'%data_order,allow_pickle=True)
    Forw_test = np.load(store_path + '/'+'Forw_test%i.npy'%data_order,allow_pickle=True)
    Pred_test = np.load(store_path + '/'+'Pred_test%i.npy'%data_order,allow_pickle=True)

    return s_T0,s_T1,s_T2,Forw,Pred,s_T0_test,s_T1_test,s_T2_test,Forw_test,Pred_test

    

