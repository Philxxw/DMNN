# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:48:40 2019

@author: xxw
"""

import time
from eval import calcMAPE,calcCOSine
import torch
import torch.utils
import torch.utils.data
from utils import load_data
from model import *
from loader import *
import pandas as pd

torch.cuda.set_device(0)
log = TrainLog().getlog()

#%%
def train_DMNNM(lr_,net_name,loss_type,origin_data,filename,data_order,num_epochs,Select_num,batch_size,
                l_forward,l_back,n,train_count,parapath):
    '''
    #########################
    ##### load data     #####
    #########################
    '''
    
    T0_train,T1_train,T2_train, \
    Forw_train,Pred_train,\
    T0_test,T1_test,T2_test,\
    Forw_test,Pred_test = load_data(filename = filename,split_rate = 1/2,train_count =train_count,data_order=data_order,
                                city_path =origin_data,store_seg ="data/seg", week_num = 24,
                                store_path = "data/train",select_num =Select_num,step =4,T = 24 ,n = n,
                                l_back = l_back,m =1,l_forward =l_forward )
    
    tr_f_dataset = forw_Data(Forw_train, Pred_train)
    tr_f_dataloader = torch.utils.data.DataLoader(tr_f_dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0)
        
    tr_p_dataset = period_Data(T0_train,T1_train,T2_train)
    tr_p_dataloader = torch.utils.data.DataLoader(tr_p_dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0)


#%%  
#####################
#  model
#####################
    
    if net_name == 'short_LSTM':
        net1 = short_LSTM(input_dim =l_forward, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
    elif net_name == 'conv_LSTM':
        net1 = conv_LSTM(input_dim =20, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
    elif net_name == 'BiLSTM':
        net1 = BiLSTM(input_dim =l_forward, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
    elif net_name == 'conv_BiLSTM':
        net1 = conv_BiLSTM(input_dim =l_forward, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
#    elif net_name == 'LSTM_with_Attention':
#        net1 = LSTM_with_Attention(input_dim =l_forward, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
#    elif net_name == 'MLP':    
#        net1 = MLP(input_dim = l_forward,common_dim =2).cuda()
    net2 = period_net(input_dim =n, hidden_dim=2, num_layers =1,batch_size= batch_size).cuda()
    net3 =  metrix(in_dim = l_back,n_hidden_1 =2*l_back, n_hidden_2 =2*l_back, out_dim = l_back).cuda()

    # Initialize train model 
    loss_MSE = torch.nn.MSELoss(size_average=False)
    loss_MAPE  = calcMAPE()
    loss_COS  = calcCOSine()
    optimizer1 = torch.optim.Adam(list(net1.parameters()), lr=lr_)
    optimizer2 = torch.optim.Adam(list(net2.parameters()), lr=lr_)
    optimizer3 = torch.optim.Adam(list(net3.parameters()), lr=lr_)

#%% 
#####################
# Train model
#####################
    t1 = time.time()
    net1.train()
    net2.train()
    net3.train()
    list_train = []
    num = []
    
    for i in range(num_epochs):
        count = 0
        t3 = time.time()
        train_loss = 0 
        #train all training sets
        for data1,data2 in (zip(tr_f_dataloader,tr_p_dataloader)):
            x = data1[0]
            y  = data1[1].view(-1,1,l_back)
            t0 = data2[0].view(batch_size,n,-1)
            t1 = data2[1].view(batch_size,n,-1)
            t2 = data2[2].view(batch_size,n,-1)
            count = count+ 1
            x,y = x.cuda(),y.cuda()
            t0,t1,t2= t0.cuda(),t1.cuda(),t2.cuda()
            result = []
     
            for j in range(l_back):  
                # insert predicted value into window data
                x0,g_truth = x.view(batch_size,-1,1).cuda(),(y[:,:,j]).cuda()
                a,b,c, = t0[:,:,j].cuda(),t1[:,:,j].cuda(),t2[:,:,j].cuda()
                if j ==0:
                    x_f = x0
                else:
                    x_f = torch.cat([(torch.tensor(x_f,dtype =torch.double)),(torch.tensor(preds_f,dtype =torch.double))],dim = 2)    
                if len(x_f[0][0]) > l_forward :
                    x_mid = x_f
                    x_f = x_mid[:,:,1:len(x_f[0][0])]
                x_f = (torch.tensor(x_f,dtype =torch.float64).cuda()).view(-1,1,l_forward)      
            
            #forward 
            #output of predition module and adjustment module: preds_f and preds_p
            #dynamic adjust single predicted value  

                preds_p = net2(a,b,c).view(-1,1,1)
                preds_f = net1(x_f.cuda()).view(-1,1,1)
            #adjust predicted value
                g_truth =g_truth.float()
                pred_ = torch.add(preds_f, preds_p)
            #calculate loss in single point predition phase
                loss1 = loss_MAPE(g_truth,pred_)
                #loss1 = loss_MSE(pred_.view(-1,1),g_truth.view(-1,1))
                optimizer1.zero_grad()
                optimizer2.zero_grad()
            #backward
                loss1.backward(retain_graph=True)
            #update parameter
                optimizer1.step()
                optimizer2.step()
                result.append(pred_)
                
            result_ = torch.cat([s for s in result], 1)
            #refitting predicted sequence 
            r = net3(result_.view(-1,l_back))
            optimizer3.zero_grad()
            
            #choose hybrid loss function with different k
            if loss_type =="MAPE":
                loss = loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)) 
                #loss = loss_MSE(r.view(-1,l_back),y.view(-1,l_back))
            elif loss_type =="k=0.5":
                loss = torch.add(loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)),(-0.5)*loss_COS(y.view(-1,l_back),r.view(-1,l_back)))
            elif loss_type =="k=0.75":
                loss = torch.add(loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)),(-0.75)*loss_COS(y.view(-1,l_back),r.view(-1,l_back)))
            elif loss_type =="k=1":
                loss = torch.add(loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)),(-1)*loss_COS(y.view(-1,l_back),r.view(-1,l_back)))
            elif loss_type =="k=1.25":
                loss = torch.add(loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)),(-1.25)*loss_COS(y.view(-1,l_back),r.view(-1,l_back)))                
            elif loss_type =="k=1.5":
                loss = torch.add(loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)),(-1.5)*loss_COS(y.view(-1,l_back),r.view(-1,l_back)))
            #backward
            loss.backward()
            #update parameter
            optimizer3.step()
            train_loss =train_loss + loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)) 
            log.info("Hybrid:","data_order:%i"%count,loss_MAPE(y.view(-1,l_back),r.view(-1,l_back)))
        
        log.info(train_loss/num_epochs)
        print("epoch:%i,"%i,"MAPE_loss",train_loss/num_epochs)
        list_train.append(train_loss/num_epochs)
        t4 = time.time()
        log.info("training time:",t4-t3)
        #print("training time:",t4-t3)
        num.append(i)
    dataframe = pd.DataFrame({"num":num,"train_MAPE":list_train})
    dataframe.to_csv(parapath+"train_MAPE"+".csv",index=False,sep=',')
    torch.save(net1.state_dict(),parapath+'%s.pkl'%"net1")
    torch.save(net2.state_dict(),parapath+'%s.pkl'%"net2")
    torch.save(net3.state_dict(),parapath+'%s.pkl'%"net3")

    t2 = time.time()
    print("training time:",t2-t1)
    log.info("training time:",t2-t1)

