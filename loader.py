# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:07:25 2019

@author: xxw
"""

from torch.utils.data import Dataset
from utils import *
import torch
import logging
'''
Input of predition module
'''
class forw_Data(Dataset):

    def __init__(self, Forw, Pred):
        self.X = Forw
        self.y = Pred
        
    def __getitem__(self, item):
        
        x_t = self.X[item]['Forw']
        y_t = self.y[item]['Pred']


        x_t = torch.Tensor(x_t)
        y_t = torch.Tensor(y_t)
        x_t = x_t.unsqueeze(0)
        y_t = y_t.unsqueeze(0)
        
        return x_t, y_t

    def __len__(self):

        return len(self.X)
'''
Input of adjustment module   
'''   
class period_Data(Dataset):

    def __init__(self, T0, T1, T2):
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2

        
    def __getitem__(self, item):
        T_t0 = self.T0[item]['T0']
        T_t1 = self.T1[item]['T1']
        T_t2 = self.T2[item]['T2']


        T_t0 = torch.Tensor(T_t0)
        T_t1 = torch.Tensor(T_t1)
        T_t2 = torch.Tensor(T_t2)

        T_t0 = T_t0.unsqueeze(0)
        T_t1 = T_t1.unsqueeze(0)
        T_t2 = T_t2.unsqueeze(0)

        
        return T_t0,T_t1,T_t2

    def __len__(self):

        return len(self.T0)

#%%

class TrainLog():
     def __init__(self , logger = None):
        '''
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        '''
 
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.INFO)
        # 创建一个handler，用于写入日志文件
        self.log_name ='train0.log'
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 这个是python3的
        fh.setLevel(logging.INFO)
 
        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
 
        # 定义handler的输出格式
        formatter = logging.Formatter(' %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
 
        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
 
        #  添加下面一句，在记录日志之后移除句柄
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)
        # 关闭打开的文件
        fh.close()
        ch.close()
     def getlog(self):
        return self.logger
    
    
    
