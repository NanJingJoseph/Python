# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:15:04 2018

@author: QCY
"""
import os
import os.path as op
import numpy as np
import mne
import random
##import matplotlib.pyplot as plt
###全局变量
ch_names={'EEG FP1-REF', 'EEG FP2-REF','EEG F3-REF',   ###需要用到的通道数共21条
          'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 
          'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 
          'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
          'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 
          'EEG T6-REF','EEG FZ-REF', 'EEG CZ-REF',
          'EEG PZ-REF','EEG A1-REF', 'EEG A2-REF'}
SampleTime=10 ##采样时间10s
FREQUENCY=250##采样频率；[起始时间*采样频率,截至时间*采样频率]为截取范围

#存储.edf和.rect文件的绝对路径，方便读取数据文件
AbsoluteRoute_edf=[]
AbsoluteRoute_rect=[]

Epi_Period=[]#存储训练集癫痫记录,每个包含了（起始时间，截至时间，取自第几个.rect(.edf)）,将发作时长小于等于10S且起始时间小于60S的记录去除
Epi_SubPeriod=[]#将大于10S的记录细分出多个长度为10S的片段
Nor_Period=[]#存储训练集正常记录,每个包含了（起始时间，截至时间，取自第几个.rect(.edf)）,将发作时长小于等于10S且起始时间小于60S的记录去除
Nor_SubPeriod=[]#将大于10S的记录细分出多个长度为10S的片段

##此处的ndarray大小是我经过实验后得到的数组大小，所以直接框定了数量的范围;若具体数据个数未知，可以初始化成一个较大的数组如（2000，21，5000）
##再对该ndarray中非0部分进行截取
Epi=np.ndarray((2000,21,2500))
Nor=np.ndarray((2000,21,2500))

#使用时，放在v1.0.4的同级目录下
RootRoute="./v1.0.4/Train/01_tcp_ar"

#1.函数定义
###########################################################################################################################
#将.rect数据中读取的数据做修剪，每行去除空格并按逗号分隔成list格式
def trim_str(string):   
    string=string.strip()        #去除首尾空格
    string=string.replace(' ','')#把空格replace掉
    string=string.split(',')     #用逗号分隔
    return string

#判断后缀名
def end_with(s,*endstring):
    array=map(s.endswith,endstring)
    if True in array:
        return True
    else:
        return False
    
#筛选需要使用的通道,返回筛选完通道的数据
def filt_ch(raw):
    DropChannels=[]
    for ch in raw.ch_names:
        if ch not in ch_names:
            DropChannels.append(ch)#用到21个名称了
    raw.drop_channels(DropChannels)   #什么是dropchannels()？？？？？？
    return raw

#读取train目录下后缀名为lastname的文件路径，
# RootRoute_train="E:\\Python\\workspace\\TUH_EEG\\v1.0.4\\train\\01_tcp_ar" ，进入该目录下的二级目录提取文件
#读取eval目录下后缀名为lastname的文件路径，
# RootRoute_eval="E:\\Python\\workspace\\TUH_EEG\\v1.0.4\\eval\\01_tcp_ar" ，进入该目录下的二级目录提取文件
def read_route(lastname,RootRoute):
    AbsoluteRoute=[]
    for one_class in os.listdir(RootRoute):
         Tmp_Route1=op.join(RootRoute,one_class)
         for two_class in os.listdir(Tmp_Route1):
             Tmp_Route2=op.join(Tmp_Route1,two_class)
             for final in os.listdir(Tmp_Route2):
                 if end_with(final,lastname):
                     AbsoluteRoute.append(op.join(Tmp_Route2,final))
    return AbsoluteRoute

#查看该.edf文件完整性，是否为采样250Hz，是否包含所需通道
def check_file_integrity(raw_instance):
    if raw_instance.info["sfreq"] == 250:
        if ch_names.issubset(set(raw_instance.info["ch_names"])):#用到21个名称了
            return True

#2.建立数据集
############################################################################################################################

#读取根目录下所有的.edf文件,将所有.edf的路径保存到AbsoluteRoute_edf内
#读取根目录下所有的.rect文件,将所有.edf的路径保存到AbsoluteRoute_rect内
AbsoluteRoute_edf=read_route('.edf',RootRoute)   #.edf .rect 均为数据集中的文件格式
AbsoluteRoute_rect=read_route('.rect',RootRoute)

'''
Seizure events, for example, typically start on a small number of channels and then spread to other channels. 

The".rec" format allows this to be captured. 

The ".rect" format requires all channels to have the same annotation.

对于.rect文件，所有通道同一时刻标注相同，因此仅需要分析第0个通道记录的癫痫发作情况   
'''

#从AbsoluteRoute_rect取出第0条通道记录的癫痫发作记录，存储每个样本中的癫痫发作时间段用于截取癫痫数据               
for i in range(0,len(AbsoluteRoute_rect)):#一维路径
#按顺序遍历所有.rect文件
    with open (AbsoluteRoute_rect[i]) as f:
        for sequence in f:
            sequence=trim_str(sequence)
            if sequence[0]=='0':
                if sequence[3]=='7':
                    #将时间段取出并转换成int型
                    sequence=[int(float(j)) for j in sequence[1:3]]
                    #在该条数据的最后添加这是第几个.rect文件(对应于.edf文件)
                    sequence.append(i)
                    Epi_Period.append(sequence)
#若list的首个字符不是'0'，也就是说已经遍历完第0个通道，就退出该循环，并且关闭该.rect文件，开始遍历下一个.rect文件                        
            else:
                break
            
#类似地，从AbsoluteRoute_rect取出第0条通道记录的正常记录，存储每个样本中的正常时间段用于截取正常数据      
for i in range(0,len(AbsoluteRoute_rect)):
#按顺序遍历所有.rect文件
    with open (AbsoluteRoute_rect[i]) as f:
        for sequence in f:
            sequence=trim_str(sequence)
            if sequence[0]=='0':
                if sequence[3]=='6':
                    #将时间段取出并转换成int型
                    sequence=[int(float(j)) for j in sequence[1:3]]
                    #在该条数据的最后添加这是第几个.rect文件(对应于.edf文件)
                    sequence.append(i)
                    Nor_Period.append(sequence)
#若list的首个字符不是'0'，也就是说已经遍历完第0个通道，就退出该循环，并且关闭该.rect文件，开始遍历下一个.rect文件                        
            else:
                break

#去除起始时间小于60s的片段 
Temp=[]
for i in Epi_Period:
    if i[1]-i[0]>SampleTime and i[0]>60:
        Temp.append(i)
Epi_Period=Temp

Temp=[]
for i in Nor_Period:
    if i[1]-i[0]>SampleTime and  i[0]>60:
        Temp.append(i)
Nor_Period=Temp

#长度大于10s的还可以细分出多个长度为10s的片段
for i in Epi_Period:
    n=(i[1]-i[0])//SampleTime
    for j in range(0,n):
        Epi_SubPeriod.append([i[0]+SampleTime*j,i[0]+SampleTime*(j+1),i[2]])    

for i in Nor_Period:
    n=(i[1]-i[0])//SampleTime
    for j in range(0,n):
        #添加样本号,当前.edf文件的有效索引
        Nor_SubPeriod.append([i[0]+SampleTime*j,i[0]+SampleTime*(j+1),i[2]]) 

Epi_SubPeriod=Epi_SubPeriod[:]
random.shuffle(Nor_SubPeriod)
Nor_SubPeriod=Nor_SubPeriod[:1000]

#利用Epi_SubPeriod的时间段乘以采样频率250Hz得出切片范围，切片后可以得到癫痫训练样本,每个大小为(21,2500)
validfile_count=0
for i in Epi_SubPeriod:  
    try: 
        raw=mne.io.read_raw_edf(AbsoluteRoute_edf[i[2]],verbose=False,preload=True)
        if check_file_integrity(raw):
            raw=filt_ch(raw)
            data=raw.get_data()
            data=data[0:data.shape[0],i[0]*FREQUENCY:i[1]*FREQUENCY]
            #若多通道的数据相同，则判断为无效数据弃用
            if (data[0]==data[1]).all():
                print(' Invalid Epi segment was detected:\n{}.'.format(data))
                continue     
            Epi[validfile_count]=data
            validfile_count+=1
            print(' Extracting the {}th valid Epi segment.'.format(validfile_count))
    except FileNotFoundError:
        continue
    except ValueError:
        continue
Epi=Epi[:validfile_count]

#利用Nor_SubPeriod的时间段乘以采样频率250Hz得出切片范围，切片后可以得到正常训练样本,每个大小为(21,2500)
validfile_count=0
for i in Nor_SubPeriod:
    try:
        raw=mne.io.read_raw_edf(AbsoluteRoute_edf[i[2]],verbose=False,preload=True)
        if check_file_integrity(raw):
            raw=filt_ch(raw)
            data=raw.get_data()        
            data=data[0:data.shape[0],i[0]*FREQUENCY:i[1]*FREQUENCY]
            #若多通道的数据相同，则判断为无效数据弃用
            if (data[0]==data[1]).all():
                print(' Invalid Nor segment was detected:\n{}.'.format(data))
                continue
            Nor[validfile_count]=data
            validfile_count+=1
            print(' Extracting the {}th valid Nor segment'.format(validfile_count))
    except FileNotFoundError:
        continue
    except ValueError:
        continue
Nor=Nor[:validfile_count]

Dataset=np.vstack((Epi,Nor))
Label=np.zeros((Dataset.shape[0],))
Label[:Epi.shape[0]]=1

Index=np.arange(Dataset.shape[0])
np.random.shuffle(Index)
Dataset=Dataset[Index]
Label=Label[Index]

###将数据存储到本地
#Dataset.tofile("./Dataset_Train.dump")
#Label.tofile("./Label_Train.dump")















