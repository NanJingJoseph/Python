# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:44:31 2018

@author: QCY
"""


#重采样前后绘图
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sklearn.preprocessing
Dataset=np.fromfile('./dataset/5000Samples/Dataset_Train.dump')
Dataset=Dataset.reshape((-1,21,5000))
Sequence_2500=Dataset[3][12][:2500]
Sequence_1000=scipy.signal.resample(Sequence_2500,1000)
plt.figure(figsize=(10,6))
plt.tick_params(labelsize=14)  
x=np.arange(0,10000,4)
y=np.arange(0,10000,10)
ax = plt.gca()  # 获取当前图像的坐标轴信息
ax.yaxis.get_major_formatter().set_powerlimits((0,1)) # 将坐标轴的base number设置为一位。
plt.plot(x,Sequence_2500)
plt.plot(y,Sequence_1000)
plt.title('Signal Resample',fontsize=20)
plt.ylabel('Amplitude(volt)',fontsize=18)
plt.xlabel('Time(ms)',fontsize=18)
plt.legend(['Before', 'After'], loc='upper left',fontsize=14)
plt.savefig('Signal Resample')
plt.show()


#对重采样信号作归一化
plt.figure(figsize=(10,6))
plt.tick_params(labelsize=15)  
x=np.arange(0,10000,10)
Sequence_1000_maxabs=sklearn.preprocessing.maxabs_scale(Sequence_1000)
plt.plot(x,Sequence_1000_maxabs)
plt.title('Max_Abs Scaled',fontsize=22)
plt.ylabel('Amplitude(volt)',fontsize=18)
plt.xlabel('Time(ms)',fontsize=18)
plt.savefig('Max_Abs Scaled')
plt.show()

















    













