# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:56:32 2018

@author: QCY
"""

import numpy as np
import matplotlib.pyplot as plt
#时间序列分析
import statsmodels.tsa.stattools as tsatools

Dataset=np.fromfile("E:/Python/workspace/TUH_EEG/dataset/1000Samples/Dataset_Train.dump")
Dataset=Dataset.reshape((-1,21,1000))

#采样频率，映射到时间
Frequency=100
#存放每个通道acf第一次降至零以下的索引号
Index=np.zeros((Dataset.shape[0],Dataset.shape[1]))
Average_Index=np.zeros((Dataset.shape[0],))
for i in range(Dataset.shape[0]):
    for j in range(Dataset.shape[1]):
        Acf=tsatools.acf(Dataset[i][j],nlags=1000)
        for k in Acf:
            if k < 0:
                Index[i][j]=list(Acf).index(k)
                break
            
            
for i in range(Dataset.shape[0]):
    avr_Index=(Index[i].sum()-Index[i].max()-Index[i].min())/(Dataset.shape[1]-2)
    Average_Index[i]= avr_Index        

    
Average_Index=Average_Index/Frequency


plt.figure(figsize=(10,6))
plt.hist(Average_Index, 200)
plt.xticks(np.linspace(0,4,17))
plt.tick_params(labelsize=12)
plt.xlabel('Time-Delay(sec.)',fontsize=15)
plt.ylabel('Counts',fontsize=15)
plt.title('Distribution of ARF Length for Samples_Train',fontsize=18)
plt.grid(True, linestyle = "-.", color = "grey", linewidth = 0.5)
#plt.savefig('./analysis for ACF/ARF_Len_for_Samples_Train')
plt.show()





