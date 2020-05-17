# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:20:27 2018

@author: QCY
"""

import numpy as np
import scipy.signal
Dataset=np.fromfile("./Dataset_Train.dump")
Dataset=Dataset.reshape((-1,21,2500))
Dataset=scipy.signal.resample(Dataset,1000,axis=2)      # 在2轴上，采样 1000 次，2500->1000
Dataset.tofile('./Dataset_Train.dump')































#Dataset_1000=scipy.signal.resample(Dataset_1000,1000,axis=2)

