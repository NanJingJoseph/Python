# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:53:34 2020

@author: 16534
"""


import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import numpy as np
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#raw 1st
r11=[0.7822,0.7763,0.7889,0.7837,0.7785,0.7881,0.7815,0.7741,0.7837,0.7852]
r12=[0.7533,0.7533,0.7333,0.7267,0.7400,0.7333,0.7600,0.7667,0.7533,0.7467]
#raw 2nd
r21=[0.7637,0.7770,0.7941,0.7881,0.7815,0.7741,0.7970,0.7874,0.7778,0.7696]
r22=[0.7667,0.7533,0.7467,0.7600,0.7200,0.7267,0.7533,0.7333,0.7733,0.7533]
#train 1st
t11=[0.9027,0.9111,0.9333,0.9119,0.9074,0.9052,0.9104,0.8941,0.9052,0.9103]
t12=[0.8733,0.8800,0.8600,0.8467,0.9000,0.8067,0.8333,0.8467,0.8200,0.8267]
#train 2nd
t21=[0.9133,0.9052,0.9311,0.9067,0.9052,0.8933,0.9141,0.9341,0.9252,0.9044]
t22=[0.8533,0.8400,0.8333,0.8467,0.8267,0.8533,0.8467,0.8333,0.8867,0.8667]
#test 1st
e11=[0.9082,0.9148,0.9030,0.9008,0.9171,0.9156,0.8897,0.9185,0.9156,0.9082]
e12=[0.8333,0.8600,0.8666,0.8333,0.8266,0.8333,0.8266,0.8466,0.8533,0.8200]
#teat 2nd
e21=[0.9015,0.9074,0.9097,0.8837,0.9089,0.9045,0.9037,0.9089,0.9052,0.9215]
e22=[0.8266,0.8333,0.8600,0.8200,0.8533,0.8600,0.8600,0.8200,0.8467,0.8400]
#axis轴
axis=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']
axis1=np.arange(1,11,1)

def to_percent(temp, position):
    return '%1.00f'%(100*temp) + '%'

#绘图
# plt.figure(figsize=(8,4.5))
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
plt.xlabel('折数', fontproperties=zhfont1)
plt.ylabel('准确率', fontproperties=zhfont1)
plt.ylim(0.6,0.95)
plt.plot(axis, e11, 'ro-', linewidth=1.0, linestyle='--',label='训练')
plt.plot(axis, e12, 'b^-', linewidth=1.0, linestyle='--',label='测试')

# for a, b ,c in zip(axis1, r11, r12):
#     plt.text(a, b, b, ha='center', va='bottom',fontsize=8)
#     plt.text(a, c, c, ha='center', va='top',fontsize=8)   
plt.legend()
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.grid()  # 生成网格
plt.savefig('3e1.png',dpi=200)
plt.show()