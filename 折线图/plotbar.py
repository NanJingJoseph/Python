# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:53:34 2020

@author: 16534
"""


"""
    默认的是竖值条形图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 将全局的字体设置为黑体
matplotlib.rcParams['font.family'] = 'SimHei'

# 数据
N = 3
y = [0.782,0.911,0.907]
y2 = [0.748,0.849,0.841]
x = np.arange(N)
# 添加地名坐标
str1 = ("预处理前", "预处理后，测试集", "预处理后，评估")

# 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
p1 = plt.bar(x-0.125, height=y, width=0.25, label="训练")
p1 = plt.bar(x+0.125, height=y2, width=0.25, label="测试")
plt.xticks( np.arange(3), ("预处理前",'预处理后，训练集', '预处理后，评估集'))
plt.ylim(0.65,0.95)

# 添加数据标签
for a, b ,c in zip(x, y,y2):
    plt.text(a-0.125, b + 0.006, '%.1f' % (100*b)+'%', ha='center', va='bottom', fontsize=12)
    plt.text(a+0.125, c + 0.006, '%.1f' % (100*c)+'%', ha='center', va='bottom', fontsize=12)
# 添加图例
plt.legend()
plt.savefig('result.png',dpi=200)
# 展示图形
plt.show()