#绘制皮尔逊矩阵图探究线性相关性
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
from pylab import *
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import sys
import time
import json
dataStr="\"[[1,2,3],[3,4,5],[3,5,6]]\""
def plotMatrix(data):
    init_data=data[2:-2]
    array=[i for i in range(len(init_data)) if init_data[i]==',' and init_data[i-1]==']' and init_data[i+1]=='[']
    num_array=[]
    number_array=[]
    for index in range(len(array)):
        if index==0:
            num_array.append(init_data[0:array[0]])
        else:
            num_array.append(init_data[array[index-1]+1:array[index]])
    num_array.append(init_data[array[index]+1:len(init_data)])
    # print(array)
    # print(number_array)
    for iarray in num_array:
        tar=iarray.replace(']','').replace('[','').split(',')
        new_tar=[float(i) for i in tar if len(i)>0]
        number_array.append(new_tar)
        # print(new_tar)
    pd_array=pd.DataFrame(number_array)
    # print(pd_array)
    pd_data=pd_array
    plt.rcParams['font.sans-serif'] =['Times New Roman'] 
    plt.rcParams['axes.unicode_minus'] = False
    x=pd_data.shape[1]#获取dataframe的列的个数，行为shape[0]
    a=pd_data.iloc[:,0:x].corr()
    plt.subplots(figsize=(18,18),dpi=600)
    mask = np.zeros_like(a, dtype=np.bool)   #定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
    mask[np.tril_indices_from(mask)]= True      #返回矩阵的上三角，并将其设置为true
    h=sns.heatmap(a,annot=True,vmax=2,square=True,cbar_kws={"shrink": 0.8},cmap="RdBu_r",linecolor="black",annot_kws={'size':22,"weight":'bold'},linewidths=3,cbar = False)
    # cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    # # cb.ax.tick_params(labelsize=20,size=2,width=2) #设置colorbar刻度字体大小。
    plt.xticks(fontsize=20,rotation=90,weight='bold')
    plt.yticks(fontsize=20,rotation=360,weight='bold')
    plt.savefig("./matrix-figure.jpg",dpi=600,transparent=True)
    print('Python have running')
# plotMatrix(dataStr)
plotMatrix(sys.argv[1])
