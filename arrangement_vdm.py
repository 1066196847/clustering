# coding=utf-8
import csv
import os
import pickle
import cPickle
import math
from math import ceil
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import operator
import random

'''
函数说明：传入一个矩阵，找到最小的一个数字，返回对应的 ij索引
'''
def find_min_ij(matrix):
    min_num = matrix[1][2]
    min_i = 0
    min_j = 0
    for i in range(0,matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(i != j):
                if(matrix[i][j]<min_num):
                    min_i = i
                    min_j = j
                    min_num = matrix[i][j]
    return min(min_i,min_j),max(min_i,min_j)


'''
函数说明：返回两个“聚类簇”间的距离。这个仅仅针对的是离散距离，所以采用VDM距离来衡量 两个样本簇间的距离
参数说明：
data -> 第一列是id列，并且这个dataframe变量的索引就是那个id列，其余所有的列是 特征列（这个算法只支持“离散”变量。但是层次聚类是支持连续、分类）
cluster_1、cluster_2 -> list变量，表示两个样本簇，里面每一个元素是一个样本的索引
'''
def distance_cluster_ave(data, cluster_1, cluster_2):
    '''
    计算两个样本簇间 一一对应的两个样本 间的 VDM距离，最后取均值返回
    '''
    dis = 0.0
    for i in cluster_1:
        for j in cluster_2:
            # VDM距离计算的时候，是一个属性一个属性的计算，所以下面需要用过for循环，每一次计算一个属性的 VDM距离
            for k in range(1,len(data.columns)):
                # i j 是一个样本的“索引号”，k 是一个 属性（或者说是特征） 所在的列号（我需要这个列号，使用ix采取到 数据中的某个值）
                # 找出来这个属性中 i j两个样本取值。在总体数据中 各具有 几个？
                i_quzhi = data.ix[i, k]
                j_quzhi = data.ix[j, k]
                i_quzhi_num = len(data[data[k] == i_quzhi])
                j_quzhi_num = len(data[data[k] == j_quzhi])

                dis += (float(1)/i_quzhi_num)**2 + (float(1)/j_quzhi_num)**2

    return float(dis)/(i*j*k)


'''
函数说明：将原始数据进行层次聚类
参数说明：data->第一列是id列，并且这个dataframe变量的索引就是那个id列，其余所有的列是 特征列（这个算法只支持连续变量。但是层次聚类是支持连续、分类）
'''
def arrange(data):
    # 初始化 len(data) 个聚类簇，每一个簇里面放一个样本的“索引”。每一个簇是一个“list”变量，所有的簇放在一个“大list变量”。之后会不断更新
    # cluster_all的索引是从0开始的，But 数据样本的索引是从1开始的
    cluster_all = range(1, len(data)+1)
    for i in range(0, len(cluster_all)):
        cluster_all[i] = [cluster_all[i]]
    # 首先初始化“距离矩阵”。m是原始数据的“样本数”，我们现在要做出来一个矩阵，矩阵对应的 i j处 是 第i个聚类簇、第j个聚类簇 间的距离
    m = data.shape[0]
    # 下面一行代码中，必须要是0.0，否则python会默认自动类型转换
    matrix = DataFrame([[0.0] * m for i in range(m)])
    for i in range(m):
        for j in range(m):
            matrix[i][j] = distance_cluster_ave(data, cluster_all[i], cluster_all[j])
            matrix[j][i] = matrix[i][j]

        # 设定当前聚类簇个数：q=n
    q = m
    # 假设我们要求得的聚类 不多于 k 个
    k = 7
    while(q > k):
        print(q)
        # 找出来距离最近的两个聚类簇 cluster_all[min_i]   cluster_all[min_j] （min_i<min_j）。min_i min_j是“matrix”中的索引，也是“cluster_all”中的索引（从0开始的）
        min_i,min_j = find_min_ij(matrix)
        print 'min_i = ',min_i,'min_j = ',min_j
        # 合并 min_i min_j 两个聚类，将 min_j 这个聚类中的所有样本 放到 min_i中，又因为 min_j>min_i，所以将min_j之后的所有 聚类簇 的索引都在 cluster_all 中前进一位
        cluster_all[min_i] += cluster_all[min_j]

        for j in range(min_j,len(cluster_all)-1):
            cluster_all[j] = cluster_all[j+1]
        # 删除掉 cluster_all 最后一个索引处的数据
        del cluster_all[len(cluster_all)-1]
        '''
       到此时为止，cluster_all 中 还有 q-1 个聚类簇（包括那个添加了其他簇的 新簇）
       '''
        # 删除距离矩阵 中 列名为
        del matrix[min_j] # 删除列
        cols = list(matrix.index)# 删除行
        cols.remove(min_j)
        matrix = matrix.loc[cols]
        # 重置matrix的列名、索引
        matrix.columns = range(0,len(matrix.columns))
        matrix.index = range(0,len(matrix))
        # 更新距离矩阵。在这更新矩阵的时候，并不需要进行完全更新，只需要更新和 min_i 簇 有关的距离
        for j in range(0, q-1):
            matrix[min_i][j] = distance_cluster_ave(data, cluster_all[min_i], cluster_all[j])
            matrix[j][min_i] = matrix[min_i][j]
        q -= 1
        # 每次处理完都打印下 cluster_all 中的数据
        for i in cluster_all:
            print i


if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv('arrangement_vdm.txt')
    data.columns = list(range(0,len(data.columns)))
    data.index = list(data[0])
    # 进行聚类（在一次层次聚类的时候，只可以支持一种 变量类型），下面这个函数是支持 离散变量的，计算距离的时候，是计算 样本簇间的“vdm距离”
    arrange(data)
















































