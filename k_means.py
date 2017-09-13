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
函数说明：求3个数字的最大值
'''
def min(a,b,c):
    temp = 0
    if(a>b):
        temp = b
    else:
        temp = a
    if(temp>c):
        temp = c
    return temp

'''
函数说明：从data这个数据集中根据现有的k个聚类中心，将所有的样本分到k个类中，并且计算出来新k个类中的“新中心点”
参数说明：
data->第一列是id列，并且这个dataframe变量的索引就是那个id列，其余所有的列是 特征列（暂时只支持连续变量、连续分类变量）
u1 u2 u3->3个聚类中心
返回值说明：
u1_new, u2_new, u3_new：3个新的聚类中心
c1, c2, c3：每一个聚类中心，对应的所有的样本
'''
def cal_ave(data, u1, u2, u3):
    # 建立3个空列表，用来装上述每一个聚类中心对应的其余样本
    c1 = []
    c2 = []
    c3 = []
    # 计算所有样本和上述3个样本间的距离
    for i in data['id']:
        dis_1 = math.sqrt((data.ix[i, 1] - u1[0]) ** 2 + (data.ix[i, 2] - u1[1]) ** 2)
        dis_2 = math.sqrt((data.ix[i, 1] - u2[0]) ** 2 + (data.ix[i, 2] - u2[1]) ** 2)
        dis_3 = math.sqrt((data.ix[i, 1] - u3[0]) ** 2 + (data.ix[i, 2] - u3[1]) ** 2)
        # print(dis_1,dis_2,dis_3)
        max_num = min(dis_1, dis_2, dis_3)
        if(max_num == dis_1):
            c1.append(i)
        elif(max_num == dis_2):
            c2.append(i)
        else:
            c3.append(i)
    # print c1
    # print c2
    # print c3
    # 计算现在每一个聚类的“均值向量”
    u1_new = [0,0]
    for i in c1:
        u1_new[0] += data.ix[i, 1]
        u1_new[1] += data.ix[i, 2]
    u1_new[0] = float(u1_new[0]) / len(c1)
    u1_new[1] = float(u1_new[1]) / len(c1)

    u2_new = [0,0]
    for i in c2:
        u2_new[0] += data.ix[i, 1]
        u2_new[1] += data.ix[i, 2]
    u2_new[0] = float(u2_new[0]) / len(c2)
    u2_new[1] = float(u2_new[1]) / len(c2)

    u3_new = [0,0]
    for i in c3:
        u3_new[0] += data.ix[i, 1]
        u3_new[1] += data.ix[i, 2]
    u3_new[0] = float(u3_new[0]) / len(c3)
    u3_new[1] = float(u3_new[1]) / len(c3)

    return u1_new,u2_new,u3_new,c1,c2,c3


'''
函数说明：进行聚类分类，然后返回一个字典，字典的键是一个聚类中心(一个样本id)---字典的键值是一个list（属于这个聚类中心的所有样本的id）
参数说明：data->第一列是id列，并且这个dataframe变量的索引就是那个id列，其余所有的列是 特征列（暂时只支持连续变量、连续分类变量）
'''
def clustering(data):
    # 设定聚类数目
    k = 3

    # 随机选取k个样本，作为初始均值向量
    # n1 = random.randrange(1, len(data))
    # n2 = random.randrange(1, len(data))
    # while (n2 == n1):
    #     n2 = random.randrange(1, len(data))
    # n3 = random.randrange(1, len(data))
    # while (n3 == n1 or n3 == n2):
    #     n3 = random.randrange(1, len(data))
    n1 = 6
    n2 = 12
    n3 = 27

    # 根据上面的“3个样本的id”，初始化3个聚类中心
    u1 = (data.ix[n1, 1], data.ix[n1, 2])
    u2 = (data.ix[n2, 1], data.ix[n2, 2])
    u3 = (data.ix[n3, 1], data.ix[n3, 2])

    i = 0
    while(1):
        print(i)
        i+=1
        # print u1
        # print u2
        # print u3
        u1_new, u2_new, u3_new, c1, c2, c3 = cal_ave(data, u1, u2, u3)
        # print u1_new
        # print u2_new
        # print u3_new
        # u1和u1_new分别是 前一个 当前 的两个聚类中心
        if(u1[0]==u1_new[0] or u1[1]==u1_new[1] or u2[0]==u2_new[0] or u2[1]==u2_new[1] or u3[0]==u3_new[0] or u3[1]==u3_new[1]):
            break
        # 更新聚类中心
        u1 = u1_new
        u2 = u2_new
        u3 = u3_new

    # 上面while停止运行的时候 u1 u2 u3是最终我们确定的3个聚类中心，c1 c2 c3是对应每一个聚类中心对应的所有“样本”
    dict_will_return = {}
    dict_will_return[tuple(u1)] = c1
    dict_will_return[tuple(u2)] = c2
    dict_will_return[tuple(u3)] = c3

    print dict_will_return

if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv('k_means.txt',header=None)
    data.columns = ['id','midu','hantang']
    data.index = list(data['id'])
    # 进行聚类
    clustering(data)
















































