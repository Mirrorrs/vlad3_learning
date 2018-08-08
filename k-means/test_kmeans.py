from numpy import *
from kmeans import *
import time
import matplotlib.pyplot as plt

# 第一步加载数据集
print("Loading data...")
dataSet = []
with open("./testSet.txt") as fileIn:
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

# 第二步做聚簇
print("Clustering...")
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet, k)

# 第三步展示结果
print("Show the result...")
showCluster(dataSet, k, centroids, clusterAssment)