from numpy import *
import time
import matplotlib.pyplot as plt

# 计算欧氏距离
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))

# 随机生成质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    # numpy中的zeros初始化方法
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

# k-means 聚簇
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    # numpy 中的mat函数将数据装为矩阵
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    # step1 初始化质心
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        # for 遍历 each sample
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            # for 遍历 each centroid
            # 找出最近的质心
            for j in range(k):
                # 计算欧氏距离
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # step3 update 聚簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2

        # step4 update 质心
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print("Cluster complete")
    return centroids, clusterAssment

# show cluster
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Dimension is not 2.")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Your K is too large.")
        return 1

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

    plt.show()

