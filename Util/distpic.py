import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


sns.set_style('white')
# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams.update({'font.size': 25})
# plt.figure(figsize=(12, 9))
p2f = {}
totalDatasetWorkerSeverityList = []
maxTaskId = 0
doubanDatasetPath = "../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_202t_269266w/"
goodReadsDatasetPath = "../datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_120415w/"
twitterDatasetPath = '../datasets/QuantitativeCrowdsourcing/TwitterDatasets/Twitter_394t_46486w/'
def init():
    f = open(goodReadsDatasetPath + 'result_h.csv', mode='r', encoding='utf8')
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    workerSeverity = {}
    for line in f:
        items = line.strip().split(',')
        person = items[0]
        h = float(items[1])
        workerSeverity[person] = h
    totalDatasetWorkerSeverityList.append(workerSeverity)

    f = open(doubanDatasetPath + 'result_h.csv', mode='r', encoding='utf8')
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    workerSeverity = {}
    for line in f:
        items = line.strip().split(',')
        person = items[0]
        h = float(items[1])
        workerSeverity[person] = h
    totalDatasetWorkerSeverityList.append(workerSeverity)

    f = open(twitterDatasetPath + 'result_h.csv', mode='r', encoding='utf8')
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    workerSeverity = {}
    for line in f:
        items = line.strip().split(',')
        person = items[0]
        h = float(items[1])
        workerSeverity[person] = h
    totalDatasetWorkerSeverityList.append(workerSeverity)

def drawDistPlot():
    init()
    fig, ax = plt.subplots(figsize=(10, 7))
    workerIdList = []
    colors = ["#8ECFC9","#FFBE7A","#FA7F6F","#82B0D2"]
    labelList = ["GoodReads", "Douban", "Twitter"]
    i = 0
    print(totalDatasetWorkerSeverityList)
    for item in totalDatasetWorkerSeverityList:
        workerSeverityList = []
        tmpMin = np.min(list(item.values()))
        M_m = np.max(list(item.values())) - tmpMin

        for workerId in item:
            workerIdList.append(workerId)
            # h = (item[workerId] - tmpMin) / M_m
            h = item[workerId]
            workerSeverityList.append(h)

        data = pd.Series(workerSeverityList)  # 将数据由数组转换成series形式
        sns.kdeplot(data, shade=True, color=colors[i], label=labelList[i], alpha=.7, ax=ax)
        i += 1
    # data.plot(kind='density', linewidth=3, label='Douban')

        # 显示图例
        ax.legend(loc="upper left",fontsize = 20)
        # 显示图形
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.text(-0.3, 1, "Harsh Worker", verticalalignment="top",horizontalalignment="right")
    plt.text(0.3, 1, "Kind Worker", verticalalignment="top", horizontalalignment="left")
    plt.axvline(x=0,c="r",ls="--",lw=2)
    plt.ylabel("Density", font={'family': 'Arial', 'size': 20})
    plt.xlabel("Kindness", font={'family':'Arial', 'size':20})
    plt.savefig("./densityMap.pdf")
    plt.show()



if __name__ == '__main__':
    drawDistPlot()