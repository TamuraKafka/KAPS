####################
## 模型偏差 数据图  ##
####################
from matplotlib.patches import Polygon
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from tqdm import *


def prosess_data(data_name):
    ## 文件地址
    h_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_h.csv'
    R_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_R.csv'
    T_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/T.csv'
    Y_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y.csv'
    truth_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/truth.csv'

    ##  读取数据
    h = {}
    t2truth = {}
    w2tl = {}
    t2wl = {}

    # 处理result_h.csv
    with open(h_file, 'r', encoding='UTF-8-sig') as f1:
        reader = csv.reader(f1)
        next(reader)
        next(reader)
        next(reader)
        next(reader)
        next(reader)
        for row in reader:
            worker_id, worker_severity = row
            worker_id = int(worker_id)
            worker_severity = float(worker_severity)
            h[worker_id] = worker_severity

    # 处理 T.csv
    T_Matrix = pd.read_csv(T_file, header=None, sep=',', skiprows=1)
    T_Matrix = np.array(T_Matrix)

    # 处理 R.csv
    R_Matrix = pd.read_csv(R_file, header=None , sep=',')
    R_Matrix = np.array(R_Matrix)

    # 处理 truth.csv
    with open(truth_file, 'r', encoding='UTF-8-sig') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            task_id, truth = row
            task_id = int(task_id)
            truth = float(truth)

            t2truth[task_id] = truth

    # 处理 Y.csv
    with open(Y_file, 'r', encoding='UTF-8-sig') as f3:
        reader = csv.reader(f3)
        next(reader)
        for row in reader:
            worker_id, task_id, label = row
            worker_id = int(worker_id)
            task_id = int(task_id)
            label = int(label)

            if worker_id not in w2tl:
                w2tl[worker_id] = {}
            w2tl[worker_id].setdefault(task_id, label)

            if task_id not in t2wl:
                t2wl[task_id] = {}
            t2wl[task_id].setdefault(worker_id, label)
    return h,t2truth,w2tl,t2wl,T_Matrix,R_Matrix
################ 画图 偏差1  圆圈型密度图  实际偏差--理论偏差  ####################
def Actual_Theoretical_Deviation(data_name):
    h,t2truth,w2tl,t2wl,T_Matrix,R_Matrix = prosess_data(data_name=data_name)

    fig = plt.figure(figsize=(6, 6))
    ## 整理数据
    X_List = []
    Y_List = []
    data_dict = {}
    for i in w2tl.keys():
        for j in w2tl[i].keys():
            y_ij = w2tl[i][j]
            key = y_ij - t2truth[j]  # 实际偏差  Actual deviation
            value = h[i] + np.dot(T_Matrix[j],R_Matrix[i])  # 理论偏差 Theoretical Deviation
            X_List.append(key)
            Y_List.append(value)

    sns.regplot(x=X_List,y=Y_List,ci=None,color='orange',scatter = False,label='Fitting Line')
    sns.kdeplot(x=X_List, y=Y_List, fill=True,cmap='Oranges')

    plt.axvline(0,color='black',linestyle='-',alpha = 0.5)
    plt.axhline(0,color='black',linestyle='-',alpha = 0.5)
    plt.axline([0, 0], [1, 1],color='black',linestyle='--',label='y = x',alpha = 0.7)
    plt.legend(fontsize = 15,loc='upper left')
    plt.grid()

    plt.xlabel("Actual deviation",fontsize = 25)
    plt.ylabel("Theoretical deviation",fontsize = 23)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))

    if "GoodReadsDatasets/GoodReads_309t_120415w" == data_name:
        plt.xlim((-4, 2))
        plt.ylim((-4, 2))
        # plt.title("GoodReads Dataset",fontsize = 30)
        plt.savefig('C:/Users/chopin/Desktop/'+'GoodReads_Actual-Theoretical_Deviation.pdf', bbox_inches='tight')
    elif "DouBanDatasets/Douban_202t_269266w" in data_name:
        plt.xlim((-2, 3))
        plt.ylim((-2, 3))
        # plt.title("Douban Dataset", fontsize=30)
        plt.savefig('C:/Users/chopin/Desktop/' + 'Douban_Actual-Theoretical_Deviation.pdf', bbox_inches='tight')

    plt.show()

################ 画图 偏差2  圆圈型密度图  实际偏差--宽容度  ####################
def ActualDeviation_Kindness(data_name):
    h, t2truth, w2tl, t2wl, T_Matrix, R_Matrix = prosess_data(data_name=data_name)

    fig = plt.figure(figsize=(6, 6))
    ## 整理数据
    X_List = []
    Y_List = []
    data_dict = {}
    for i in w2tl.keys():
        for j in w2tl[i].keys():
            y_ij = w2tl[i][j]
            key = y_ij - t2truth[j]  # 实际偏差  Actual deviation
            value = h[i]
            X_List.append(key)
            Y_List.append(value)

    sns.regplot(x=X_List, y=Y_List, ci=None, color='blue', scatter=False, label='Fitting Line')
    sns.kdeplot(x=X_List, y=Y_List, fill=True, cmap='Blues')

    plt.axvline(0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    plt.axline([0, 0], [1, 1], color='black', linestyle='--', label='y = x', alpha=0.7)
    plt.legend(fontsize = 15,loc='upper left')
    plt.grid()

    plt.xlabel("Actual deviation", fontsize=25)
    plt.ylabel("Kindness", fontsize=23)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if "GoodReadsDatasets/GoodReads_309t_120415w" == data_name:
        plt.xlim((-4, 2))
        plt.ylim((-4, 2))
        # plt.title("GoodReads Dataset", fontsize=30)
        plt.savefig('C:/Users/chopin/Desktop/' + 'GoodReads_ActualDeviation-Kindness.pdf', bbox_inches='tight')
    elif "DouBanDatasets/Douban_202t_269266w" in data_name:
        plt.xlim((-2, 3))
        plt.ylim((-2, 3))
        # plt.title("Douban Dataset", fontsize=30)
        plt.savefig('C:/Users/chopin/Desktop/' + 'Douban_ActualDeviation-Kindness.pdf', bbox_inches='tight')

    plt.show()

def AbsoluteError(data_name):
    h, t2truth, w2tl, t2wl, T_Matrix, R_Matrix = prosess_data(data_name=data_name)
    ################# 画图 偏差2  山峰型密度图 ####################
    point1 = []
    point2 = []
    ## sns设置风格的代码一定要放在plt代码的前面才能生效
    sns.set_style(style='white')
    fig, ax = plt.subplots(figsize=(7, 4))

    for i in w2tl.keys():
        for j in w2tl[i].keys():
            y_ij = w2tl[i][j]
            point1.append(math.fabs(y_ij - t2truth[j] - h[i] - np.dot(T_Matrix[j], R_Matrix[i])))
            point2.append(math.fabs(y_ij - t2truth[j] - h[i]))

    data1 = pd.Series(point1)
    data2 = pd.Series(point2)
    ave1 = np.mean(data1)
    ave2 = np.mean(data2)

    # ["#8ECFC9","#FFBE7A","#FA7F6F","#82B0D2"]
    sns.kdeplot(data1, fill=True, color="#8ECFC9", label="KARS", alpha=0.7, ax=ax)
    sns.kdeplot(data2, fill=True, color="#FA7F6F", label=r'KARS-$\bar{p}$', alpha=0.7, ax=ax)

    plt.axvline(ave1, color='#8ECFC9', linestyle='--', label='Average Error of KARS',lw=2)
    plt.axvline(ave2, color='#FA7F6F', linestyle='--', label=r'Average Error of KARS-$\bar{p}$',lw=2)


    ax.legend(fontsize=16)

    plt.xlabel('Absolute error', fontsize=25)
    plt.ylabel('Density', fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().spines["top"].set_alpha(0.5)
    plt.gca().spines["right"].set_alpha(0.5)
    plt.gca().spines["bottom"].set_alpha(0.5)
    plt.gca().spines["left"].set_alpha(0.5)

    if "DouBanDatasets/Douban_202t_269266w" == data_name:
        plt.xlim((0, 3.5))
        # plt.title("Douban Dataset", fontsize=30)
        plt.savefig('C:/Users/chopin/Desktop/' + 'Douban_Preferences.pdf', bbox_inches='tight')

    if "GoodReadsDatasets/GoodReads_309t_120415w" == data_name:
        plt.xlim((0,2.5))
        # plt.title("GoodReads Dataset",fontsize = 30)
        plt.savefig('C:/Users/chopin/Desktop/'+'GoodReads_Preferences.pdf', bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    data_name_GoodReads = "GoodReadsDatasets/GoodReads_309t_120415w"
    data_name_Douban = "DouBanDatasets/Douban_202t_269266w"
    data_name_test = "DouBanDatasets_RedundancyCut/Douban_202t_110w_3r" # 测试数据

    # Actual_Theoretical_Deviation(data_name=data_name_test)

    # Actual_Theoretical_Deviation(data_name=data_name_GoodReads)
    # print("Done 1")
    # Actual_Theoretical_Deviation(data_name=data_name_Douban)
    # print("Done 2")
    #
    # ActualDeviation_Kindness(data_name=data_name_GoodReads)
    # print("Done 3")
    # ActualDeviation_Kindness(data_name=data_name_Douban)
    # print("Done 4")

    AbsoluteError(data_name=data_name_GoodReads)
    AbsoluteError(data_name=data_name_Douban)





