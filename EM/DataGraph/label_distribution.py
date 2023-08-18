####################
##   标签分布实验   ##
####################

import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
from tqdm import *

class DistributionTest:
    def __init__(self,truth_file,Y_file,h_file,R_file,T_file,lpd_file,a_file):
        self.t2truth,self.w2tl,self.t2wl,self.w2t_diff= self.initDataset(truth_file,Y_file)
        self.w2sum_dic , self.sum2w_dic_list = self.getDataStructure()


    def initDataset(self,truth_file,Y_file):
        ##  读取数据
        t2truth = {}
        w2tl = {}
        w2t_diff = {}
        t2wl = {}


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

                if worker_id not in w2t_diff:
                    w2t_diff[worker_id] = {}
                w2t_diff[worker_id].setdefault(task_id,label - t2truth[task_id])


        return t2truth,w2tl,t2wl,w2t_diff

    def getDataStructure(self):
        ### 将w2tl转化为向量
        w2sum = {}  # 工人的做题数

        for i in self.w2tl.keys():
            w2sum[i] = len(self.w2tl[i].keys())
        w2sum = sorted(w2sum.items(), key=lambda x: x[1], reverse=True)

        w2s_dic = {}  # 工人i : 做题数
        for w in w2sum:
            w2s_dic[w[0]] = w[1]
        s2w_dic = {}  # 做题数 : 工人编号列表
        for k, v in w2s_dic.items():
            if v not in s2w_dic.keys():
                s2w_dic[v] = []
            s2w_dic[v].append(k)
        return w2s_dic,s2w_dic

    # 通过减少 做题数<cut_size的工人 来计算List，并画图
    def getListBySize(self,cut_size):
        person_list = []
        cut_worker_num = 0
        sum2w_dic_list = dis_test.sum2w_dic_list
        for task_sum in sum2w_dic_list:
            if task_sum <= cut_size:
                cut_worker_num += len(sum2w_dic_list[task_sum])
                continue

            list1 = sum2w_dic_list[task_sum]
            for i in list1:
                person_list.append(i)

        return person_list, len(self.w2tl) - cut_worker_num



def twitterLabelDis():
    sns.set_style('white')
    fig = plt.figure()
    Y = open('../../datasets/QuantitativeCrowdsourcing/TwitterDatasets/Twitter_394t_46486w/Y.csv', mode="r", encoding="utf8")
    w2tl = {}
    line = Y.readline()
    labelList = []
    while line:
        line = Y.readline()
        if len(line.strip().split(',')) < 3:
            break
        (worker, task, label) = line.strip().split(',')
        # print(worker, task, label)
        if worker not in w2tl:
            w2tl[worker] = {}
        w2tl[worker][task] = label
        labelList.append(float(label))
    data = pd.Series(labelList)  # 将数据由数组转换成series形式

    sns.kdeplot(data, fill=False, color="#FA7F6F", label="Twitter", alpha=.7)
    # 显示图例
    ax.legend(loc="upper right",fontsize = 15)
    plt.ylabel("Density",fontsize = 25)
    plt.xlabel("Label",fontsize = 25)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize=15)
   # plt.title('Twitter Dataset', fontsize=30)
    plt.savefig('C:/Users/chopin/Desktop/' + 'Label_Distribution_Twitter.pdf', bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    ###### Douban 数据集
    # data_name = "DouBanDatasets_RedundancyCut/Douban_202t_197w_5r"  # 测试数据集
    data_name = "DouBanDatasets/Douban_202t_269266w"                # Douban
    # data_name = "GoodReadsDatasets/GoodReads_309t_120415w"            # GoodReads

    ## 文件地址
    h_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_h.csv'
    R_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_R.csv'
    T_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/T.csv'
    Y_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y.csv'
    truth_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/truth.csv'
    lpd_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/t2lpd.csv'
    a_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_a.csv'

    dis_test = DistributionTest(truth_file=truth_file,Y_file=Y_file,h_file=h_file,R_file=R_file,T_file=T_file,lpd_file=lpd_file,a_file=a_file)
    list,worker_number = dis_test.getListBySize(cut_size=-1)

    # list = []
    # for i in dis_test.w2t_diff.keys():
    #     for j in dis_test.w2t_diff[i].keys():
    #         list.append(dis_test.w2t_diff[i][j])
    # sns.kdeplot(list,fill=True)
    # plt.show()

    ## 取数据画图
    min_value_graph_list = []
    max_value_graph_list = []
    ave_value_graph_list = []
    cnt_min = 0
    cnt_max = 0
    for i in list:
        min_val = 100
        max_val = -100
        sum = 0
        size = 0
        for j in dis_test.w2t_diff[i].keys():
            sum += dis_test.w2t_diff[i][j]
            size += 1
            if dis_test.w2t_diff[i][j] < min_val:
                min_val = dis_test.w2t_diff[i][j]
            if dis_test.w2t_diff[i][j] > max_val:
                max_val = dis_test.w2t_diff[i][j]
        ave_value_graph_list.append(sum / size)
        min_value_graph_list.append(min_val)
        max_value_graph_list.append(max_val)

        if min_val > 0: # 如果一个人所有的标注的最小值，都大于0，说明这个人kindness很大
            cnt_min += 1
        if max_val < 0: # 如果一个人所有的标注的最大值，都小于0，说明这个人kindness很小
            cnt_max += 1
    print(data_name)
    print("cnt_min=",cnt_min)
    print("cnt_max = ",cnt_max)
    print("len(list) = ",len(list))
    print("cnt_min / len = ",cnt_min/len(list))
    print("cnt_max / len = ", cnt_max / len(list))
    ### 画图

    sns.set()
    sns.set_style('white')
    fig, ax = plt.subplots()

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.axvline(0, color='black', linestyle='--')

    sns.kdeplot(min_value_graph_list,fill=False,label = 'min_g',ax=ax)
    sns.kdeplot(max_value_graph_list, fill=False,label = 'max_g',ax=ax)
    sns.kdeplot(ave_value_graph_list,fill=False,label = 'mean_g',ax=ax)
    plt.legend(loc = 'upper left',fontsize =13)
    plt.xlabel('Label - Ground Truth',fontsize = 25)
    plt.ylabel('Density',fontsize = 25)

    # plt.title('Douban Dataset',fontsize = 30)
    plt.savefig('C:/Users/chopin/Desktop/' + 'Label_Distribution_Douban.pdf', bbox_inches='tight')

    # plt.title('GoodReads Dataset',fontsize = 30)
    # plt.savefig('C:/Users/chopin/Desktop/' + 'Label_Distribution_GoodReads.pdf', bbox_inches='tight')

    plt.show()


    ### 画 twitter
    twitterLabelDis()