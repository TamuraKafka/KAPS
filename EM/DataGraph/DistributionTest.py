####################
## 正态分布检测实验  ##
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
        self.t2truth,self.w2tl,self.t2wl,self.h,self.T_Matrix,self.R_Matrix,self.t2lpd,self.a = self.initDataset(truth_file,Y_file,h_file,R_file,T_file,lpd_file,a_file)
        self.w2sum_dic , self.sum2w_dic_list = self.getDataStructure()
        self.w2tl_standard = self.transformY()


    def initDataset(self,truth_file,Y_file,h_file,R_file,T_file,lpd_file,a_file):
        ##  读取数据
        h = {}
        t2truth = {}
        w2tl = {}
        t2wl = {}
        t2lpd = {}
        a = {}
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
        R_Matrix = pd.read_csv(R_file, header=None, sep=',')
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
        # 处理 t2lpd.csv
        with open(lpd_file,'r',encoding='UTF-8-sig') as f4:
            reader = csv.reader(f4)
            next(reader)
            next(reader)
            next(reader)
            next(reader)
            next(reader)
            for row in reader:
                task_id, prediction_label = row
                task_id = int(task_id)
                prediction_label = float(prediction_label)
                t2lpd[task_id] = prediction_label

        # 处理result_a.csv
        with open(a_file, 'r', encoding='UTF-8-sig') as f5:
            reader = csv.reader(f5)
            next(reader)
            next(reader)
            next(reader)
            next(reader)
            next(reader)
            for row in reader:
                worker_id, worker_ability = row
                worker_id = int(worker_id)
                worker_ability = float(worker_ability)
                a[worker_id] = worker_ability


        return t2truth,w2tl,t2wl,h,T_Matrix,R_Matrix,t2lpd,a

    def transformY(self):
        w2tl_standard = {}
        for i in self.w2tl.keys():
            w2tl_standard[i] = {}
            for j in self.w2tl[i].keys():
                w2tl_standard[i][j] = (  self.w2tl[i][j] - self.t2lpd[j] - self.h[i] - np.dot(self.T_Matrix[j],self.R_Matrix[i])  )  / self.a[i]
        return w2tl_standard

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

    '''输出结果中第一个为统计量，第二个为P值（统计量越接近1越表明数据和正态分布拟合的好，P值大于指定的显著性水平(通常是0.05)，接受原假设，认为样本来自服从正态分布的总体）'''
    def getShapiroWilkData(self,worker_id):
        list = []
        for task_id in self.w2tl_standard[worker_id].keys():
            list.append(self.w2tl_standard[worker_id][task_id])
        w_val,p_val = stats.shapiro(list)
        # return w_val,p_val
        return w_val

    # 通过减少 做题数<cut_size的工人 来计算List，并画图
    def getWilkListBySize(self,cut_size):
        ans_list = []
        ans_dic = {}
        cut_worker_num = 0
        sum2w_dic_list = dis_test.sum2w_dic_list
        for task_sum in sum2w_dic_list:
            if task_sum <= cut_size:
                cut_worker_num += len(sum2w_dic_list[task_sum])
                continue

            list1 = sum2w_dic_list[task_sum]
            for i in list1:
                ans_dic[i] = dis_test.getShapiroWilkData(worker_id=i)

        temp = 0
        for key in ans_dic.keys():
            ans_list.append(ans_dic[key])
            if ans_dic[key] <= 0.7:
                temp += 1
        print("cut_size = ",cut_size,"temp = ",temp)
        return ans_dic, ans_list, len(self.w2tl) - cut_worker_num



if __name__ == "__main__":
    ###### Douban 数据集
    # data_name = "DouBanDatasets_RedundancyCut/Douban_202t_197w_5r"  # 测试数据集
    # data_name = "DouBanDatasets/Douban_202t_269266w"  # Douban
    data_name = "GoodReadsDatasets/GoodReads_309t_120415w"            # GoodReads

    ## 文件地址
    h_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_h.csv'
    R_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_R.csv'
    T_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/T.csv'
    Y_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y.csv'
    truth_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/truth.csv'
    lpd_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/t2lpd.csv'
    a_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/result_a.csv'

    # 工人id：做题数  {43374: 143, 22151: 150, 2226: 153, 9660: 155, 2182: 176, 9182: 181, 17094: 184, 9430: 188, 4459: 194, 9791: 199}
    dis_test = DistributionTest(truth_file=truth_file,Y_file=Y_file,h_file=h_file,R_file=R_file,T_file=T_file,lpd_file=lpd_file,a_file=a_file)

    for size in [5]: #[2,5,10,20,50,80,100,120]:
        fig = plt.figure()
        sns.set()

        ans_dic, ans_list,worker_num = dis_test.getWilkListBySize(cut_size=size)


        # sns.kdeplot(ans_list, fill=True, color='g',kernel='gau')
        sns.displot(ans_list,kde=True)
        plt.title('cut_size = ' + str(size) + "  worker = " + str(worker_num))
        plt.show()

    # for i in dis_test.sum2w_dic_list.keys():
    #     print(i,":",len(dis_test.sum2w_dic_list[i]))
