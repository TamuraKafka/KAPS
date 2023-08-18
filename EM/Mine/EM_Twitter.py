import copy
import math
import csv
import multiprocessing
import numpy
import numpy as np
import pandas as pd
from tqdm import *
import time

######################################

class EM:
    def __init__(self, data_file,truth_file,E_file,G_file,T_file,ALPHA, PHI_A, PHI_R, PHI_H,G_Size, k_values,data_name,output_file_path, **kwargs):

        startTime = time.time()

        # 读取数据文件并封装
        t2wl, w2tl, label_list = self.getT2WLandW2TL(data_file)
        self.t2wl = t2wl
        self.w2tl = w2tl
        # label_list是所有工人标注过的k值的去重列表
        # self.label_list = label_list
        self.workers = self.w2tl.keys()
        self.tasks = self.t2wl.keys()
        # 工人数量,从0开始编号
        self.M = len(w2tl)
        # 任务数量,从0开始编号
        self.N = len(t2wl)
        # 人为创造的k值范围
        self.k_values = k_values
        # 取值数量
        self.K = len(k_values)
        self.data_name = data_name
        self.output_file_path = output_file_path

        # 先验值
        self.ALPHA = ALPHA
        self.PHI_A = PHI_A
        self.PHI_R = PHI_R
        self.PHI_H = PHI_H
        self.G_Size = G_Size

        # 读取三个文件并封装数据为三个数据结构
        E_dict,T_Matrix,G_dict,G_reverse_dict = self.getE_T_G_Greverse(E_file=E_file,G_file=G_file,T_file=T_file)
        self.E_dict = E_dict
        self.T_Matrix = T_Matrix
        self.G_dict = G_dict                # G_dict中是关注者为key，被关注者为value.G_dict[i][j] = 1，代表i关注了j。 G_dict中存储了自己关注自己的情况
        self.G_reverse_dict= G_reverse_dict # G_reverse_dict 中被关注者为key，关注者为value， 和 G_dict 相反。也储存了自己关注自己的情况

        # 创建G_part_dict
        G_part_dict, G_reverse_part_dict = self.get_G_part_dict_G_reverse_part_dict()
        self.G_part_dict = G_part_dict
        self.G_reverse_part_dict = G_reverse_part_dict

        # 任务类别数量(T矩阵的列数)
        self.L = T_Matrix.shape[1]

        # 初始化ARHphi
        a,h,phi = self.init_ARH_phi()
        self.a = a
        self.h = h
        self.phi = phi

        # 初始化ln_P_y, ln_P_y_C, ln_P_y_S
        ln_P_y, ln_P_y_C, ln_P_y_S = self.init_ln_P_y()
        self.ln_P_y = ln_P_y
        self.ln_P_y_C = ln_P_y_C
        self.ln_P_y_S = ln_P_y_S

        # 初始化E步的变量
        self.w = {}
        for j in self.tasks:
            self.w[j] = {}
            for k in self.k_values:
                self.w[j][k] = 1.0 / self.K

        # 提前计算所有的w_i_ii
        self.weight_dict = self.calculate_all_weight()
        for key in kwargs:
            setattr(self, key, kwargs[key])

        endTime = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("__init__ 执行结束，执行时间为:" + str(endTime - startTime) + "秒\n")

    ############################################################ EM: 计算E步 ############################################################
    # 计算w_j^k时，k值的范围不是任务j的k范围，而是列表k_values
    def update_wjk(self):
        ##### 先更新计算所有的ln[P(y_ij)],然后下面直接调用即可 #####
        self.calculate_all_ln_P_y()
        start = time.time()
        ##### 计算w_j^k #####
        # 记录w_j^k=0的k值
        k_0_set = set()
        # 遍历每个任务,计算1/w_j^k
        for j in self.tasks:   # for j in tqdm(self.tasks,position=0,desc="正在更新w"):
            for k in self.k_values:
                # 求w_j^k前先判断phi[k]是否有等于0的情况,若有，记录下来并停止此轮计算
                if self.phi[k] == 0:
                    k_0_set.add(k)
                    continue

                flag = 1
                sum = 0.0  # sum为1/w_j^k的和,每个sum值最终应该在5左右
                # 计算λ_k^1 ~ λ_k^K
                for kk in self.k_values:
                    # 求λ前先处理phi[kk]=0的情况,若有，则停止计算当前λ_k^k'，不计算该项的 e^{λ_k^k'}
                    if self.phi[kk] == 0:
                        continue

                    sum_la = 0.0 # 求λ
                    sum_la +=  math.log(self.phi[kk]) - math.log(self.phi[k])
                    # print("la1=",la)
                    for i in self.t2wl[j].keys():
                        # print(self.ln_P_y[i][j][kk],self.ln_P_y[i][j][k])
                        sum_la += (self.ln_P_y[i][j][kk] - self.ln_P_y[i][j][k])
                    # print("la2=", la)
                    # 当算完一个λ时，如果la太大，会导致e^la太大,然后就会导致1/w_j^k太大，那么直接停止运算，直接将w_j^k赋值为0
                    if sum_la >= 30: #如果不加这一行限制，la会到1000多，直接影响下面的式子
                        flag = 0
                        break
                    sum += pow(math.e, sum_la)
                if flag == 1:
                    self.w[j][k] = 1.0 / sum
                else:
                    self.w[j][k] = 0.0

        # 将k_0_set中的w_j^k全部赋值为0.0
        for k in k_0_set:
            for j in self.tasks:
                self.w[j][k] = 0.0

        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("update_wjk 执行结束，执行时间为(不包括计算ln_P_y):" + str(end - start) + "秒\n")

    ############################################################ EM: 计算M步 ############################################################

    # 更新所有k取值的phi_k
    def update_phik(self):
        start = time.time()

        for k in self.k_values:
            sum_wjk = 0.0
            for j in self.tasks:
                sum_wjk += self.w[j][k]
            self.phi[k] = sum_wjk / self.N

        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("update_phik 执行结束，执行时间为:"+str(end - start)+"秒\n")

    # 更新a
    def update_a(self):
        start = time.time()

        # 先计算出T_j * R_i
        # 创建
        # T_j_R_i_dict = {}
        # for j in self.tasks:
        #     T_j_R_i_dict[j] = {}
        # # 计算
        # for j in self.tasks:
        #     for i in self.workers:
        #         T_j_R_i_dict[j][i] = np.dot(self.T_Matrix[j], self.R_Matrix[i])

        for i in self.workers:
            # 计算部分1  常数部分
            A_i = - 1 / (self.PHI_A * self.PHI_A)
            B_i = - self.L / 2
            C_i =  (self.h[i] * self.h[i]) / (self.PHI_H * self.PHI_H)
            sumB1 = 0
            sumC1 = 0
            sumB2 = 0
            sumC2 = 0
            for k in self.k_values:
                for t in self.G_reverse_part_dict[i]:
                    for j in self.w2tl[t].keys():
                        # 计算部分2  三个求和符号部分
                        ytj = self.w2tl[t][j]
                        sumB1 += self.w[j][k] * self.weight_dict[t][i]
                        sumC1 += self.w[j][k] * self.weight_dict[t][i] * ( (ytj - k - self.h[i] ) ** 2 )
                # 计算部分3  两个求和符号部分
                for j in self.w2tl[i].keys():
                    yij = self.w2tl[i][j]
                    sumB2 += self.w[j][k]
                    sumC2 += self.w[j][k] * ((yij - k - self.h[i] )**2)
            B_i -= self.ALPHA * sumB1
            C_i += self.ALPHA * sumC1
            B_i -= (1 - self.ALPHA) * sumB2
            C_i += (1 - self.ALPHA) * sumC2

            x = math.sqrt((- math.sqrt(B_i * B_i - 4 * A_i * C_i) - B_i) / (2 * A_i))
            if x > 0:
                self.a[i] = x

        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("update_a 执行结束，执行时间为:"+str(end-start)+"秒\n")

    # 更新R和h(先算h再算R)
    def update_R_h(self):
        start = time.time()
        L_Vector = np.ones((self.L))

        for i in self.workers:
            # 计算部分1  初始化和常数部分
            # A_i = 1 / (self.PHI_R**2) * np.eye(self.L)
            B_i = np.zeros((self.L, 1))
            C_i = np.zeros((self.L, 1))
            # sumA1 = 0
            sumB1 = 0
            sumC1 = 0
            # sumA2 = 0
            sumB2 = 0
            sumC2 = 0
            for k in self.k_values:
                for t in self.G_reverse_part_dict[i]:
                    for j in self.w2tl[t].keys():
                        # T_j_vector = np.array(self.T_Matrix[j]).reshape(-1, 1)
                        # 计算部分2  三个求和符号部分
                        ytj = self.w2tl[t][j]
                        # sumA1 += self.w[j][k] * self.weight_dict[t][i] * self.T_j_T_j_top_product[j]
                        sumB1 += self.w[j][k] * self.weight_dict[t][i]
                        sumC1 += self.w[j][k] * self.weight_dict[t][i] * (ytj - k)
                for j in self.w2tl[i].keys():
                    yij = self.w2tl[i][j]
                    # sumA2 += self.w[j][k] * self.T_j_T_j_top_product[j]
                    sumB2 += self.w[j][k]
                    sumC2 += self.w[j][k] * (yij - k)

            # A_i += self.ALPHA * sumA1
            B_i -= self.ALPHA * sumB1
            C_i += self.ALPHA * sumC1

            # A_i += (1 - self.ALPHA) * sumA2
            B_i -= (1 - self.ALPHA) * sumB2
            C_i += (1 - self.ALPHA) * sumC2

            self.h[i] = np.array(- C_i / B_i)[0][0]
            # 计算R_i和h_i
            # 求A的伪逆
            # AA_i = np.linalg.pinv(A_i)
            # 减去重复计算
            # AAi_Bi = np.dot(AA_i , B_i)
            # AAi_Ci = np.dot(AA_i , C_i)

            # self.h[i] = -1 * np.dot(L_Vector , AAi_Ci)[0] / np.dot(L_Vector , AAi_Bi)[0]
            # self.R_Matrix[i] = np.array(self.h[i] * AAi_Bi + AAi_Ci).reshape(1,-1)

        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("update_R_h 执行结束，执行时间为:"+str(end - start)+"秒\n")


    ############################################################ 迭代运行 ############################################################
    def run(self, iter):
        projectStartTime = time.time()

        for i in tqdm(range(iter) ,desc="EM迭代中"):

            # 保存上一轮的迭代的数据
            old_a = copy.deepcopy(self.a)
            old_h = copy.deepcopy(self.h)
            old_phi = copy.deepcopy(self.phi)

            # E-step
            self.update_wjk()
            # M-step
            # 在M步，先计算hi，再算Ri，最后再算ai
            self.update_phik()
            self.update_R_h()
            self.update_a()

            # 记录每次迭代的结果
            t2lpd = self.get_accuracy()
            print(t2lpd)
            # 计算每轮数据的差距
            diff_a = 0.0
            diff_h = 0.0
            diff_phi = 0.0
            sum_a_fenzi = 0.0
            sum_h_fenzi = 0.0
            sum_phi_fenzi = 0.0
            sum_a_new = 0.0
            sum_a_old = 0.0
            sum_h_new = 0.0
            sum_h_old = 0.0
            sum_phi_new = 0.0
            sum_phi_old = 0.0
            if i > 0:
                for x in self.workers:
                    # 计算 a 相关部分
                    sum_a_fenzi += math.fabs( self.a[x] - old_a[x] )
                    sum_a_new += math.fabs( self.a[x] )
                    sum_a_old += math.fabs( old_a[x] )
                    # 计算 h 相关部分
                    sum_h_fenzi += math.fabs( self.h[x] - old_h[x] )
                    sum_h_new += math.fabs( self.h[x] )
                    sum_h_old += math.fabs( old_h[x] )


                # 计算 phi 相关部分
                for k in self.k_values:
                    sum_phi_fenzi += math.fabs( self.phi[k] - old_phi[k] )
                    sum_phi_new += math.fabs( self.phi[k] )
                    sum_phi_old += math.fabs( old_phi[k] )

                diff_a = sum_a_fenzi / max( sum_a_new , sum_a_old )
                diff_h = sum_h_fenzi / max( sum_h_new , sum_h_old )
                diff_phi = sum_phi_fenzi / max( sum_phi_new , sum_phi_old )



            # 写入文件
            # 迭代细节记录
            with open('record.txt', 'a', encoding="utf-8") as f:
                f.write("----------------------------------------\n")
                f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +"  第<"+str(i+1)+">次迭代输出："+ "\n")
                f.write("数据名称:" + str(self.data_name) + "\n")
                f.write("该轮先验参数为：PHI_A="+str(self.PHI_A)+"     PHI_H="+str(self.PHI_H)+"     PHI_R="+str(self.PHI_R)+"     ALPHA="+str(self.ALPHA)+"     G_Size="+str(self.G_Size)+"\n")
                f.write("----------------------------------------\n")
                f.write("本轮参数较上一轮迭代的差距：\n")
                f.write("a 差距："+str(diff_a)+"\n")
                f.write("h 差距："+str(diff_h)+"\n")
                f.write("phi 差距："+str(diff_phi)+"\n")


                # f.write("\nt2lpd:\n")
                # cnt = 0
                # for t in sorted(t2lpd):
                #     f.write(f"<{t}:{t2lpd[t]}> ".format(t,t2lpd[t]))
                #     cnt += 1
                #     if cnt % 20 == 0:
                #         f.write("\n")
                # f.write("\n")
                #
                #
                # count2 = 0
                # f.write("前10个w:\n")
                # for j in sorted(self.w.keys()):
                #     sum_w_j_k = 0.0
                #     count2 += 1
                #     # f.write(str(j) + ":   ")
                #     for k, l in self.w[j].items():
                #         sum_w_j_k += self.w[j][k]
                #         # f.write(f"<{k}:{l}> ".format(k, l))
                #     # f.write("\n")
                #     f.write("w" + str(j) + "之和=" + str(sum_w_j_k) + "   ")
                #
                #     if count2 >= 10:
                #         break
                #
                # f.write("\n所有phi:\n")
                # for key, val in self.phi.items():
                #     f.write(f"<{key}:{val}> ".format(key, val))
                # f.write("\n")
                #
                # f.write("\n前30个a:\n")
                # count = 0
                # for key, val in self.a.items():
                #     count += 1
                #     f.write(f"<{key}:{val}> ".format(key, val))
                #     if count >= 30:
                #         break
                # f.write("\n")
                #
                # f.write("\n前30个h:\n")
                # count = 0
                # for key, val in self.h.items():
                #     count += 1
                #     f.write(f"<{key}:{val}> ".format(key, val))
                #     if count >= 30:
                #         break
                # f.write("\n")
                #
                # f.write("\n前5个R:\n")
                # for x in range(5):
                #     f.write("R[" + str(x) + "]=" + str(self.R_Matrix[x]))
                #     f.write("\n")
                f.write("\n----------------------------------------------------------------------------------------------------------------------------------------------------\n")

            # 如果迭代进行到最后一轮，或者变量的更新程度小于目标值，则停止算法迭代，并将最终结果进行记录
            if i == iter - 1 or (i > 0 and diff_a <= 0.00001 and diff_h <= 0.00001 and diff_phi <= 0.00001):
                # 将最终结果记录到result文件
                with open('result.txt', 'a', encoding="utf-8") as f:
                    f.write(time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))
                    f.write("数据名称:" + str(self.data_name) + "\n")
                    f.write("总共迭代次数: "+str(i) +"\n")
                    f.write("该轮先验参数为：PHI_A=" + str(self.PHI_A) + "     PHI_H=" + str(self.PHI_H) + "     PHI_R=" + str(self.PHI_R)+"     G_Size="+str(self.G_Size) + "     ALPHA=" + str(self.ALPHA) + "\n")


                # 将最终结果的所有工人的a，R，h，phi数据都记录到txt文件中
                if self.ALPHA == 0.1 and self.G_Size == 5:
                    # 写入工人能力a
                    with open(self.output_file_path+'result_a.csv' , 'w' , newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows([("dataset",self.data_name),("ALPHA",self.ALPHA),("G_Size",self.G_Size),("Iteration Count",i)])
                        writer.writerow(["worker_id" , "worker_ability"])
                        for row in self.a.items():
                            writer.writerow(row)
                    # 写入工人苛刻度h
                    with open(self.output_file_path+'result_h.csv' , 'w' , newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows([("dataset", self.data_name), ("ALPHA", self.ALPHA), ("G_Size", self.G_Size),("Iteration Count",i)])
                        writer.writerow(["worker_id", "worker_severity"])
                        for row in self.h.items():
                            writer.writerow(row)
                    # 写入预测值
                    with open(self.output_file_path + 't2lpd.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows([("dataset", self.data_name), ("ALPHA", self.ALPHA), ("G_Size", self.G_Size),("Iteration Count", i)])
                        writer.writerow(["task_id", "prediction_label"])
                        for row in t2lpd.items():
                            writer.writerow(row)

                with open('record.txt', 'a', encoding="utf-8") as f:
                    f.write("####################################################################################################################################################\n")
                    f.write("####################################################################################################################################################\n")

                break


        projectEndTime = time.time()
        with open('result.txt', 'a', encoding="utf-8") as f:
            f.write("算法总耗时：" + str(int(projectEndTime - projectStartTime)) + "秒\n")
            f.write("####################################################################################################################################################\n")
        return

    ############################################################ 待调用的函数 ############################################################

    # 计算所有的w_i_ii
    def calculate_all_weight(self):
        start = time.time()

        sum = {} # 工人i受到的所有的影响力之和
        for i in self.workers:
            sum[i] = 0.0
            for t in self.G_part_dict[i]:
                sum[i] += self.E_dict[t]

        weight_dict = {}
        for i in self.workers:
            weight_dict[i] = {}
            for ii in self.G_part_dict[i]:
                weight_dict[i][ii] = self.E_dict[ii] / sum[i]

        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("calculate_all_weight 执行结束，执行时间为:" + str(end - start) + "秒\n")
        return weight_dict

    def calculate_all_ln_P_y(self):
        start = time.time()

        # 先计算出T_j * R_i
        # 创建
        # T_j_R_i_dict = {}
        # for j in self.tasks:
        #     T_j_R_i_dict[j] = {}
        # # 计算
        # for j in self.tasks:
        #     for i in self.workers:
        #         T_j_R_i_dict[j][i] = np.dot(self.T_Matrix[j] , self.R_Matrix[i])

        # 计算所有的 ln[P (l_ij^C = yij |zj = k, θ, D)]
        for k in self.k_values:
            for j in self.tasks:
                for i in self.t2wl[j].keys():
                    y_ij = self.w2tl[i][j]
                    # 只需要计算  i关注了ii  的ln_P_y_C[ii][i][j][k]
                    for ii in self.G_part_dict[i]:
                        self.ln_P_y_C[ii][i][j][k] = - math.log(math.sqrt(2 * math.pi) * self.a[ii]) - ((y_ij - k - self.h[ii] ) ** 2 / (2 * self.a[ii] * self.a[ii]))

        # 计算所有的 ln[P (l_ij^S = yij |zj = k, θ, D)]
        for k in self.k_values:
            for j in self.tasks:
                for i in self.t2wl[j].keys():
                    sum_yijs = 0.0
                    # 遍历每一个i关注的人ii
                    for ii in self.G_part_dict[i]:
                        # calCount += 1
                        sum_yijs += self.weight_dict[i][ii] * self.ln_P_y_C[ii][i][j][k]
                    self.ln_P_y_S[i][j][k] = sum_yijs

                    # 当计算好所有的ln[P (l_ij^C = yij |zj = k, θ, D)] 和 ln[P (l_ij^S = yij |zj = k, θ, D)]，就可以计算ln [P(y_ij|z_j=k)]了
                    self.ln_P_y[i][j][k] = self.ALPHA * self.ln_P_y_S[i][j][k] + (1 - self.ALPHA) * self.ln_P_y_C[i][i][j][k]

        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("calculate_all_ln_P_y 执行结束，执行时间为:" + str(end - start) + "秒\n")

    # def calculate_all_T_j_T_j_top(self):
    #     start = time.time()
    #     T_j_T_j_top_product = np.zeros((self.N,self.L,self.L))
    #     for j in self.tasks:
    #         T_j_T_j_top_product[j] = np.dot(np.array(self.T_Matrix[j]).reshape(-1, 1), np.array([self.T_Matrix[j]]))
    #         # T_j_T_j_top_product[j] = np.matmul(np.array(self.T_Matrix[j]).reshape(-1, 1)  ,  np.array([self.T_Matrix[j]]))
    #     end = time.time()
    #     with open('record.txt', 'a', encoding="utf-8") as f:
    #         f.write("calculate_all_T_j_T_j_top 执行结束，执行时间为:" + str(end - start) + "秒\n")
    #
    #     return T_j_T_j_top_product

    ############################################################ 初始化值 ############################################################
    # 初始化ln[P(y_ij)],ln[P(y_ij^C)],ln[P(y_ij^S)]
    # ln_P_y = {i:{j:{k:概率值}}}
    def init_ln_P_y(self):
        # 只需要计算  i关注了ii  的ln_P_y_C[ii][i][j][k]

        ln_P_y = {}
        ln_P_y_S = {}
        for i in self.workers:
            ln_P_y[i] = {}
            ln_P_y_S[i] = {}
            for j in self.w2tl[i].keys():
                ln_P_y[i][j] = {}
                ln_P_y_S[i][j] = {}

        # for i in self.workers:
        #     for j in self.w2tl[i].keys():
        #         if i not in ln_P_y.keys():
        #             ln_P_y[i] = {}
        #         if i not in ln_P_y_S.keys():
        #             ln_P_y_S[i] = {}
        #         if j not in ln_P_y[i].keys():
        #             ln_P_y[i][j] = {}
        #         if j not in ln_P_y_S[i].keys():
        #             ln_P_y_S[i][j] = {}

        ln_P_y_C = {}
        for ii in self.workers:
            ln_P_y_C[ii] = {}
            for i in self.G_reverse_part_dict[ii]:
                ln_P_y_C[ii][i] = {}
                for j in self.w2tl[i].keys():
                    ln_P_y_C[ii][i][j] = {}

        # for ii in self.workers:
        #     for i in self.G_reverse_part_dict[ii]:
        #         for j in self.w2tl[i].keys():
        #             if ii not in ln_P_y_C.keys():
        #                 ln_P_y_C[ii] = {}
        #             if i not in ln_P_y_C[ii].keys():
        #                 ln_P_y_C[ii][i] = {}
        #             if j not in ln_P_y_C[ii][i].keys():
        #                 ln_P_y_C[ii][i][j] = {}

        return ln_P_y,ln_P_y_C,ln_P_y_S

    # 初始化a,R,H,phi

    def init_ARH_phi(self):

        # 工人苛刻度初始化，初始值为0
        # 工人能力初始化，初始值为1
        a = {}
        h = {}
        for i in range(self.M):
            h[i] = 0.0
            a[i] = 1.0

        # 工人偏好初始化为全0,下标从0开始(M行L列)
        # R_Matrix = np.zeros((self.M,self.L))

        # 初始化phi_k，初始值为1/K
        phi = {}
        for label in self.k_values:                #for label in self.label_list:
            phi[label] = 1.0 / len(self.k_values)

        return a,h,phi

    ########################################################################################################################
    # The above is the EM method (a class)
    # The following are several external functions
    ########################################################################################################################
    def get_accuracy(self):
        # 最终预测的真值（和ground truth进行对比计算准确率）
        t2lpd = {}

        # 按权重计算 t2lpd
        for j in self.w.keys():
            lpd = 0.0
            for k in self.w[j].keys():
                lpd += k * self.w[j][k]
            t2lpd[j] = lpd

        return t2lpd

    ############################################################ 读取文件并封装数据 ############################################################

    def getE_T_G_Greverse(self,E_file,G_file,T_file):

        # 将类别属性T文件转化为矩阵
        T_Matrix = pd.read_csv(T_file, header=None, sep=',', skiprows=1)
        T_Matrix = np.array(T_Matrix)

        # 将社交关系G文件转化为字典
        G_dict = {}
        G_reverse_dict = {}
        with open(G_file,'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                id_1 , id_2 = row
                worker_id_1 = int(id_1)
                worker_id_2 = int(id_2)
                # 数据集中，工人2关注工人1。则以工人2为key，工人1为value
                if worker_id_2 not in G_dict.keys():
                    G_dict[worker_id_2] = []
                # 避免数据集里有重复的数据
                if worker_id_1 not in G_dict[worker_id_2]:
                    G_dict[worker_id_2].append(worker_id_1)

                if worker_id_1 not in G_reverse_dict.keys():
                    G_reverse_dict[worker_id_1] = []
                # 避免数据集里有重复的数据
                if worker_id_2 not in G_reverse_dict[worker_id_1]:
                    G_reverse_dict[worker_id_1].append(worker_id_2)

        # 把自己关注自己的情况加进去
        for i in self.workers:
            if i not in G_dict.keys():
                G_dict[i] = []
            if i not in G_dict[i]:
                G_dict[i].append(i)
            if i not in G_reverse_dict.keys():
                G_reverse_dict[i] = []
            if i not in G_reverse_dict[i]:
                G_reverse_dict[i].append(i)

        # 将社交影响E转化为字典
        E_dict = {}
        with open(E_file,'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                i,e = row
                E_dict[int(i)] = float(float(e))

        return E_dict,T_Matrix,G_dict,G_reverse_dict

    def get_G_part_dict_G_reverse_part_dict(self):
        G_part_dict = {}
        G_reverse_part_dict = {}
        tempList = []

        # 将G按照影响力从大到小排序(结构体排序)
        for i in self.G_dict.keys():
            tempList.clear()
            if i not in G_part_dict.keys():
                G_part_dict[i] = []
            if i not in G_reverse_part_dict.keys():
                G_reverse_part_dict[i] = []
            for v in self.G_dict[i]:
                tempList.append([v,self.E_dict[v]])
            tempList.sort(key=lambda x:x[1] , reverse=True)

            count = 0
            for sortVal in tempList:
                count += 1
                value1 = int(sortVal[0])
                if value1 not in G_reverse_part_dict.keys():
                    G_reverse_part_dict[value1] = []

                if i != value1: # 把除自己之外的人加进去
                    # 生成 G_part_dict
                    G_part_dict[i].append(value1)
                    # 同时生成 G_reverse_part_dict
                    G_reverse_part_dict[value1].append(i)

                # 控制规模
                if count == self.G_Size:
                    break
            # 把自己关注自己的情况也要加进去
            if i not in G_part_dict[i]:
                G_part_dict[i].append(i)
            if i not in G_reverse_part_dict[i]:
                G_reverse_part_dict[i].append(i)

        return G_part_dict , G_reverse_part_dict


    def getT2WLandW2TL(self, data_file):
        t2wl = {}
        w2tl = {}
        label_list = []

        f = open(data_file, 'r',encoding='utf-8')
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            worker_id, task_id, label = row
            worker_id = int(worker_id)
            task_id = int(task_id)
            label = float(label)
            if task_id not in t2wl:
                t2wl[task_id] = {}
            t2wl[task_id].setdefault(worker_id,label)
            #t2wl[task_id].append([worker_id, label])

            if worker_id not in w2tl:
                w2tl[worker_id] = {}
            w2tl[worker_id].setdefault(task_id,label)
            #w2tl[worker_id].append([task_id, label])

            if label not in label_list:
                label_list.append(label)

        return t2wl, w2tl, label_list

def runEm(data_file, truth_file, E_file, G_file, T_file, ALPHA, PHI_A, PHI_R, PHI_H,G_Size, k_values, data_name,output_file_path):
    em = EM(data_file=data_file, truth_file=truth_file, E_file=E_file, G_file=G_file, T_file=T_file, ALPHA=ALPHA, PHI_A=PHI_A, PHI_R=PHI_R, PHI_H=PHI_H,G_Size=G_Size, k_values=k_values, data_name=data_name,output_file_path=output_file_path)
    em.run(iter = 200)

if __name__ == "__main__":
    ############################################################ 设置参数 ############################################################
    PHI_A = 1.5
    PHI_R = 1.0
    PHI_H = 1.0
    ALPHA = 0.1 # 目前来看是 最优参数
    G_Size = 5

    # 根据不用数据集来设置不同的 K值范围
    # DouBan/GoodReads数据集k值：2.1~~5.0共30个k值，可适当减去前半部分的k值
    k_values = []
    for i in range(0, 101):
        k_values.append(i / 100)
    # print(k_values)
    ############################################################   数据集  ###########################################################

    ####### Twitter 数据集
    data_name = "TwitterDatasets/Twitter_394t_46486w"

    output_file_path = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/'
    data_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/Y.csv'
    truth_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/truth.csv'
    G_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/G.csv'
    T_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/T.csv'
    E_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/E.csv'

    ################################################################ 迭代运行并显示结果 ################################################################
    ###################### 单进程 单参数 迭代  ######################
    em = EM(data_file=data_file, truth_file=truth_file, E_file=E_file, G_file=G_file, T_file=T_file, ALPHA=ALPHA, PHI_A=PHI_A, PHI_R=PHI_R, PHI_H=PHI_H,G_Size=G_Size, k_values=k_values, data_name=data_name,output_file_path=output_file_path)

    em.run(iter=200)

