import copy
import math
import csv
import multiprocessing
import numpy
import numpy as np
import pandas as pd
from tqdm import *
import time
import os
import re

######################################

class EM:
    def __init__(self, data_file,truth_file,E_file,G_file,T_file,ALPHA, PHI_A, PHI_R, PHI_H,G_Size, k_values,data_name,output_file_path, **kwargs):

        startTime = time.time()

        # 读取数据文件并封装
        t2wl, w2tl, label_list = self.getT2WLandW2TL(data_file)
        self.t2truth = self.getT2Truth(truth_file=truth_file)
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
        a,R_Matrix,h,phi = self.init_ARH_phi()
        self.a = a
        self.R_Matrix = R_Matrix
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

        # 提前计算T_j * T_j^T
        self.T_j_T_j_top_product = self.calculate_all_T_j_T_j_top()
        # 提前计算所有的w_i_ii
        self.weight_dict = self.calculate_all_weight()
        for key in kwargs:
            setattr(self, key, kwargs[key])

        endTime = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("__init__ 执行结束，执行时间为:" + str(endTime - startTime) + "秒\n")

    ############################################################ KAPS: 计算E步 ############################################################
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

    ############################################################ KAPS: 计算M步 ############################################################

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

    # 更新a `
    def update_a(self):
        start = time.time()

        # 先计算出T_j * R_i
        # 创建
        T_j_R_i_dict = {}
        for j in self.tasks:
            T_j_R_i_dict[j] = {}
        # 计算
        for j in self.tasks:
            for i in self.workers:
                T_j_R_i_dict[j][i] = np.dot(self.T_Matrix[j], self.R_Matrix[i])

        for i in self.workers:
            # 计算部分1  常数部分
            A_i = - 1 / (self.PHI_A * self.PHI_A)
            B_i = - self.L
            C_i = np.dot(self.R_Matrix[i], self.R_Matrix[i]) / (self.PHI_R * self.PHI_R) + (self.h[i] * self.h[i]) / (self.PHI_H * self.PHI_H)
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
                        sumC1 += self.w[j][k] * self.weight_dict[t][i] * ( (ytj - k - self.h[i] - T_j_R_i_dict[j][i] ) ** 2 )
                # 计算部分3  两个求和符号部分
                for j in self.w2tl[i].keys():
                    yij = self.w2tl[i][j]
                    sumB2 += self.w[j][k]
                    sumC2 += self.w[j][k] * ((yij - k - self.h[i] - T_j_R_i_dict[j][i])**2)
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
            A_i = 1 / (self.PHI_R ** 2) * np.eye(self.L)
            B_i = np.zeros((self.L, 1))
            C_i = np.zeros((self.L, 1))
            sumA1 = 0
            sumB1 = 0
            sumC1 = 0
            sumA2 = 0
            sumB2 = 0
            sumC2 = 0
            for k in self.k_values:
                for t in self.G_reverse_part_dict[i]:
                    for j in self.w2tl[t].keys():
                        T_j_vector = np.array(self.T_Matrix[j]).reshape(-1, 1)
                        # 计算部分2  三个求和符号部分
                        ytj = self.w2tl[t][j]
                        sumA1 += self.w[j][k] * self.weight_dict[t][i] * self.T_j_T_j_top_product[j]
                        sumB1 += self.w[j][k] * self.weight_dict[t][i] * T_j_vector
                        sumC1 += self.w[j][k] * self.weight_dict[t][i] * (ytj - k) * T_j_vector
                for j in self.w2tl[i].keys():
                    T_j_vector = np.array(self.T_Matrix[j]).reshape(-1, 1)
                    yij = self.w2tl[i][j]
                    sumA2 += self.w[j][k] * self.T_j_T_j_top_product[j]
                    sumB2 += self.w[j][k] * T_j_vector
                    sumC2 += self.w[j][k] * (yij - k) * T_j_vector

            A_i += self.ALPHA * sumA1
            B_i -= self.ALPHA * sumB1
            C_i += self.ALPHA * sumC1

            A_i += (1 - self.ALPHA) * sumA2
            B_i -= (1 - self.ALPHA) * sumB2
            C_i += (1 - self.ALPHA) * sumC2

            # 计算R_i和h_i
            # 求A的伪逆
            AA_i = np.linalg.pinv(A_i)
            # 减去重复计算
            AAi_Bi = np.dot(AA_i, B_i)
            AAi_Ci = np.dot(AA_i, C_i)

            self.h[i] = -1 * np.dot(L_Vector, AAi_Ci)[0] / np.dot(L_Vector, AAi_Bi)[0]
            self.R_Matrix[i] = np.array(self.h[i] * AAi_Bi + AAi_Ci).reshape(1, -1)

        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("update_R_h 执行结束，执行时间为:" + str(end - start) + "秒\n")

    ############################################################ 迭代运行 ############################################################
    def run(self, iter):
        projectStartTime = time.time()

        for i in tqdm(range(iter) ,desc="EM迭代中"):

            # 保存上一轮的迭代的数据
            old_a = copy.deepcopy(self.a)
            old_h = copy.deepcopy(self.h)
            old_R_Matrix = copy.deepcopy(self.R_Matrix)
            old_phi = copy.deepcopy(self.phi)

            # E-step
            self.update_wjk()
            # M-step
            # 在M步，先计算hi，再算Ri，最后再算ai
            self.update_phik()
            self.update_R_h()
            self.update_a()

            # 记录每次迭代的结果
            t2lpd, RMSE , MAE = self.get_accuracy()

            # 计算每轮数据的差距
            diff_a = 0.0
            diff_h = 0.0
            diff_phi = 0.0
            diff_R = 0.0
            sum_a_fenzi = 0.0
            sum_h_fenzi = 0.0
            sum_R_fenzi = 0.0
            sum_phi_fenzi = 0.0
            sum_a_new = 0.0
            sum_a_old = 0.0
            sum_h_new = 0.0
            sum_h_old = 0.0
            sum_R_new = 0.0
            sum_R_old = 0.0
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
                    # 计算 R 相关部分
                    s1 = 0.0
                    s2 = 0.0
                    s3 = 0.0
                    for y in range(self.L):
                        s1 += math.fabs( self.R_Matrix[x][y] - old_R_Matrix[x][y] )
                        s2 += math.fabs( self.R_Matrix[x][y] )
                        s3 += math.fabs( old_R_Matrix[x][y] )
                    sum_R_fenzi += s1
                    sum_R_new += s2
                    sum_R_old += s3

                # 计算 phi 相关部分
                for k in self.k_values:
                    sum_phi_fenzi += math.fabs( self.phi[k] - old_phi[k] )
                    sum_phi_new += math.fabs( self.phi[k] )
                    sum_phi_old += math.fabs( old_phi[k] )

                diff_a = sum_a_fenzi / max( sum_a_new , sum_a_old )
                diff_h = sum_h_fenzi / max( sum_h_new , sum_h_old )
                diff_R = sum_R_fenzi / max( sum_R_new , sum_R_old )
                diff_phi = sum_phi_fenzi / max( sum_phi_new , sum_phi_old )



            # 写入文件
            # 迭代细节记录
            with open('record.txt', 'a', encoding="utf-8") as f:
                f.write("----------------------------------------\n")
                f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +"  第<"+str(i+1)+">次迭代输出："+ "\n")
                f.write("数据名称:" + str(self.data_name) + "\n")
                f.write("该轮先验参数为：PHI_A="+str(self.PHI_A)+"     PHI_H="+str(self.PHI_H)+"     PHI_R="+str(self.PHI_R)+"     ALPHA="+str(self.ALPHA)+"     G_Size="+str(self.G_Size)+"\n")
                f.write("RMSE:" + str(RMSE) + "\n")
                f.write("MAE:" + str(MAE) + "\n")
                f.write("----------------------------------------\n")
                f.write("本轮参数较上一轮迭代的差距：\n")
                f.write("a 差距："+str(diff_a)+"\n")
                f.write("R 差距：" + str(diff_R) + "\n")
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
            if i == iter - 1 or (i > 0 and diff_a <= 0.00001 and diff_h <= 0.00001 and diff_phi <= 0.00001 and diff_R <= 0.00001):
                # 将最终结果记录到result文件
                with open('result.txt', 'a', encoding="utf-8") as f:
                    f.write(time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))
                    f.write("数据名称:" + str(self.data_name) + "\n")
                    f.write("总共迭代次数: "+str(i) +"\n")
                    f.write("该轮先验参数为：PHI_A=" + str(self.PHI_A) + "     PHI_H=" + str(self.PHI_H) + "     PHI_R=" + str(self.PHI_R)+"     G_Size="+str(self.G_Size) + "     ALPHA=" + str(self.ALPHA) + "\n")
                    f.write("RMSE:" + str(RMSE) + "\n")
                    f.write("MAE:" + str(MAE) + "\n")

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
                    # 写入工人偏好向量R
                    numpy.savetxt(self.output_file_path+'result_R.csv',self.R_Matrix,delimiter=',')
                    # 写入phi_k
                    with open(self.output_file_path+'result_phi.csv' , 'w' , newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows([("dataset", self.data_name), ("ALPHA", self.ALPHA), ("G_Size", self.G_Size),("Iteration Count",i)])
                        writer.writerow(["k_value", "phi_k"])
                        for row in self.phi.items():
                            writer.writerow(row)
                    # 写入预测值
                    with open(self.output_file_path+'t2lpd.csv' , 'w' , newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows([("dataset", self.data_name), ("ALPHA", self.ALPHA), ("G_Size", self.G_Size),("Iteration Count",i)])
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

        return RMSE , MAE

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
        T_j_R_i_dict = {}
        for j in self.tasks:
            T_j_R_i_dict[j] = {}
        # 计算
        for j in self.tasks:
            for i in self.workers:
                T_j_R_i_dict[j][i] = np.dot(self.T_Matrix[j] , self.R_Matrix[i])

        # 计算所有的 ln[P (l_ij^C = yij |zj = k, θ, D)]
        for k in self.k_values:
            for j in self.tasks:
                for i in self.t2wl[j].keys():
                    y_ij = self.w2tl[i][j]
                    # 只需要计算  i关注了ii  的ln_P_y_C[ii][i][j][k]
                    for ii in self.G_part_dict[i]:
                        self.ln_P_y_C[ii][i][j][k] = - math.log(math.sqrt(2 * math.pi) * self.a[ii]) - ((y_ij - k - self.h[ii] - T_j_R_i_dict[j][ii]) ** 2 / (2 * self.a[ii] * self.a[ii]))

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

    def calculate_all_T_j_T_j_top(self):
        start = time.time()
        T_j_T_j_top_product = np.zeros((self.N,self.L,self.L))
        for j in self.tasks:
            T_j_T_j_top_product[j] = np.dot(np.array(self.T_Matrix[j]).reshape(-1, 1), np.array([self.T_Matrix[j]]))
            # T_j_T_j_top_product[j] = np.matmul(np.array(self.T_Matrix[j]).reshape(-1, 1)  ,  np.array([self.T_Matrix[j]]))
        end = time.time()
        with open('record.txt', 'a', encoding="utf-8") as f:
            f.write("calculate_all_T_j_T_j_top 执行结束，执行时间为:" + str(end - start) + "秒\n")

        return T_j_T_j_top_product

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

        ln_P_y_C = {}
        for ii in self.workers:
            ln_P_y_C[ii] = {}
            for i in self.G_reverse_part_dict[ii]:
                ln_P_y_C[ii][i] = {}
                for j in self.w2tl[i].keys():
                    ln_P_y_C[ii][i][j] = {}

        return ln_P_y,ln_P_y_C,ln_P_y_S

    # 初始化a,R,H,phi

    def init_ARH_phi(self):

        # 工人苛刻度初始化，初始值为0
        # 工人能力初始化，初始值为1
        a = {}
        h = {}
        for i in self.workers:
            h[i] = 0.0
            a[i] = 1.0

        # 工人偏好初始化为全0,下标从0开始(M行L列)
        # 按照工人的最大编号来设置矩阵行数，因为有的数据集的Y文件的工人编号是乱序的
        max_worker_id = max(self.workers)
        R_Matrix = np.zeros((max_worker_id + 2 , self.L))

        # 初始化phi_k，初始值为1/K
        phi = {}
        for label in self.k_values:                #for label in self.label_list:
            phi[label] = 1.0 / len(self.k_values)

        return a,R_Matrix,h,phi

    ########################################################################################################################
    # The above is the KAPS method (a class)
    # The following are several external functions
    ########################################################################################################################
    def get_accuracy(self):
        # 最终预测的真值（和ground truth进行对比计算准确率）
        t2lpd = {}
        t2truth = self.t2truth
        count1 = count2 = 0

        # 按权重计算 t2lpd
        for j in self.w.keys():
            lpd = 0.0
            for k in self.w[j].keys():
                lpd += k * self.w[j][k]
            t2lpd[j] = lpd

        # 计算RMSE
        sum_RMSE = 0.0
        # 计算MAE
        sum_MAE = 0.0
        for j in self.tasks:
            sum_RMSE += (t2lpd[j] - t2truth[j]) ** 2
            sum_MAE += math.fabs(t2lpd[j] - t2truth[j])
        RMSE = math.sqrt(sum_RMSE / self.N)
        MAE = sum_MAE / self.N

        return t2lpd, RMSE , MAE

    ############################################################ 读取文件并封装数据 ############################################################
    def getE_T_G_Greverse(self,E_file,G_file,T_file):

        # 将类别属性T文件转化为矩阵
        T_Matrix = pd.read_csv(T_file, header=None, sep=',', skiprows=1)
        T_Matrix = np.array(T_Matrix)

        # 将社交关系G文件转化为字典
        G_dict = {}
        G_reverse_dict = {}
        with open(G_file,'r',encoding='UTF-8-sig') as f:
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
        with open(E_file,'r',encoding='UTF-8-sig') as f:
            reader = csv.reader(f)
            # 按照数据集结构是否跳过
            next(reader)
            for row in reader:
                if len(row) == 0:
                    continue
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

    def getT2Truth(self,truth_file):
        t2truth = {}
        f = open(truth_file, 'r',encoding='UTF-8-sig')
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            task_id, truth = row
            task_id = int(task_id)
            truth = float(truth)

            t2truth[task_id] = truth
        return t2truth

    def getT2WLandW2TL(self, data_file):
        t2wl = {}
        w2tl = {}
        label_list = []

        f = open(data_file, 'r',encoding='UTF-8-sig')
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            worker_id, task_id, label = row
            worker_id = int(worker_id)
            task_id = int(task_id)
            label = int(label)
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
    result = {}
    em = EM(data_file=data_file, truth_file=truth_file, E_file=E_file, G_file=G_file, T_file=T_file, ALPHA=ALPHA, PHI_A=PHI_A, PHI_R=PHI_R, PHI_H=PHI_H,G_Size=G_Size, k_values=k_values, data_name=data_name,output_file_path=output_file_path)
    RMSE, MAE = em.run(iter = 200)
    result["data_name"] = data_name
    result["RMSE"] = RMSE
    result["MAE"] = MAE
    result["ALPHA"] = ALPHA
    result["G_Size"] = G_Size
    return  result


if __name__ == "__main__":
    ############################################################ 设置参数 ############################################################
    PHI_A = 1.5
    PHI_R = 1.0
    PHI_H = 1.0
    ALPHA = 0.1
    G_Size = 5

    # 根据不用数据集来设置不同的 K值范围
    # DouBan/GoodReads数据集k值：2.1~~5.0共30个k值，可适当减去前半部分的k值
    k_values = []
    for i in range(0, 51):
        k_values.append(i / 10)

    ############################################################   数据集  ###########################################################
    # data_name = "DouBanDatasets_RedundancyCut/Douban_202t_197w_5r"  # 测试数据
    ####### 旧数据集
    # data_name = "DouBanDatasets/Douban_202t_269266w" # douban 完整
    # data_name = "GoodReadsDatasets/GoodReads_309t_120415w" # GoodReads 完整

    # data_name = "DouBanDatasets/Douban_202t_2692w" # Douban 分割数据
    # data_name = "GoodReadsDatasets/GoodReads_309t_96302w" # GoodReads 分割数据

    ###### 冗余度  数据集
    # data_name = "DouBanDatasets_RedundancyCut/Douban_202t_197w_5r"
    # data_name = "GoodReadsDatasets_RedundancyCut/GoodReads_309t_245w_3r"


    # data_name = "GoodReadsDatasets_SWTest/GoodReads_5702_309t_0%"
    #
    # output_file_path = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/'
    # data_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/Y.csv'
    # truth_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/truth.csv'
    # G_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/G.csv'
    # T_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/T.csv'
    # E_file = '../../datasets/QuantitativeCrowdsourcing/'+data_name+'/E.csv'

    ################################################################ 迭代运行并显示结果 ################################################################
    ###################### 单进程 单参数 迭代  ######################

    # em = KAPS(data_file=data_file, truth_file=truth_file, E_file=E_file, G_file=G_file, T_file=T_file, ALPHA=ALPHA, PHI_A=PHI_A, PHI_R=PHI_R, PHI_H=PHI_H,G_Size=G_Size, k_values=k_values, data_name=data_name,output_file_path=output_file_path)
    #
    # em.run(iter=200)

    ###################### 单进程 多参数 迭代  ######################
    # # data_name = "GoodReadsDatasets/GoodReads_309t_120415w"
    # DataNameValues = ["DouBanDatasets/Douban_202t_269266w", "GoodReadsDatasets/GoodReads_309t_120415w"]
    # # alphaValues = [0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # for x in DataNameValues:
    #     G_Size = 5
    #     ALPHA = 0.1
    #     data_name = x
    #     output_file_path = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/'
    #     data_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y.csv'
    #     truth_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/truth.csv'
    #     G_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/G.csv'
    #     T_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/T.csv'
    #     E_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/E.csv'
    #     em = KAPS(data_file=data_file, truth_file=truth_file, E_file=E_file, G_file=G_file, T_file=T_file, ALPHA=ALPHA, PHI_A=PHI_A, PHI_R=PHI_R, PHI_H=PHI_H,G_Size=G_Size, k_values=k_values, data_name=data_name,output_file_path=output_file_path)
    #     em.run(iter=200)

    ###################### 多进程 多参数 迭代  ######################

    ############# 数据集 #############

    # DataNameValues = ["DouBanDatasets_RedundancyCut/Douban_202t_110w_3r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_145w_4r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_197w_5r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_217w_6r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_271w_7r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_332w_8r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_380w_9r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_432w_10r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_483w_11r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_526w_12r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_594w_13r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_640w_14r",
    #                   "DouBanDatasets_RedundancyCut/Douban_202t_681w_15r"]


    # DataNameValues = ["GoodReadsDatasets_RedundancyCut/GoodReads_309t_245w_3r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_329w_4r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_406w_5r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_490w_6r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_583w_7r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_674w_8r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_760w_9r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_854w_10r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_943w_11r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_1029w_12r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_1126w_13r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_1227w_14r",
    #                   "GoodReadsDatasets_RedundancyCut/GoodReads_309t_1312w_15r"]

    # DataNameValues = ["DouBanDatasets_SWTest/Douban_3635w_50%",
    #                   "DouBanDatasets_SWTest/Douban_3998w_45%",
    #                   "DouBanDatasets_SWTest/Douban_4362w_40%",
    #                   "DouBanDatasets_SWTest/Douban_4725w_35%",
    #                   "DouBanDatasets_SWTest/Douban_5089w_30%",
    #                   "DouBanDatasets_SWTest/Douban_5452w_25%",
    #                   "DouBanDatasets_SWTest/Douban_5816w_20%",
    #                   "DouBanDatasets_SWTest/Douban_6179w_15%",
    #                   "DouBanDatasets_SWTest/Douban_6543w_10%",
    #                   "DouBanDatasets_SWTest/Douban_6906w_5%",
    #                   "DouBanDatasets_SWTest/Douban_7271w_0%"]


    DataNameValues = ["GoodReadsDatasets_SWTest/GoodReads_5702_309t_0%",
                      "GoodReadsDatasets_SWTest/GoodReads_5415_309t_5%",
                      "GoodReadsDatasets_SWTest/GoodReads_5130_309t_10%",
                      "GoodReadsDatasets_SWTest/GoodReads_4845_309t_15%",
                      "GoodReadsDatasets_SWTest/GoodReads_4560_309t_20%",
                      "GoodReadsDatasets_SWTest/GoodReads_4275_309t_25%",
                      "GoodReadsDatasets_SWTest/GoodReads_3990_309t_30%",
                      "GoodReadsDatasets_SWTest/GoodReads_3705_309t_35%",
                      "GoodReadsDatasets_SWTest/GoodReads_3420_308t_40%",
                      "GoodReadsDatasets_SWTest/GoodReads_3135_308t_45%",
                      "GoodReadsDatasets_SWTest/GoodReads_2850_308t_50%"]

    # DataNameValues = ["DouBanDatasets/Douban_202t_269266w","GoodReadsDatasets/GoodReads_309t_120415w"]  # douban 完整  # GoodReads 完整

    dataset_result = []
    process_pool = multiprocessing.Pool(len(DataNameValues)+1)
    for x in DataNameValues:
        ALPHA = 0.1
        G_Size = 5
        data_name = x
        output_file_path = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/'
        data_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y.csv'
        truth_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/truth.csv'
        G_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/G.csv'
        T_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/T.csv'
        E_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/E.csv'
        dataset_result.append(process_pool.apply_async(runEm, args=(data_file, truth_file, E_file, G_file, T_file, ALPHA, PHI_A, PHI_R, PHI_H,G_Size, k_values, data_name,output_file_path)))
    process_pool.close()
    process_pool.join()

    # 整理数据
    show_data_dict = {}
    for res in dataset_result:
        dic = res.get()
        show_data_dict[dic["data_name"]] = [dic["RMSE"], dic["MAE"]]
    # 打印数据
    data_name_list = []
    RMSE_list = []
    MAE_list = []
    s_key = sorted(show_data_dict.keys(),reverse=True)
    for key1 in s_key:
        data_name_list.append(key1)
        RMSE_list.append(show_data_dict[key1][0])
        MAE_list.append(show_data_dict[key1][1])
    print("data_name = ", data_name_list)
    print("RMSE = ", RMSE_list)
    print("MAE = ", MAE_list)
    with open('result.txt', 'a', encoding="utf-8") as f:
        f.write("data_name_list = " + str(data_name_list) + "\n")
        f.write("RMSE = " + str(RMSE_list) + "\n")
        f.write("MAE = " + str(MAE_list) + "\n")

    ############# ALPHA #############

    # alpha_result = []
    # alphaValues = [0,0.1]#,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # process_pool = multiprocessing.Pool(len(alphaValues)+1)
    # for x in alphaValues:
    #     ALPHA = x
    #     alpha_result.append(process_pool.apply_async(runEm, args=(data_file, truth_file, E_file, G_file, T_file, ALPHA, PHI_A, PHI_R, PHI_H,G_Size, k_values, data_name,output_file_path)))
    # process_pool.close()
    # process_pool.join()
    #
    # # 整理数据
    # show_data_dict = {}
    # for res in alpha_result:
    #     dic = res.get()
    #     show_data_dict[dic["ALPHA"]] = [dic["RMSE"],dic["MAE"]]
    #
    # # 打印数据
    # ALPHA_list = []
    # RMSE_list = []
    # MAE_list = []
    # s_key = sorted(show_data_dict.keys())
    # for key1 in s_key:
    #     ALPHA_list.append(key1)
    #     RMSE_list.append(show_data_dict[key1][0])
    #     MAE_list.append(show_data_dict[key1][1])
    # print("ALPHA = ",ALPHA_list)
    # print("RMSE = ",RMSE_list)
    # print("MAE = ",MAE_list)
    # with open('result.txt', 'a', encoding="utf-8") as f:
    #     f.write("ALPHA = "+str(ALPHA_list)+"\n")
    #     f.write("RMSE = "+str(RMSE_list)+"\n")
    #     f.write("MAE = "+str(MAE_list)+"\n")


    ############ G_Size #############
    # G_SizeValues = [3,4,5,6,7,8,9,10]
    # G_SizeValues = [5,10,20,40,80,160,320]
    # process_pool = multiprocessing.Pool(len(G_SizeValues) + 1)
    # for y in G_SizeValues:
    #     ALPHA = 0.1
    #     G_Size = y
    #     process_pool.apply_async(runEm, args=(data_file, truth_file, E_file, G_file, T_file, ALPHA, PHI_A, PHI_R, PHI_H, G_Size, k_values, data_name,output_file_path))
    # process_pool.close()
    # process_pool.join()
###############################################################################################################################################################################




















    ###################### 跑平均值与标准差  ######################
    # # path = "../../datasets/QuantitativeCrowdsourcing/" + "DouBanDatasets_RedundancyCut_Group"
    # path = "../../datasets/QuantitativeCrowdsourcing/" + "GoodReadsDatasets_RedundancyCut_Group"
    # final_result = {}
    # result = {}
    # for filename1 in tqdm(os.listdir(path),desc="多进程运行中"):
    #     final_result[filename1] = {}
    #     result[filename1] = {}
    #
    #     # 计算每个最小单位的数据集的结果，并储存
    #     temp_result = []
    #     process_pool = multiprocessing.Pool(11)
    #     for filename2 in os.listdir(path + "/" + filename1):
    #         file_path = path + "/" + filename1 + "/" + filename2
    #         data_name = filename2
    #         output_file_path = file_path + '/'
    #         data_file = file_path + '/Y.csv'
    #         truth_file = file_path + '/truth.csv'
    #         G_file = file_path + '/G.csv'
    #         T_file = file_path + '/T.csv'
    #         E_file = file_path + '/E.csv'
    #
    #         ## temp_result是一个list，每个元素为dic
    #         temp_result.append(process_pool.apply_async(runEm, args=(data_file, truth_file, E_file, G_file, T_file, ALPHA, PHI_A, PHI_R, PHI_H, G_Size, k_values, data_name ,output_file_path)))
    #     process_pool.close()
    #     process_pool.join()
    #
    #     # 遍历每个大数据集的10个小数据集，处理结果
    #     RMSE_HARS_list = []
    #     MAE_HARS_list = []
    #     for res in temp_result:
    #         dic = res.get()
    #         RMSE_HARS_list.append(dic["RMSE"])
    #         MAE_HARS_list.append(dic["MAE"])
    #
    #     final_result[filename1]["ave_RMSE_HARS"] = np.mean(np.array(RMSE_HARS_list))
    #     final_result[filename1]["std_RMSE_HARS"] = np.std(np.array(RMSE_HARS_list) , ddof=1)
    #     final_result[filename1]["ave_MAE_HARS"] = np.mean(np.array(MAE_HARS_list))
    #     final_result[filename1]["std_MAE_HARS"] = np.std(np.array(MAE_HARS_list) , ddof=1)
    #
    #     temp_result.clear()
    #     RMSE_HARS_list.clear()
    #     MAE_HARS_list.clear()

    ################################  跑完了再处理数据  ######################################

    # print("#############################################################################")
    # ### 打印最终结果 放到record 文件中
    # for f_name in final_result.keys():
    #     print(f_name,"  ave_RMSE_HARS = ",final_result[f_name]["ave_RMSE_HARS"])
    #     print(f_name,"  std_RMSE_HARS = ",final_result[f_name]["std_RMSE_HARS"])
    #     print(f_name,"  ave_MAE_HARS = ",final_result[f_name]["ave_MAE_HARS"])
    #     print(f_name,"  std_MAE_HARS = ",final_result[f_name]["std_MAE_HARS"])
    #     print("--------------------------------------------------------------------")
    #
    # ### 打印画图需要的数据
    # x_list = []
    # y_ave_RMSE_HARS_list = []
    # std_RMSE_HARS_list = []
    # y_ave_MAE_HARS_list = []
    # std_MAE_HARS_list = []
    #
    # for filename1 in os.listdir(path):
    #     # 填入X轴数据 (冗余度)
    #     x = re.findall(r'(?<=_)\d+\.?\d*(?=r)', filename1)
    #     x_list.append(int(x[0]))
    #
    #     y_ave_RMSE_HARS_list.append(final_result[filename1]["ave_RMSE_HARS"])
    #     std_RMSE_HARS_list.append(final_result[filename1]["std_RMSE_HARS"])
    #     y_ave_MAE_HARS_list.append(final_result[filename1]["ave_MAE_HARS"])
    #     std_MAE_HARS_list.append(final_result[filename1]["std_MAE_HARS"])
    #
    #
    #
    # print("#############################################################################")
    # print("画图所需数据：")
    # print("x_list = ",x_list)
    # print("y_ave_RMSE_HARS_list = ",y_ave_RMSE_HARS_list)
    # print("std_RMSE_HARS_list = ",std_RMSE_HARS_list)
    # print("y_ave_MAE_HARS_list = ",y_ave_MAE_HARS_list)
    # print("std_MAE_HARS_list = ",std_MAE_HARS_list)
    # print("#############################################################################")
    #
    # with open('result.txt', 'a', encoding="utf-8") as f:
    #     f.write("x_list = " + str(x_list) + "\n")
    #     f.write("y_ave_RMSE_HARS_list = " + str(y_ave_RMSE_HARS_list) + "\n")
    #     f.write("std_RMSE_HARS_list = "+ str(std_RMSE_HARS_list) + "\n")
    #     f.write("y_ave_MAE_HARS_list = " + str(y_ave_MAE_HARS_list) + "\n")
    #     f.write("std_MAE_HARS_list = " + str(std_MAE_HARS_list) + "\n")

    ################################################################ 迭代运行并显示结果 ################################################################





####################################################################   测试   ####################################################################

    ####### 计算a_i的样本方差 ########
    # 用任务k值的均值代替ground_truth

    # ave_j = {}
    # for j in em.tasks:
    #     sum_k = 0
    #     for k in em.t2wl[j].values():
    #         sum_k += k
    #     ave_j[j] = sum_k / len(em.t2wl[j])
    #     # print("ave[",j,"]=",ave_j[j])
    #     # print("t2truth[",j,"]=",em.t2truth[j])
    #
    # # 求a_i的样本
    # a_hat = {}
    # for i in em.workers:
    #     sum = 0
    #     for j in em.w2tl[i].keys():
    #         y_ij = em.w2tl[i][j]
    #         sum += math.fabs(y_ij - ave_j[j])
    #     a_hat[i] = sum / len(em.w2tl[i].keys())
    # print("a_hat=", a_hat)
    #
    # # 求a_i的样本的均值
    # sum1 = 0
    # for i in a_hat:
    #     sum1 += a_hat[i]
    # ave = sum1 / len(a_hat)
    # print("ave=", ave)
    #
    # # 求a_i的样本的方差
    # sum2 = 0.0
    # for i in em.workers:
    #     sum2 += (a_hat[i] - ave)**2
    # a_hat_sigma_fang = sum2 / (len(a_hat) - 1)
    # print("a_hat_sigma_fang = ", a_hat_sigma_fang)
    # print("varphi_a=",math.sqrt((0.15171*2)/(4-math.pi)))
    ####### 计算a_i的样本方差 ########




    ################################ 单元测试 ################################
    # em = KAPS(data_file=data_file, truth_file=truth_file, E_file=E_file, G_file=G_file, T_file=T_file, ALPHA=ALPHA,
    #         PHI_A=PHI_A, PHI_R=PHI_R, PHI_H=PHI_H, k_values=k_values, data_name=data_name,
    #         output_file_path=output_file_path)

    # ********** 测试 calculate_sum_weight 函数   √√√√√
    # print(em.sum_weight_dict)

    # ********** 测试 calculate_all_weight 函数   √√√√√

    # weight = em.calculate_all_weight()
    # for i in em.workers:
    #     sum = 0.0
    #     if i in em.G_dict.keys():
    #         for ii in em.G_dict[i]:
    #             sum += weight[i][ii]
    #     sum += weight[i][i]
    #     print("sum=", sum)

    # ********** 测试 init函数   √√√√√
    # print(em.M)
    # print(em.N)
    # print(em.K)
    # print(em.k_values)
    # print(em.workers)
    # print(em.tasks)

    # ********** 测试 getE_T_G_Greverse   出现数据重复问题，已解决   √√√√√
    # E_dict, T_Matrix, G_dict, G_reverse_dict = em.getE_T_G_Greverse(E_file=E_file, G_file=G_file, T_file=T_file)
    # print("E_dict:",E_dict)
    # print("T_Matrix",T_Matrix)
    # index = 495
    # print("G_part_dict:",em.G_part_dict[897])
    # print("G_part_dict[897]:",em.G_part_dict[897])
    # print("G_dict",em.G_dict[2])
    # print(em.G_dict[1137])
    # print("G_reverse_dict[13306]",em.G_reverse_dict[13306])
    # print("G_reverse_part_dict[13306]:", em.G_reverse_part_dict[13306])
    # print(index,":",em.G_reverse_dict[index])
    # print(index,":", em.G_reverse_part_dict[index])
    # for i in em.G_dict[index]:
    #     print(i,em.E_dict[i])
    # ********** 测试 getT2Truth   √√√√√
    # print(em.t2truth)

    # ********** 测试 getT2WLandW2TL   √√√√√
    # print(em.w2tl)
    # print(em.t2wl)
    # print(em.label_list)

    # ********** 测试 三个初始化函数   √√√√√
    # print(em.ln_P_y)
    # print(em.P_y)
    # print(em.phi)
    # print(em.a)
    # print(em.h)
    # print(em.R_Matrix)

    # ********** 测试 calculate_GE  √√√√√
    # print(em.GE_dict)
    # print(em.E_dict[3024],em.E_dict[2106],em.E_dict[866],em.E_dict[5])
    # print(em.E_dict[3024]+em.E_dict[2106]+em.E_dict[866]+em.E_dict[5]) # 0.0035194744251525107

    # ********** 测试 calculate_all_sumg     √√√√√
    # print(em.sum_gea_GE_dict)

    # ********** 测试 calculate_all_ln_P_y
    # em.calculate_all_ln_P_y()
    # if i == 12 and j == 0 and k == 2.0:
    #     print("a[i]=", self.a[i])
    #     print("yij=", y_ij)
    #     print("k=", k)
    #     print("hi=", self.h[i])
    #     print("Tj=", self.T_Matrix[j])
    #     print("Ri=", self.R_Matrix[i])
    #     print("TjRi=", np.dot(self.T_Matrix[j], self.R_Matrix[i]))
    #     print("ln_P_y_C[i][j][k]=", self.ln_P_y_C[i][j][k])
    #     print("ln sqrt 2 pi=", math.log(math.sqrt(2 * math.pi)))
    #
    # cnt = 0
    # for i in em.workers:
    #     cnt += 1
    #     for j in em.w2tl[i].keys():
    #         k = 4.5
    #         print(em.ln_P_y[i][j][k])
    #         print(em.ln_P_y_C[i][j][k])
    #         print(em.ln_P_y_S[i][j][k])
    #         print("--------")
    #     if cnt >5: break


    ################################ 单元测试 ################################




    #################### grammar tests ####################
    # t2wl, w2tl, label_list = em.getT2WLandW2TL(data_file)
    # print(t2wl)
    # print(w2tl)
    # print(label_list)
    # print(em.workers)
    # for i in em.workers:
    #     print(i,end=" ")
    # print(em.tasks)
    # print("M =",em.M,"\n","N =",em.N,"\n",em.K,"\n")
    #
    # print("-------------")
    #
    # E_Matrix,G_Matrix,T_Matrix = em.getE_G_T(E_file=E_file,G_file=G_file,T_file=T_file)
    # print("G_Matrix:\n",G_Matrix)
    # print("T_Matrix:\n", T_Matrix)
    # print("E_Matrix:\n", E_Matrix)
    # print("R_Matrix:\n", em.R_Matrix)
    # print("--------------")
    # sss = np.inner(em.G_Matrix[0],em.E_Matrix)[0]
    # print(type(sss))
    # x = np.zeros((5, 5))
    # print(x)
    # print(x[0][2])
    # print(np.inner(T_Matrix[0],em.R_Matrix[0]))
    # x = [[0 for j in range(5)]for i in range(3)]
    # x[0][0] =x[2][0]= 1
    # print(x)
    # x = {2,6,5,1,8}
    # print(type(x))
    # dt = {i: i + 1 for i in range(10)}
    #
    # for i in dt.items():
    #     print("字典的键值对：", i)
    #     print("字典的键：", i[0])
    #     print("字典的值：", i[1])
    # L = np.ones((em.M))
    # print(L)

    # print(type(em.T_Matrix[0]))
    # x = np.transpose([em.T_Matrix[0]])
    # y = np.array([em.T_Matrix[0]]).T
    # print(y)
    # print(np.multiply(x, em.T_Matrix[0]))
    # print(np.multiply(y,em.T_Matrix[0]))
    # print(2/math.sqrt(2*math.pi)*pow(math.e,-24.5))
    # print(2/math.sqrt(2*math.pi))
    # print(pow(math.e,-24.5))
    # print(math.e)
    # print(math.pi)
    # print(-25.5*25.5*2)
    # print(math.pow(math.e,-25.5*25.5*2))
    # print(math.pow(math.e,-25.5*25.5*2) * (2/math.sqrt(2*math.pi) ))
    # for i in trange(100):
    #     # do something
    #     time.sleep(0.1)
    # d = {0:10,1:11,2:10,3:11,4:12,5:13}
    # s = set(d.values())
    # for i in s:
    #     print(i)
    # print(d.values())
    # print(s)

    # print(em.T_Matrix[0])
    # print(em.R_Matrix[0])
    # print(np.inner(em.T_Matrix[0], em.R_Matrix[0]))
    # print(np.dot(em.T_Matrix[0], em.R_Matrix[0]))

    # with open('./record.txt','a') as f:
    #     f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"\n")
    #     f.write("hello!\n\n")
    # x = math.log(math.e)
    # y = math.log(8)
    # z = pow(math.e,math.log(8))
    # print(x)
    # print(y)
    # print(z)
    # for i in range(1,-50,-1):
    #     print(i,math.pow(math.e,i))

    # for i in range(1,50):
    #     print(i,math.log(1/(10**i)))

    # with open('./record.txt', 'a', encoding="utf-8") as f:
    #     f.write("{:.3s}".format(str(155.32369)))

    # dic = {2:"x",1:"x",3:"x"}
    # print(dic)
    # print(type(dic))
    # for i in sorted(dic):
    #     print(i,dic[i])

    # a = np.matrix([[1,2,3],[4,5,6]])
    # print(a)
    # print(0.2*a)
    # sum = 0
    # for j in em.tasks:
    #     sum += len(em.t2wl[j].keys())
    #     print(j,":",len(em.t2wl[j].keys()))
    # print(sum / em.N)

    # grammar tests......

