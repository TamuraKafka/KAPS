##############################
#   一键启动所有baseline代码   #
##############################

import os
import time
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import *

from GTM.catd import CATD_Output
from GTM.GTM import GTM_Output
from KDEm.test import KDEm_Output
from l_LFCcont.method import l_LFCcont_Output
from GTM.CRH import CRH_Output


def pd_toExcel(data,fileName):
    data_name = []
    indicator = []


def run(data_name,data_file1,data_file2,truth_file):
    result = {}
    excelData = []

    with open('result_baseline.txt', 'a', encoding="utf-8") as f:


        # print("数据集：",data_name)
        f.write(time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))
        f.write("数据名称:" + str(data_name) + "\n")
        f.write('-------------------------------------------------------\n')

        # KDEm
        # print("正在运行KDEm...")
        RMSE_KDEm, MAE_KDEm, Runtime_KDEm = KDEm_Output(data_file2, truth_file)
        f.write("KDEm        " + str(data_name) + ":" + "\n")
        f.write("RMSE_KDEm: " + str(RMSE_KDEm) + "\n")
        f.write("MAE_KDEm: " + str(MAE_KDEm) + "\n")
        f.write("Runtime_KDEm: " + str(Runtime_KDEm) + "s\n")
        f.write('-------------------------------------------------------\n')
        # print("KDEm     RMSE: %f     MAE: %f     Runtime: %ss" % (RMSE_KDEm, MAE_KDEm, str(Runtime_KDEm)))
        result["KDEm"] = {}
        result["KDEm"]["RMSE_KDEm"] = RMSE_KDEm
        result["KDEm"]["MAE_KDEm"] = MAE_KDEm

        # CATD
        # print("正在运行CATD...")
        RMSE_CATD,MAE_CATD,Runtime_CATD = CATD_Output(datafile = data_file2,truth_file = truth_file)
        f.write("CATD        "+str(data_name)+":"+"\n")
        f.write("RMSE_CATD: " + str(RMSE_CATD) + "\n")
        f.write("MAE_CATD: " + str(MAE_CATD) + "\n")
        f.write("Runtime_CATD:"+str(Runtime_CATD)+"s\n")
        f.write('-------------------------------------------------------\n')
        # print("CATD     RMSE: %f     MAE: %f     Runtime: %ss" % (RMSE_CATD, MAE_CATD, str(Runtime_CATD)))
        result["CATD"] = {}
        result["CATD"]["RMSE_CATD"] = RMSE_CATD
        result["CATD"]["MAE_CATD"] = MAE_CATD


        # CRH
        # print("正在运行CRH...")
        RMSE_CRH, MAE_CRH, Runtime_CRH = CRH_Output(data_file2, truth_file)
        f.write("CRH        " + str(data_name) + ":" + "\n")
        f.write("RMSE_CRH: " + str(RMSE_CRH) + "\n")
        f.write("MAE_CRH: " + str(MAE_CRH) + "\n")
        f.write("Runtime_CRH: " + str(Runtime_CRH) + "s\n")
        f.write('-------------------------------------------------------\n')
        # print("CRH     RMSE: %f     MAE: %f     Runtime: %ss" % (RMSE_CRH, MAE_CRH, str(Runtime_CRH)))
        result["CRH"] = {}
        result["CRH"]["RMSE_CRH"] = RMSE_CRH
        result["CRH"]["MAE_CRH"] = MAE_CRH


        # GTM
        # print("正在运行GTM...")
        RMSE_GTM, MAE_GTM, Runtime_GTM = GTM_Output(data_file2,truth_file)
        f.write("GTM        "+str(data_name)+":"+"\n")
        f.write("RMSE_GTM: " + str(RMSE_GTM) + "\n")
        f.write("MAE_GTM: " + str(MAE_GTM) + "\n")
        f.write("Runtime_GTM: " + str(Runtime_GTM) + "s\n")
        f.write('-------------------------------------------------------\n')
        # print("GTM     RMSE: %f     MAE: %f     Runtime: %ss" % (RMSE_GTM, MAE_GTM, str(Runtime_GTM)))
        result["GTM"] = {}
        result["GTM"]["RMSE_GTM"] = RMSE_GTM
        result["GTM"]["MAE_GTM"] = MAE_GTM


        # LFC_N
        # print("正在运行LFC_N...")
        RMSE_LFC_N, MAE_LFC_N, Runtime_LFC_N = l_LFCcont_Output(data_file1, truth_file)
        f.write("LFC_N        " + str(data_name) + ":" + "\n")
        f.write("RMSE_LFC_N: " + str(RMSE_LFC_N) + "\n")
        f.write("MAE_LFC_N: " + str(MAE_LFC_N) + "\n")
        f.write("Runtime_LFC_N: " + str(Runtime_LFC_N) + "s\n")
        f.write('-------------------------------------------------------\n')
        # print("LFC_N     RMSE: %f     MAE: %f     Runtime: %ss" % (RMSE_LFC_N, MAE_LFC_N, str(Runtime_LFC_N)))
        result["LFC_N"] = {}
        result["LFC_N"]["RMSE_LFC_N"] = RMSE_LFC_N
        result["LFC_N"]["MAE_LFC_N"] = MAE_LFC_N


        # Median
        # print("正在运行Median...")
        # RMSE_Median, MAE_Median, Runtime_Median = Median_Output(data_file1, truth_file)
        # f.write("Median        " + str(data_name) + ":" + "\n")
        # f.write("RMSE_Median: " + str(RMSE_Median) + "\n")
        # f.write("MAE_Median: " + str(MAE_Median) + "\n")
        # f.write("Runtime_Median: " + str(Runtime_Median) + "s\n")
        # f.write('-------------------------------------------------------\n')
        # print("Median     RMSE: %f     MAE: %f     Runtime: %ss" % (RMSE_Median, MAE_Median, str(Runtime_Median)))

        # Mean
        # print("正在运行Mean...")
        # RMSE_Mean, MAE_Mean, Runtime_Mean = Mean_Output(data_file1, truth_file)
        # f.write("Mean        " + str(data_name) + ":" + "\n")
        # f.write("RMSE_Mean: " + str(RMSE_Mean) + "\n")
        # f.write("MAE_Mean: " + str(MAE_Mean) + "\n")
        # f.write("Runtime_Mean: " + str(Runtime_Mean) + "s\n")
        # f.write('-------------------------------------------------------\n')
        # print("Mean     RMSE: %f     MAE: %f     Runtime: %ss" % (RMSE_Mean, MAE_Mean, str(Runtime_Mean)))

        f.write('####################################################################################\n')
        f.write('####################################################################################\n')

    return result

if __name__ == "__main__":
    ########################## 跑一个数据集  ##########################

    # data_name = "DouBanDatasets_RedundancyCut/Douban_202t_110w_3r"
    # # data_name = "GoodReadsDatasets_RedundancyCut/GoodReads_309t_2864w_30r"
    #
    # data_file1 = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y.csv'
    # data_file2 = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y2.csv'
    # truth_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/truth.csv'
    #
    # result = run(data_file1=data_file1,data_file2=data_file2,truth_file=truth_file)
    # print(result)


    ##########################  跑多个数据集  ##########################

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

    # DataNameValues = ["DouBanDatasets/Douban_202t_26926w",
    #                   "DouBanDatasets/Douban_202t_53853w",
    #                   "DouBanDatasets/Douban_202t_80779w",
    #                   "DouBanDatasets/Douban_202t_107706w",
    #                   "DouBanDatasets/Douban_202t_134633w",
    #                   "DouBanDatasets/Douban_202t_161559w",
    #                   "DouBanDatasets/Douban_202t_188486w",
    #                   "DouBanDatasets/Douban_202t_215412w",
    #                   "DouBanDatasets/Douban_202t_242339w",
    #                   "DouBanDatasets/Douban_202t_269266w"]

    # DataNameValues = ["GoodReadsDatasets/GoodReads_309t_12021w",
    #                   "GoodReadsDatasets/GoodReads_309t_23941w",
    #                   "GoodReadsDatasets/GoodReads_309t_35947w",
    #                   "GoodReadsDatasets/GoodReads_309t_48021w",
    #                   "GoodReadsDatasets/GoodReads_309t_60105w",
    #                   "GoodReadsDatasets/GoodReads_309t_72166w",
    #                   "GoodReadsDatasets/GoodReads_309t_84218w",
    #                   "GoodReadsDatasets/GoodReads_309t_96302w",
    #                   "GoodReadsDatasets/GoodReads_309t_108363w",
    #                   "GoodReadsDatasets/GoodReads_309t_120415w"]

    # DataNameValues = ["DouBanDatasets_SWTest/Douban_7271w_0%",
    #                   "DouBanDatasets_SWTest/Douban_6906w_5%",
    #                   "DouBanDatasets_SWTest/Douban_6543w_10%",
    #                   "DouBanDatasets_SWTest/Douban_6179w_15%",
    #                   "DouBanDatasets_SWTest/Douban_5816w_20%",
    #                   "DouBanDatasets_SWTest/Douban_5452w_25%",
    #                   "DouBanDatasets_SWTest/Douban_5089w_30%",
    #                   "DouBanDatasets_SWTest/Douban_4725w_35%",
    #                   "DouBanDatasets_SWTest/Douban_4362w_40%",
    #                   "DouBanDatasets_SWTest/Douban_3998w_45%",
    #                   "DouBanDatasets_SWTest/Douban_3635w_50%"]

    # DataNameValues = ["GoodReadsDatasets_SWTest2/GoodReads_5702w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_5415w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_5130w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_4845w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_4560w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_4275w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_3990w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_3705w_309t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_3420w_308t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_3135w_308t",
    #                   "GoodReadsDatasets_SWTest2/GoodReads_2850w_308t"]

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

    result = {}
    for x in tqdm(DataNameValues,desc="运行中"):
        data_name = x
        data_file1 = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y.csv'
        data_file2 = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/Y2.csv'
        truth_file = '../../datasets/QuantitativeCrowdsourcing/' + data_name + '/truth.csv'
        result[x] = run(data_name=data_name,data_file1=data_file1, data_file2=data_file2, truth_file=truth_file)

    ### 处理跑完的数据 ###
    RMSE_KDEm_list = []
    MAE_KDEm_list = []
    RMSE_CATD_list = []
    MAE_CATD_list = []
    RMSE_CRH_list = []
    MAE_CRH_list = []
    RMSE_GTM_list = []
    MAE_GTM_list = []
    RMSE_LFC_N_list = []
    MAE_LFC_N_list = []
    for filename in DataNameValues:
        # 处理 KDEm 数据
        if "KDEm" in result[filename].keys():
            RMSE_KDEm_list.append(result[filename]["KDEm"]["RMSE_KDEm"])
            MAE_KDEm_list.append(result[filename]["KDEm"]["MAE_KDEm"])
        # 处理 CATD 数据
        if "CATD" in result[filename].keys():
            RMSE_CATD_list.append(result[filename]["CATD"]["RMSE_CATD"])
            MAE_CATD_list.append(result[filename]["CATD"]["MAE_CATD"])
        # 处理 CRH 数据
        if "CRH" in result[filename].keys():
            RMSE_CRH_list.append(result[filename]["CRH"]["RMSE_CRH"])
            MAE_CRH_list.append(result[filename]["CRH"]["MAE_CRH"])
        # 处理 GTM 数据
        if "GTM" in result[filename].keys():
            RMSE_GTM_list.append(result[filename]["GTM"]["RMSE_GTM"])
            MAE_GTM_list.append(result[filename]["GTM"]["MAE_GTM"])
        # 处理 LFC_N 数据
        if "LFC_N" in result[filename].keys():
            RMSE_LFC_N_list.append(result[filename]["LFC_N"]["RMSE_LFC_N"])
            MAE_LFC_N_list.append(result[filename]["LFC_N"]["MAE_LFC_N"])

    ## 将结果记录到文件中

    with open('result_baseline.txt', 'a', encoding="utf-8") as f:
        f.write("#############################################################################\n")
        f.write("Datasets = "+ str(DataNameValues) +"\n\n")
        f.write("RMSE 数据：\n")
        f.write("RMSE_KDEm  = "+ str(RMSE_KDEm_list)+"\n")
        f.write("RMSE_CATD  = "+ str (RMSE_CATD_list) +"\n")
        f.write("RMSE_CRH  = "+ str(RMSE_CRH_list) +"\n")
        f.write("RMSE_GTM  = "+  str(RMSE_GTM_list) +"\n")
        f.write("RMSE_LFCN  = " +str(RMSE_LFC_N_list) +"\n")

        f.write("#############################################################################\n")
        f.write("MAE 数据：\n")
        f.write("MAE_KDEm  = "+ str(MAE_KDEm_list)+"\n")
        f.write("MAE_CATD  = "+ str(MAE_CATD_list)+"\n")
        f.write("MAE_CRH  = "+ str( MAE_CRH_list)+"\n")
        f.write("MAE_GTM  = "+ str( MAE_GTM_list)+"\n")
        f.write("MAE_LFCN  = "+ str( MAE_LFC_N_list)+"\n")

    ##########################  跑平均值与标准差  ##########################

    # ### 遍历数据集 计算实验结果
    # path = "../../datasets/QuantitativeCrowdsourcing/" + "DouBanDatasets_RedundancyCut_Group"
    # # path = "../../datasets/QuantitativeCrowdsourcing/" + "GoodReadsDatasets_RedundancyCut_Group"
    #
    # baseline_result = {}
    # result = {}  # result[数据集名][方法名][指标名] = 数值
    # for filename1 in tqdm(os.listdir(path)):
    #     baseline_result[filename1] = {}
    #     # print(filename1)
    #
    #     # 计算每个最小单位的数据集的结果，并储存
    #     for filename2 in os.listdir(path + "/" + filename1):
    #         file_path = path + "/" + filename1 + "/" + filename2
    #         # print(filename2)
    #         data_file1 = file_path + '/Y.csv'
    #         data_file2 = file_path + '/Y2.csv'
    #         truth_file = file_path + '/truth.csv'
    #
    #         result[filename2] = run(data_name=filename2,data_file1=data_file1,data_file2=data_file2,truth_file=truth_file)
    #
    #     # 将每个filename1文件夹的十组数据分别计算出平均值和标准差并储存
    #     RMSE_KDEm_list = []
    #     MAE_KDEm_list = []
    #     RMSE_CATD_list = []
    #     MAE_CATD_list = []
    #     RMSE_CRH_list = []
    #     MAE_CRH_list = []
    #     RMSE_GTM_list = []
    #     MAE_GTM_list = []
    #     RMSE_LFC_N_list = []
    #     MAE_LFC_N_list = []
    #     for filename2 in os.listdir(path + "/" + filename1):
    #         # 处理 KDEm 数据
    #         if "KDEm" in result[filename2].keys():
    #             baseline_result[filename1]["KDEm"] = {}
    #             RMSE_KDEm_list.append(result[filename2]["KDEm"]["RMSE_KDEm"])
    #             MAE_KDEm_list.append(result[filename2]["KDEm"]["MAE_KDEm"])
    #         # 处理 CATD 数据
    #         if "CATD" in result[filename2].keys():
    #             baseline_result[filename1]["CATD"] = {}
    #             RMSE_CATD_list.append(result[filename2]["CATD"]["RMSE_CATD"])
    #             MAE_CATD_list.append(result[filename2]["CATD"]["MAE_CATD"])
    #         # 处理 CRH 数据
    #         if "CRH" in result[filename2].keys():
    #             baseline_result[filename1]["CRH"] = {}
    #             RMSE_CRH_list.append(result[filename2]["CRH"]["RMSE_CRH"])
    #             MAE_CRH_list.append(result[filename2]["CRH"]["MAE_CRH"])
    #         # 处理 GTM 数据
    #         if "GTM" in result[filename2].keys():
    #             baseline_result[filename1]["GTM"] = {}
    #             RMSE_GTM_list.append(result[filename2]["GTM"]["RMSE_GTM"])
    #             MAE_GTM_list.append(result[filename2]["GTM"]["MAE_GTM"])
    #         # 处理 LFC_N 数据
    #         if "LFC_N" in result[filename2].keys():
    #             baseline_result[filename1]["LFC_N"] = {}
    #             RMSE_LFC_N_list.append(result[filename2]["LFC_N"]["RMSE_LFC_N"])
    #             MAE_LFC_N_list.append(result[filename2]["LFC_N"]["MAE_LFC_N"])
    #
    #
    #
    #     # 计算KDEm 平均值及标准差
    #     baseline_result[filename1]["KDEm"]["ave_RMSE_KDEm"] = np.mean(np.array(RMSE_KDEm_list))
    #     baseline_result[filename1]["KDEm"]["std_RMSE_KDEm"] = np.std(np.array(RMSE_KDEm_list) , ddof=1)
    #     baseline_result[filename1]["KDEm"]["ave_MAE_KDEm"] = np.mean(np.array(MAE_KDEm_list))
    #     baseline_result[filename1]["KDEm"]["std_MAE_KDEm"] = np.std(np.array(MAE_KDEm_list) , ddof=1)
    #     # 计算CATD 平均值及标准差
    #     baseline_result[filename1]["CATD"]["ave_RMSE_CATD"] = np.mean(np.array(RMSE_CATD_list))
    #     baseline_result[filename1]["CATD"]["std_RMSE_CATD"] = np.std(np.array(RMSE_CATD_list) , ddof=1)
    #     baseline_result[filename1]["CATD"]["ave_MAE_CATD"] = np.mean(np.array(MAE_CATD_list))
    #     baseline_result[filename1]["CATD"]["std_MAE_CATD"] = np.std(np.array(MAE_CATD_list) , ddof=1)
    #     # 计算CRH 平均值及标准差
    #     baseline_result[filename1]["CRH"]["ave_RMSE_CRH"] = np.mean(np.array(RMSE_CRH_list))
    #     baseline_result[filename1]["CRH"]["std_RMSE_CRH"] = np.std(np.array(RMSE_CRH_list), ddof=1)
    #     baseline_result[filename1]["CRH"]["ave_MAE_CRH"] = np.mean(np.array(MAE_CRH_list))
    #     baseline_result[filename1]["CRH"]["std_MAE_CRH"] = np.std(np.array(MAE_CRH_list), ddof=1)
    #     # 计算GTM 平均值及标准差
    #     baseline_result[filename1]["GTM"]["ave_RMSE_GTM"] = np.mean(np.array(RMSE_GTM_list))
    #     baseline_result[filename1]["GTM"]["std_RMSE_GTM"] = np.std(np.array(RMSE_GTM_list), ddof=1)
    #     baseline_result[filename1]["GTM"]["ave_MAE_GTM"] = np.mean(np.array(MAE_GTM_list))
    #     baseline_result[filename1]["GTM"]["std_MAE_GTM"] = np.std(np.array(MAE_GTM_list), ddof=1)
    #     # 计算LFC_N 平均值及标准差
    #     baseline_result[filename1]["LFC_N"]["ave_RMSE_LFC_N"] = np.mean(np.array(RMSE_LFC_N_list))
    #     baseline_result[filename1]["LFC_N"]["std_RMSE_LFC_N"] = np.std(np.array(RMSE_LFC_N_list), ddof=1)
    #     baseline_result[filename1]["LFC_N"]["ave_MAE_LFC_N"] = np.mean(np.array(MAE_LFC_N_list))
    #     baseline_result[filename1]["LFC_N"]["std_MAE_LFC_N"] = np.std(np.array(MAE_LFC_N_list), ddof=1)
    #
    #
    #
    # ##################  记录并打印数据  ###################
    # ## 记录运行结果到record文件
    # baseline_name = ["KDEm", "CATD", "CRH", "GTM", "LFC_N"]
    # print("RMSE 数据：")
    # print("#############################################################################")
    # for filename1 in os.listdir(path):
    #     print("数据集： ",filename1)
    #     for name in baseline_name:
    #         key1 = "ave_RMSE_" + name
    #         key2 = "std_RMSE_" + name
    #         key3 = "ave_MAE_" + name
    #         key4 = "std_MAE_" + name
    #         print(filename1, "  ", name,"  " , key1, " = ", baseline_result[filename1][name][key1])
    #         print(filename1, "  ", name, "  ", key2, " = ", baseline_result[filename1][name][key2])
    #         print(filename1, "  ", name, "  ", key3, " = ", baseline_result[filename1][name][key3])
    #         print(filename1, "  ", name, "  ", key4, " = ", baseline_result[filename1][name][key4])
    #
    #         print("-------------------------")
    #     print("*************************************************")
    # print("#############################################################################")
    #
    # y_ave_RMSE_KDEm_list = []
    # std_RMSE_KDEm_list = []
    # y_ave_MAE_KDEm_list = []
    # std_MAE_KDEm_list = []
    #
    # y_ave_RMSE_CATD_list = []
    # std_RMSE_CATD_list = []
    # y_ave_MAE_CATD_list = []
    # std_MAE_CATD_list = []
    #
    # y_ave_RMSE_CRH_list = []
    # std_RMSE_CRH_list = []
    # y_ave_MAE_CRH_list = []
    # std_MAE_CRH_list = []
    #
    # y_ave_RMSE_GTM_list = []
    # std_RMSE_GTM_list = []
    # y_ave_MAE_GTM_list = []
    # std_MAE_GTM_list = []
    #
    # y_ave_RMSE_LFC_N_list = []
    # std_RMSE_LFC_N_list = []
    # y_ave_MAE_LFC_N_list = []
    # std_MAE_LFC_N_list = []
    #
    # x_list = []
    # ## 整理 画图所需的数据
    # for filename1 in os.listdir(path):
    #     # 填入X轴数据 (冗余度)
    #     x = re.findall(r'(?<=_)\d+\.?\d*(?=r)', filename1)
    #     x_list.append(int(x[0]))
    #
    #     # 封装 KDEm
    #     y_ave_RMSE_KDEm_list.append(baseline_result[filename1]["KDEm"]["ave_RMSE_KDEm"])
    #     std_RMSE_KDEm_list.append(baseline_result[filename1]["KDEm"]["std_RMSE_KDEm"])
    #     y_ave_MAE_KDEm_list.append(baseline_result[filename1]["KDEm"]["ave_MAE_KDEm"])
    #     std_MAE_KDEm_list.append(baseline_result[filename1]["KDEm"]["std_MAE_KDEm"])
    #
    #     # 封装 CATD
    #     y_ave_RMSE_CATD_list.append(baseline_result[filename1]["CATD"]["ave_RMSE_CATD"])
    #     std_RMSE_CATD_list.append(baseline_result[filename1]["CATD"]["std_RMSE_CATD"])
    #     y_ave_MAE_CATD_list.append(baseline_result[filename1]["CATD"]["ave_MAE_CATD"])
    #     std_MAE_CATD_list.append(baseline_result[filename1]["CATD"]["std_MAE_CATD"])
    #
    #     # 封装 CRH
    #     y_ave_RMSE_CRH_list.append(baseline_result[filename1]["CRH"]["ave_RMSE_CRH"])
    #     std_RMSE_CRH_list.append(baseline_result[filename1]["CRH"]["std_RMSE_CRH"])
    #     y_ave_MAE_CRH_list.append(baseline_result[filename1]["CRH"]["ave_MAE_CRH"])
    #     std_MAE_CRH_list.append(baseline_result[filename1]["CRH"]["std_MAE_CRH"])
    #
    #     # 封装 GTM
    #     y_ave_RMSE_GTM_list.append(baseline_result[filename1]["GTM"]["ave_RMSE_GTM"])
    #     std_RMSE_GTM_list.append(baseline_result[filename1]["GTM"]["std_RMSE_GTM"])
    #     y_ave_MAE_GTM_list.append(baseline_result[filename1]["GTM"]["ave_MAE_GTM"])
    #     std_MAE_GTM_list.append(baseline_result[filename1]["GTM"]["std_MAE_GTM"])
    #
    #     # 封装 LFC_N
    #     y_ave_RMSE_LFC_N_list.append(baseline_result[filename1]["LFC_N"]["ave_RMSE_LFC_N"])
    #     std_RMSE_LFC_N_list.append(baseline_result[filename1]["LFC_N"]["std_RMSE_LFC_N"])
    #     y_ave_MAE_LFC_N_list.append(baseline_result[filename1]["LFC_N"]["ave_MAE_LFC_N"])
    #     std_MAE_LFC_N_list.append(baseline_result[filename1]["LFC_N"]["std_MAE_LFC_N"])
    #
    # ## 打印 画图所需的数据
    # print("#############################################################################")
    # print("RMSE 数据：")
    # print("y_ave_RMSE_KDEm_list  = ",y_ave_RMSE_KDEm_list)
    # print("std_RMSE_KDEm_list  = ",std_RMSE_KDEm_list)
    # print("y_ave_RMSE_CATD_list  = ", y_ave_RMSE_CATD_list)
    # print("std_RMSE_CATD_list  = ", std_RMSE_CATD_list)
    # print("y_ave_RMSE_CRH_list  = ", y_ave_RMSE_CRH_list)
    # print("std_RMSE_CRH_list  = ", std_RMSE_CRH_list)
    # print("y_ave_RMSE_GTM_list  = ", y_ave_RMSE_GTM_list)
    # print("std_RMSE_GTM_list  = ", std_RMSE_GTM_list)
    # print("y_ave_RMSE_LFC_N_list  = ", y_ave_RMSE_LFC_N_list)
    # print("std_RMSE_LFC_N_list  = ", std_RMSE_LFC_N_list)
    #
    # print("#############################################################################")
    # print("MAE 数据：")
    # print("y_ave_MAE_KDEm_list  = ", y_ave_MAE_KDEm_list)
    # print("std_MAE_KDEm_list  = ", std_MAE_KDEm_list)
    # print("y_ave_MAE_CATD_list  = ", y_ave_MAE_CATD_list)
    # print("std_MAE_CATD_list  = ", std_MAE_CATD_list)
    # print("y_ave_MAE_CRH_list  = ", y_ave_MAE_CRH_list)
    # print("std_MAE_CRH_list  = ", std_MAE_CRH_list)
    # print("y_ave_MAE_GTM_list  = ", y_ave_MAE_GTM_list)
    # print("std_MAE_GTM_list  = ", std_MAE_GTM_list)
    # print("y_ave_MAE_LFC_N_list  = ", y_ave_MAE_LFC_N_list)
    # print("std_MAE_LFC_N_list  = ", std_MAE_LFC_N_list)
    #
    # print("#############################################################################")
    #
    # with open('result_baseline.txt', 'a', encoding="utf-8") as f:
    #     f.write("#############################################################################\n")
    #     f.write("RMSE 数据：\n")
    #     f.write("y_ave_RMSE_KDEm_list  = "+ str(y_ave_RMSE_KDEm_list)+"\n")
    #     f.write("std_RMSE_KDEm_list  = "+ str( std_RMSE_KDEm_list) +"\n")
    #     f.write("y_ave_RMSE_CATD_list  = "+ str (y_ave_RMSE_CATD_list) +"\n")
    #     f.write("std_RMSE_CATD_list  = " + str(std_RMSE_CATD_list) +"\n")
    #     f.write("y_ave_RMSE_CRH_list  = "+ str(y_ave_RMSE_CRH_list) +"\n")
    #     f.write("std_RMSE_CRH_list  = " +str(std_RMSE_CRH_list) +"\n")
    #     f.write("y_ave_RMSE_GTM_list  = "+  str(y_ave_RMSE_GTM_list) +"\n")
    #     f.write("std_RMSE_GTM_list  = "+ str(std_RMSE_GTM_list) +"\n")
    #     f.write("y_ave_RMSE_LFC_N_list  = " +str(y_ave_RMSE_LFC_N_list) +"\n")
    #     f.write("std_RMSE_LFC_N_list  = "+str(std_RMSE_LFC_N_list) +"\n")
    #
    #     f.write("#############################################################################\n")
    #     f.write("MAE 数据：\n")
    #     f.write("y_ave_MAE_KDEm_list  = "+ str(y_ave_MAE_KDEm_list)+"\n")
    #     f.write("std_MAE_KDEm_list  = "+ str(std_MAE_KDEm_list)+"\n")
    #     f.write("y_ave_MAE_CATD_list  = "+ str(y_ave_MAE_CATD_list)+"\n")
    #     f.write("std_MAE_CATD_list  = "+ str( std_MAE_CATD_list)+"\n")
    #     f.write("y_ave_MAE_CRH_list  = "+ str( y_ave_MAE_CRH_list)+"\n")
    #     f.write("std_MAE_CRH_list  = "+ str( std_MAE_CRH_list)+"\n")
    #     f.write("y_ave_MAE_GTM_list  = "+ str( y_ave_MAE_GTM_list)+"\n")
    #     f.write("std_MAE_GTM_list  = "+ str( std_MAE_GTM_list)+"\n")
    #     f.write("y_ave_MAE_LFC_N_list  = "+ str( y_ave_MAE_LFC_N_list)+"\n")
    #     f.write("std_MAE_LFC_N_list  = "+ str( std_MAE_LFC_N_list)+"\n")
