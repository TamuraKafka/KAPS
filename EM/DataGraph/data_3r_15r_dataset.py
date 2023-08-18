#####################################
## RedundancyCut  3r_15r ##
#####################################
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from tqdm import *

##############  我的方法与baseline比较结果图  ######################################################################################################################################

################################## Douban RMSE ###################################

DatasetSize = [3,4,5,6,7,8,9,10,11,12,13,14,15]

Douban_RMSE_KARS = [0.9414094371163544, 0.8895591785049096, 0.9098816929588938, 0.9450652458758702, 0.9735109972757386, 0.9631251046734102, 0.954936657275371, 0.9675806574578207, 0.9854473465879046, 0.9662465975491626, 0.9619060305328745, 0.9550876421015759, 0.9562141829860934]
Douban_RMSE_KDEm =[1.07291,1.06442,1.03479,1.03043,1.06522,1.04500,1.00847,0.99629,1.00592,1.02928,1.01939,1.03246,1.02555]
Douban_RMSE_CATD = [1.16798,1.14800,1.15699,1.13665,1.19178,1.14937,1.16218,1.12339,1.15819,1.17397,1.18947,1.18801,1.18416]
Douban_RMSE_CRH =[1.00787,0.98516,0.99046,0.97737,0.99803,0.99468,1.00532,0.99178,0.99695,0.98550,0.98736,0.99600,0.99449]
Douban_RMSE_GTM =[3.39217,3.22669,3.06876,2.99895,2.83493,2.69387,2.54152,2.46165,2.36490,2.26699,2.14469,2.11671,2.03105]
Douban_RMSE_LFCN = [1.07165,1.06362,1.16525,1.11058,1.12947,1.16661,1.19154,1.17142,1.19977,1.19298,1.17600,1.19747,1.20892]
Douban_RMSE_Median = [1.07346,1.03124,1.11686,1.07139,1.11952,1.10315,1.15005,1.10202,1.15778,1.10359,1.12964,1.13106,1.14185]
Douban_RMSE_Mean = [0.96442, 0.94818 ,0.95619 ,0.94178 ,0.95834 ,0.95756 ,0.96848 ,0.95942 ,0.96655 ,0.95295 ,0.95618 ,0.96819, 0.96642 ]
Douban_RMSE_CTD = [1.18189, 1.20048 ,1.23246 ,1.23409 ,1.28822 ,1.28053 ,1.30774 ,1.29494 ,1.28884 ,1.26474 ,1.27433 ,1.27704 ,1.31183 ]
# 对数化指标
Log_Douban_RMSE_KARS = []
Log_Douban_RMSE_KDEm = []
Log_Douban_RMSE_CATD = []
Log_Douban_RMSE_CRH = []
Log_Douban_RMSE_GTM = []
Log_Douban_RMSE_LFCN = []
Log_Douban_RMSE_CTD = []
for val in Douban_RMSE_KARS:
    Log_Douban_RMSE_KARS.append(math.log(val))
for val in Douban_RMSE_KDEm:
    Log_Douban_RMSE_KDEm.append(math.log(val))
for val in Douban_RMSE_CATD:
    Log_Douban_RMSE_CATD.append(math.log(val))
for val in Douban_RMSE_CRH:
    Log_Douban_RMSE_CRH.append(math.log(val))
for val in Douban_RMSE_GTM:
    Log_Douban_RMSE_GTM.append(math.log(val))
for val in Douban_RMSE_LFCN:
    Log_Douban_RMSE_LFCN.append(math.log(val))
for val in Douban_RMSE_CTD:
    Log_Douban_RMSE_CTD.append(math.log(val))

fig = plt.figure()

#  画 真实指标
# plt.plot(DatasetSize,Douban_RMSE_KARS,marker = 'o',color = 'r', label = 'KARS')
# plt.plot(DatasetSize,Douban_RMSE_KDEm,marker = '^',color = 'fuchsia', label = 'KDEm')
# plt.plot(DatasetSize,Douban_RMSE_CATD,marker = 's',color = 'deepskyblue', label = 'CATD')
# plt.plot(DatasetSize,Douban_RMSE_CRH, marker = '.',color = 'gray', label = 'CRH')
# plt.plot(DatasetSize,Douban_RMSE_GTM, marker = 'D',color = 'gold', label = 'GTM')
# plt.plot(DatasetSize,Douban_RMSE_LFCN,marker = 'v',color = 'g', label = 'LFC_N')


#  画 对数化指标
plt.plot(DatasetSize,Log_Douban_RMSE_KARS,marker = 'o',color = 'r', label = 'KARS')
plt.plot(DatasetSize,Log_Douban_RMSE_KDEm,marker = '^',color = 'fuchsia', label = 'KDEm')
plt.plot(DatasetSize,Log_Douban_RMSE_CATD,marker = 's',color = 'deepskyblue', label = 'CATD')
plt.plot(DatasetSize,Log_Douban_RMSE_CRH, marker = '.',color = 'gray', label = 'CRH')
plt.plot(DatasetSize,Log_Douban_RMSE_GTM, marker = 'D',color = 'gold', label = 'GTM')
plt.plot(DatasetSize,Log_Douban_RMSE_LFCN,marker = 'v',color = 'g', label = 'LFC_N')
plt.plot(DatasetSize,Log_Douban_RMSE_CTD,marker = 'p',color = 'peru', label = 'CTD')

plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.xticks(np.arange(3,16,1))
plt.legend()
# plt.rcParams.update({'font.size':14})
plt.legend(loc='center left',bbox_to_anchor=(0, 0.57))
plt.xlabel("Redundancy",fontsize = 25)
plt.ylabel("Log(RMSE)",fontsize = 25)
#plt.title("Douban Dataset",fontsize = 30)
plt.savefig('C:/Users/chopin/Desktop/'+'Douban_RMSE_Methods_Comparison.pdf', bbox_inches='tight')
plt.show()



############################################## Douban MAE ###############################################
fig = plt.figure()

DatasetSize = [3,4,5,6,7,8,9,10,11,12,13,14,15]
#  真实指标

Douban_MAE_KARS = [0.7464526509222578, 0.6946513094334246, 0.721252763880024, 0.7481620548650574, 0.7628660605959874, 0.7657879087715656, 0.7561291781529876, 0.7667146496846896, 0.7743807291346724, 0.763476426267721, 0.7602462024553821, 0.7620803315485999, 0.7643059874236837]
Douban_MAE_KDEm = [0.87123,0.84975,0.83102,0.83332,0.86270,0.84345,0.82230,0.79471,0.80963,0.83926,0.83092,0.84894,0.84459]
Douban_MAE_CATD = [0.95851,0.92553,0.93956,0.94271,0.98742,0.96211,0.97547,0.93626,0.95507,0.97269,0.98496,0.98776,0.96848]
Douban_MAE_CRH = [0.80648,0.76931,0.78361,0.77684,0.79135,0.79742,0.80849,0.79493,0.79627,0.78896,0.78732,0.79873,0.80277]
Douban_MAE_GTM = [3.19158,3.00129,2.79320,2.70494,2.48865,2.31086,2.17412,2.08194,1.98978,1.87844,1.77051,1.74326,1.66714]
Douban_MAE_LFCN = [0.86238,0.85042,0.96441,0.90256,0.92357,0.96434,0.98713,0.97377,0.98481,0.99423,0.96349,0.98408,0.99627]
Douban_MAE_Median = [0.85495,0.82277,0.89455,0.85891,0.90248,0.89109,0.94208,0.89901,0.95198,0.89802,0.92327,0.92970,0.94208]
Douban_MAE_Mean = [0.77211, 0.75074 ,0.75495 ,0.75149 ,0.75820 ,0.76782 ,0.77618 ,0.76535 ,0.77232 ,0.76279 ,0.76295 ,0.77405, 0.77838 ]
Douban_MAE_CTD = [0.91488, 0.91090 ,0.95864 ,0.96346 ,1.02497 ,1.02313 ,1.05681 ,1.04696 ,1.03707 ,1.00834 ,1.03019 ,1.03713, 1.07772 ]

# 对数化指标
Log_Douban_MAE_KARS = []
Log_Douban_MAE_KDEm = []
Log_Douban_MAE_CATD = []
Log_Douban_MAE_CRH = []
Log_Douban_MAE_GTM = []
Log_Douban_MAE_LFCN = []
Log_Douban_MAE_CTD = []
for val in Douban_MAE_KARS:
    Log_Douban_MAE_KARS.append(math.log(val))
for val in Douban_MAE_KDEm:
    Log_Douban_MAE_KDEm.append(math.log(val))
for val in Douban_MAE_CATD:
    Log_Douban_MAE_CATD.append(math.log(val))
for val in Douban_MAE_CRH:
    Log_Douban_MAE_CRH.append(math.log(val))
for val in Douban_MAE_GTM:
    Log_Douban_MAE_GTM.append(math.log(val))
for val in Douban_MAE_LFCN:
    Log_Douban_MAE_LFCN.append(math.log(val))
for val in Douban_MAE_CTD:
    Log_Douban_MAE_CTD.append(math.log(val))

# 画 真实指标
# plt.plot(DatasetSize,Douban_MAE_KARS, marker = 'o',color = 'r', label = 'KARS')
# plt.plot(DatasetSize,Douban_MAE_KDEm, marker = '^',color = 'fuchsia', label = 'KDEm')
# plt.plot(DatasetSize,Douban_MAE_CATD, marker = 's',color = 'deepskyblue', label = 'CATD')
# plt.plot(DatasetSize,Douban_MAE_CRH,  marker = '.',color = 'gray', label = 'CRH')
# plt.plot(DatasetSize,Douban_MAE_GTM,marker = 'D',color = 'gold', label = 'GTM')
# plt.plot(DatasetSize,Douban_MAE_LFCN, marker = 'v',color = 'g', label = 'LFC_N')

#  画 对数化指标
plt.plot(DatasetSize,Log_Douban_MAE_KARS,marker = 'o',color = 'r', label = 'KARS')
plt.plot(DatasetSize,Log_Douban_MAE_KDEm,marker = '^',color = 'fuchsia', label = 'KDEm')
plt.plot(DatasetSize,Log_Douban_MAE_CATD,marker = 's',color = 'deepskyblue', label = 'CATD')
plt.plot(DatasetSize,Log_Douban_MAE_CRH, marker = '.',color = 'gray', label = 'CRH')
plt.plot(DatasetSize,Log_Douban_MAE_GTM, marker = 'D',color = 'gold', label = 'GTM')
plt.plot(DatasetSize,Log_Douban_MAE_LFCN,marker = 'v',color = 'g', label = 'LFC_N')
plt.plot(DatasetSize,Log_Douban_MAE_CTD,marker = 'p',color = 'peru', label = 'CTD')

plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xticks(np.arange(3,16,1))
plt.legend(fontsize = 15)
plt.legend(loc='center left',bbox_to_anchor=(0, 0.55))
plt.xlabel("Redundancy",fontsize = 25)
#plt.title("Douban Dataset",fontsize = 30)
plt.ylabel("Log(MAE)",fontsize = 25)

plt.savefig('C:/Users/chopin/Desktop/'+'Douban_MAE_Methods_Comparison.pdf', bbox_inches='tight')
plt.show()


# ######################################################### GoodReads RMSE #################################################
fig = plt.figure()

DatasetSize = [3,4,5,6,7,8,9,10,11,12,13,14,15]


GoodReads_RMSE_KARS = [0.6591769078151984, 0.6846754110644023, 0.6038345036544089, 0.6072087877796629, 0.5483624018664239, 0.5779313232762178, 0.5052871176275671, 0.5150103679983572, 0.5075515206408928,0.5009596397587451, 0.5177094973478193, 0.5047625178081295, 0.48493793340512714]
GoodReads_RMSE_KDEm =[0.95576,1.02259,0.86353,0.84285,0.84676,0.86725,0.74295,0.80113,0.73713,0.73276,0.67805,0.72969,0.69374]
GoodReads_RMSE_CATD =[0.96787,0.96433,0.99307,0.96074,0.94784,0.99953,0.92896,0.90201,0.91964,0.90145,0.91741,0.84261,0.83919]
GoodReads_RMSE_CRH =[0.64020,0.62376,0.59531,0.57778,0.55567,0.55093,0.54762,0.52610,0.51261,0.50964,0.50631,0.49926,0.50361]
GoodReads_RMSE_GTM =[3.61855,3.49796,3.39104,3.26787,3.17290,3.11324,3.04035,2.95378,2.88326,2.82206,2.74789,2.67572,2.64141]
GoodReads_RMSE_LFCN =[0.67705,0.70563,0.66804,0.62662,0.63579,0.58482,0.59807,0.57190,0.55392,0.55148,0.52754,0.52783,0.52621]
GoodReads_RMSE_Median = [0.75112,0.62971,0.63839,0.58661,0.61568,0.56939,0.58716,0.53571,0.55134,0.52334,0.56411,0.51993,0.56526]
GoodReads_RMSE_Mean = [0.62737, 0.61817 ,0.58318 ,0.57296 ,0.55034 ,0.54500 ,0.54394 ,0.52303 ,0.51110 ,0.50796 ,0.50442 ,0.49857, 0.50327 ]
GoodReads_RMSE_CTD = [1.35683, 1.34787 ,1.30843 ,1.32309 ,1.29955 ,1.30088 ,1.29941 ,1.28780 ,1.29657 ,1.28379 ,1.28895 ,1.28689, 1.29158 ]

# 对数化指标
Log_GoodReads_RMSE_KARS = []
Log_GoodReads_RMSE_KDEm = []
Log_GoodReads_RMSE_CATD = []
Log_GoodReads_RMSE_CRH = []
Log_GoodReads_RMSE_GTM = []
Log_GoodReads_RMSE_LFCN = []
Log_GoodReads_RMSE_CTD = []
for val in GoodReads_RMSE_KARS:
    Log_GoodReads_RMSE_KARS.append(math.log(val))
for val in GoodReads_RMSE_KDEm:
    Log_GoodReads_RMSE_KDEm.append(math.log(val))
for val in GoodReads_RMSE_CATD:
    Log_GoodReads_RMSE_CATD.append(math.log(val))
for val in GoodReads_RMSE_CRH:
    Log_GoodReads_RMSE_CRH.append(math.log(val))
for val in GoodReads_RMSE_GTM:
    Log_GoodReads_RMSE_GTM.append(math.log(val))
for val in GoodReads_RMSE_LFCN:
    Log_GoodReads_RMSE_LFCN.append(math.log(val))
for val in GoodReads_RMSE_CTD:
    Log_GoodReads_RMSE_CTD.append(math.log(val))

# 画 真实指标
# plt.plot(DatasetSize,GoodReads_RMSE_KARS, marker = 'o',color = 'r', label = 'KARS')
# plt.plot(DatasetSize,GoodReads_RMSE_KDEm, marker = '^',color = 'fuchsia', label = 'KDEm')
# plt.plot(DatasetSize,GoodReads_RMSE_CATD, marker = 's',color = 'deepskyblue', label = 'CATD')
# plt.plot(DatasetSize,GoodReads_RMSE_CRH,  marker = '.',color = 'gray', label = 'CRH')
# plt.plot(DatasetSize,GoodReads_RMSE_GTM,marker = 'D',color = 'gold', label = 'GTM')
# plt.plot(DatasetSize,GoodReads_RMSE_LFCN, marker = 'v',color = 'g', label = 'LFC_N')

# 画 对数化指标
plt.plot(DatasetSize,Log_GoodReads_RMSE_KARS, marker = 'o',color = 'r', label = 'KARS')
plt.plot(DatasetSize,Log_GoodReads_RMSE_KDEm, marker = '^',color = 'fuchsia', label = 'KDEm')
plt.plot(DatasetSize,Log_GoodReads_RMSE_CATD, marker = 's',color = 'deepskyblue', label = 'CATD')
plt.plot(DatasetSize,Log_GoodReads_RMSE_CRH,  marker = '.',color = 'gray', label = 'CRH')
plt.plot(DatasetSize,Log_GoodReads_RMSE_GTM,marker = 'D',color = 'gold', label = 'GTM')
plt.plot(DatasetSize,Log_GoodReads_RMSE_LFCN, marker = 'v',color = 'g', label = 'LFC_N')
plt.plot(DatasetSize,Log_GoodReads_RMSE_CTD,marker = 'p',color = 'peru', label = 'CTD')

plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xticks(np.arange(3,16,1))
plt.legend(loc='center left',bbox_to_anchor=(0, 0.73),prop={'size':9})
# plt.legend(loc='center right',bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Redundancy",fontsize = 25)
# plt.title("GoodReads Dataset",fontsize = 30)
plt.ylabel("Log(RMSE)",fontsize = 25)

plt.savefig('C:/Users/chopin/Desktop/'+'GoodReads_RMSE_Methods_Comparison.pdf', bbox_inches='tight')
plt.show()




###################################################### GoodReads MAE ######################################

fig = plt.figure()

DatasetSize = [3,4,5,6,7,8,9,10,11,12,13,14,15]


GoodReads_MAE_KARS = [0.5091172894924819, 0.5245054770210781, 0.4783699519303365, 0.47982394790476984, 0.4324067760761198, 0.4630517419575198, 0.40023863157140255, 0.4105251262712765, 0.40316483971774697,0.4083064101466439, 0.4218758572035236, 0.4026817498329264, 0.3892017172830876]
GoodReads_MAE_KDEm =[0.73391,0.77408,0.62760,0.62656,0.61172,0.63307,0.54656,0.60440,0.55369,0.55433,0.51709,0.55931,0.52099]
GoodReads_MAE_CATD = [0.73965,0.72620,0.74639,0.73222,0.71291,0.74834,0.70885,0.69119,0.70449,0.67038,0.69339,0.64521,0.64102]
GoodReads_MAE_CRH =[0.49611,0.49145,0.47459,0.46535,0.44051,0.44712,0.44405,0.42529,0.41658,0.41585,0.41324,0.40476,0.41035]
GoodReads_MAE_GTM =[3.46187,3.31606,3.17694,3.02576,2.90895,2.81832,2.74395,2.65005,2.57275,2.50891,2.41895,2.34328,2.30915]
GoodReads_MAE_LFCN =[0.53555,0.54599,0.53963,0.51049,0.50401,0.47240,0.48434,0.45088,0.44653,0.44308,0.41718,0.42531,0.42087]
GoodReads_MAE_Median = [0.60421,0.50388,0.51294,0.47864,0.49159,0.45825,0.47152,0.42330,0.44110,0.41521,0.44887,0.41392,0.45728]
GoodReads_MAE_Mean = [0.49029, 0.48447 ,0.46246 ,0.45987 ,0.43703 ,0.44353 ,0.44196 ,0.42168 ,0.41621 ,0.41559 ,0.41357, 0.40495, 0.41219 ]
GoodReads_MAE_CTD = [0.90319, 0.88762 ,0.83958 ,0.86098 ,0.81701 ,0.82869 ,0.82417 ,0.80027 ,0.80739 ,0.79509 ,0.80678 ,0.79972, 0.80941 ]

# 对数化指标
Log_GoodReads_MAE_KARS = []
Log_GoodReads_MAE_KDEm = []
Log_GoodReads_MAE_CATD = []
Log_GoodReads_MAE_CRH = []
Log_GoodReads_MAE_GTM = []
Log_GoodReads_MAE_LFCN = []
Log_GoodReads_MAE_CTD = []
for val in GoodReads_MAE_KARS:
    Log_GoodReads_MAE_KARS.append(math.log(val))
for val in GoodReads_MAE_KDEm:
    Log_GoodReads_MAE_KDEm.append(math.log(val))
for val in GoodReads_MAE_CATD:
    Log_GoodReads_MAE_CATD.append(math.log(val))
for val in GoodReads_MAE_CRH:
    Log_GoodReads_MAE_CRH.append(math.log(val))
for val in GoodReads_MAE_GTM:
    Log_GoodReads_MAE_GTM.append(math.log(val))
for val in GoodReads_MAE_LFCN:
    Log_GoodReads_MAE_LFCN.append(math.log(val))
for val in GoodReads_MAE_CTD:
    Log_GoodReads_MAE_CTD.append(math.log(val))


# 画 真实指标
# plt.plot(DatasetSize,GoodReads_MAE_KARS, marker = 'o',color = 'r', label = 'KARS')
# plt.plot(DatasetSize,GoodReads_MAE_KDEm, marker = '^',color = 'fuchsia', label = 'KDEm')
# plt.plot(DatasetSize,GoodReads_MAE_CATD, marker = 's',color = 'deepskyblue', label = 'CATD')
# plt.plot(DatasetSize,GoodReads_MAE_CRH,  marker = '.',color = 'gray', label = 'CRH')
# plt.plot(DatasetSize,GoodReads_MAE_GTM,marker = 'D',color = 'gold', label = 'GTM')
# plt.plot(DatasetSize,GoodReads_MAE_LFCN, marker = 'v',color = 'g', label = 'LFC_N')

# 画 对数化指标
plt.plot(DatasetSize,Log_GoodReads_MAE_KARS, marker = 'o',color = 'r', label = 'KARS')
plt.plot(DatasetSize,Log_GoodReads_MAE_KDEm, marker = '^',color = 'fuchsia', label = 'KDEm')
plt.plot(DatasetSize,Log_GoodReads_MAE_CATD, marker = 's',color = 'deepskyblue', label = 'CATD')
plt.plot(DatasetSize,Log_GoodReads_MAE_CRH,  marker = '.',color = 'gray', label = 'CRH')
plt.plot(DatasetSize,Log_GoodReads_MAE_GTM,marker = 'D',color = 'gold', label = 'GTM')
plt.plot(DatasetSize,Log_GoodReads_MAE_LFCN, marker = 'v',color = 'g', label = 'LFC_N')
plt.plot(DatasetSize,Log_GoodReads_MAE_CTD,marker = 'p',color = 'peru', label = 'CTD')

plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xticks(np.arange(3,16,1))
plt.legend(loc='center left',bbox_to_anchor=(0, 0.65))
plt.xlabel("Redundancy",fontsize = 25)
# plt.title("GoodReads Dataset",fontsize = 30)
plt.ylabel("Log(MAE)",fontsize = 25)

plt.savefig('C:/Users/chopin/Desktop/'+'GoodReads_MAE_Methods_Comparison.pdf', bbox_inches='tight')
plt.show()





