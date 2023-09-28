#####################################
## RedundancyCut 数据集的实验结果图 ##
#####################################

import matplotlib.pyplot as plt
import numpy as np

##############  我的方法与baseline比较结果图  ######################################################################################################################################

############## Douban RMSE ###########
# fig = plt.figure()
#
# DatasetSize = [5,10,15,20,25,30,35,40,45,50]
#
# Douban_RMSE_HARS =  [0.87989, 0.96539 ,0.97720 ,0.99935 ,0.99004 ,0.98743 ,0.98224 ,0.98349 ,0.97998 ,0.96977 ]
# Douban_RMSE_KDEm =[1.03479,0.99629,1.02555,0.99421,1.02374,1.03715,1.05329,1.04452,1.04680,1.03083]
# # Douban_RMSE_CATD = [1.15848,1.05195,1.05500,1.06627,1.09238,1.08107,1.08636,1.11850,1.12594,1.12310]
# Douban_RMSE_CRH =[0.99046,0.99178,0.99449,1.00003,0.98850,0.97908,0.98100,0.98136,0.98075,0.98556]
# Douban_RMSE_GTM =[3.06876,2.46165,2.03105,1.65090,1.45471,1.30056,1.15992,1.05948,0.99590,0.95982]
# Douban_RMSE_LFCN = [1.16525, 1.17142 ,1.20892 ,1.24999 ,1.25502 ,1.28350 ,1.26481 ,1.28279 ,1.28866 ,1.29332 ]
#
#
# plt.plot(DatasetSize,Douban_RMSE_HARS,marker = 's',color = 'r', label = 'HARS')
# plt.plot(DatasetSize,Douban_RMSE_KDEm,marker = '*',color = 'g', label = 'KDEm')
# # plt.plot(DatasetSize,Douban_RMSE_CATD,marker = 'v',color = 'maroon', label = 'CATD')
# plt.plot(DatasetSize,Douban_RMSE_CRH,marker = 'o',color = 'b', label = 'CRH')
# # plt.plot(DatasetSize,Douban_RMSE_GTM,marker = '>',color = 'gold', label = 'GTM')
# plt.plot(DatasetSize,Douban_RMSE_LFCN,marker = '<',color = 'gray', label = 'LFC_N')
#
#
# plt.legend()
# plt.xlabel("DatasetSize")
# plt.title("Douban Dataset")
# plt.ylabel("RMSE")
# plt.show()




############# Douban MAE ###########
# fig = plt.figure()
#
# DatasetSize = [5,10,15,20,25,30,35,40,45,50]
#
# Douban_MAE_HARS =[0.69546,0.76131,0.78567,0.79597,0.78470,0.78241,0.77556,0.77696,0.77634,0.76680 ]
# Douban_MAE_KDEm = [0.83102,0.79471,0.84459,0.81490,0.84079,0.85238,0.86342,0.85892,0.86914,0.84961]
# # Douban_MAE_CATD = [0.84233,0.81949,0.84384,0.87068,0.89337,0.88909,0.88417,0.91673,0.92649,0.91815]
# Douban_MAE_CRH = [0.78361,0.79493,0.80277,0.79849,0.79425,0.78017,0.78084,0.78142,0.78094,0.78453]
# Douban_MAE_GTM = [2.79320,2.08194,1.66714,1.33429,1.17205,1.05151,0.94560,0.87168,0.82386,0.78485]
# Douban_MAE_LFCN = [0.96441, 0.97377 ,0.99627 ,1.02765 ,1.04435 ,1.07477 ,1.05350 ,1.07987 ,1.08547, 1.09270 ]
#
#
# plt.plot(DatasetSize,Douban_MAE_HARS,marker = 's',color = 'r', label = 'HARS')
# plt.plot(DatasetSize,Douban_MAE_KDEm,marker = '*',color = 'g', label = 'KDEm')
# # plt.plot(DatasetSize,Douban_MAE_CATD,marker = 'v',color = 'maroon', label = 'CATD')
# plt.plot(DatasetSize,Douban_MAE_CRH,marker = 'o',color = 'b', label = 'CRH')
# # plt.plot(DatasetSize,Douban_MAE_GTM,marker = '>',color = 'gold', label = 'GTM')
# plt.plot(DatasetSize,Douban_MAE_LFCN,marker = '<',color = 'gray', label = 'LFC_N')
#
# plt.legend()
# plt.xlabel("DatasetSize")
# plt.title("Douban Dataset")
# plt.suptitle("")
# plt.ylabel("MAE")
# plt.show()




############# GoodReads RMSE ###########
fig = plt.figure()

DatasetSize = [5,10,15,20,25,30,35,40,45,50]

GoodReads_RMSE_HARS =[0.56558, 0.47790 ,0.47158 ,0.46913 ,0.46185 ,0.45180 ,0.44035 ,0.43361 ,0.43825, 0.42977 ]
GoodReads_RMSE_KDEm =[0.86353,0.80113,0.69374,0.67393,0.63412,0.60040,0.55798,0.66471,0.59829,0.58251]
# GoodReads_RMSE_CATD =[1.24773 ,0.86663 ,0.60686 ,0.75282 ,0.83503 ,3.18202 ,0.84671 ,1.15239 ,0.65136 ,0.76660 ]
GoodReads_RMSE_CRH =[0.59531,0.52610,0.50361,0.48550,0.46978,0.45702,0.44346,0.43541,0.43250,0.42793]
GoodReads_RMSE_GTM =[3.39104,2.95378,2.64141,2.34422,2.12998,1.97584,1.80803,1.65121,1.54932,1.42808]
GoodReads_RMSE_LFCN =[0.66804, 0.57190, 0.52621, 0.47401, 0.46962, 0.45271, 0.45240, 0.43623, 0.43731, 0.42500 ]


plt.plot(DatasetSize,GoodReads_RMSE_HARS,marker = 's',color = 'r', label = 'HARS')
plt.plot(DatasetSize,GoodReads_RMSE_KDEm,marker = '*',color = 'g', label = 'KDEm')
# plt.plot(DatasetSize,GoodReads_RMSE_CATD,marker = 'v',color = 'maroon', label = 'CATD')
plt.plot(DatasetSize,GoodReads_RMSE_CRH,marker = 'o',color = 'b', label = 'CRH')
plt.plot(DatasetSize,GoodReads_RMSE_GTM,marker = '>',color = 'gold', label = 'GTM')
plt.plot(DatasetSize,GoodReads_RMSE_LFCN,marker = '<',color = 'gray', label = 'LFC_N')

plt.legend()
plt.xlabel("DatasetSize")
plt.title("GoodReads Dataset")
plt.ylabel("RMSE")
plt.show()




############## GoodReads MAE ###########

# fig = plt.figure()
#
# DatasetSize = [5,10,15,20,25,30,35,40,45,50]
#
# GoodReads_MAE_HARS =[0.44636, 0.37957 ,0.38468 ,0.38809 ,0.38363 ,0.37242 ,0.36996 ,0.36565, 0.36837, 0.35780 ]
# GoodReads_MAE_KDEm =[0.62760,0.60440,0.52099,0.48979,0.46031,0.45232,0.41079,0.45750,0.41050,0.40292]
# #GoodReads_MAE_CATD = [0.77169, 0.73063 ,0.72408 ,0.73291 ,0.78311 ,0.67730 ,0.64321 ,0.75231 ,0.61615 ,0.60175 ]
# GoodReads_MAE_CRH =[0.47459,0.42529,0.41035,0.40155,0.38828,0.37752,0.36737,0.36014,0.35520,0.35068]
# GoodReads_MAE_GTM =[3.17694,2.65005,2.30915,1.99580,1.78319,1.63686,1.48456,1.34825,1.26067,1.15960]
# GoodReads_MAE_LFCN =[0.53963,0.45088,0.42087,0.37526,0.38104,0.36376,0.36609,0.35477,0.35561,0.34369]
#
#
# plt.plot(DatasetSize,GoodReads_MAE_HARS,marker = 's',color = 'r', label = 'HARS')
# plt.plot(DatasetSize,GoodReads_MAE_KDEm,marker = '*',color = 'g', label = 'KDEm')
# # plt.plot(DatasetSize,GoodReads_MAE_CATD,marker = 'v',color = 'maroon', label = 'CATD')
# plt.plot(DatasetSize,GoodReads_MAE_CRH,marker = 'o',color = 'b', label = 'CRH')
# # plt.plot(DatasetSize,GoodReads_MAE_GTM,marker = '>',color = 'gold', label = 'GTM')
# plt.plot(DatasetSize,GoodReads_MAE_LFCN,marker = '<',color = 'gray', label = 'LFC_N')
#
# plt.legend()
# plt.xlabel("DatasetSize")
# plt.title("GoodReads Dataset")
# plt.ylabel("MAE")
# plt.show()







