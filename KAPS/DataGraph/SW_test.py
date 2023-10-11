##################
## SW 测试 画图  ##
##################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import math



###############################  两个数据集画一张图上  ##########################################
DataClippingScale = [0,0.05,0.1,0.15,0.2,0.25,.3,0.35,0.4,0.45,0.5]
Douban_RMSE_HARS =  [0.9549024787895418,0.949014157871162,0.9444667233384225,0.9411886644343485,0.9319225402593098,0.9280813927301313,0.9238979301235463,0.9210864737708325,0.9160688700233759,0.9079389320953095,0.9083231779606425]
Douban_MAE_HARS =  [0.7574701683414208,0.7516614450006354,0.7491603526689402,0.7465377691537911,0.7381292168001409,0.7346272286056329,0.7313213118324855, 0.7284021735554137,0.7239129176001797,0.716038145913582,0.7194054584116987]
GoodReads_RMSE_HARS =  [0.40338203631556063, 0.4038697484247689, 0.40748551008459044, 0.4080572894818407, 0.41365923151225575, 0.41840602055197057, 0.4152086982641798, 0.41946050424241754, 0.4240696091718623, 0.4297141947376352, 0.4247464142412669]
GoodReads_MAE_HARS =  [0.32508660558184677, 0.32445000792669704, 0.32916187591465074, 0.332557379275999, 0.3374146421315535, 0.34404708311964205, 0.3409725140367548, 0.34532015553873235, 0.34861563984987226, 0.3552309963754425, 0.3495219951694368]


####### RMSE  #############

fig = plt.figure(1,figsize=(6, 6))

ax1 = plt.subplot(211)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax1.plot(DataClippingScale,Douban_RMSE_HARS,color = 'red',label = 'Douban')
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15,rotation = 270)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0,0.55,0.05))
plt.ylim((0.9,1.0))
plt.ylabel("RMSE",fontsize = 25)
plt.legend(loc='upper left',fontsize = 15)
plt.grid(axis='both', alpha=0.3)



ax2 = plt.subplot(212)
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax2.plot(DataClippingScale,GoodReads_RMSE_HARS,color = 'limegreen',label = 'GoodReads')
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15,rotation = 270)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0,0.55,0.05))
plt.ylim((0.36,0.46))
plt.xlabel("Data Clipping Scale",fontsize = 25)
plt.ylabel("RMSE",fontsize = 25)
plt.legend(loc='upper left',fontsize = 15)
plt.grid(axis='both', alpha=0.3)


plt.savefig('C:/Users/chopin/Desktop/'+'SW_RMSE.pdf', bbox_inches='tight')

plt.show()









####### MAE  #############

fig = plt.figure(1,figsize=(6, 6))

ax1 = plt.subplot(211)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax1.plot(DataClippingScale,Douban_MAE_HARS,color = 'red',label = 'Douban')
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15,rotation = 270)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0,0.55,0.05))
plt.ylim((0.7,0.8))
plt.ylabel("MAE",fontsize = 25)
plt.legend(loc='upper left',fontsize = 15)
plt.grid(axis='both', alpha=0.3)


ax2 = plt.subplot(212)
plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax2.plot(DataClippingScale,GoodReads_MAE_HARS,color = 'limegreen',label = 'GoodReads')
plt.gca().spines["top"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.3)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["left"].set_alpha(0.3)
plt.xticks(fontsize=15,rotation = 270)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0,0.55,0.05))
plt.ylim((0.3,0.4))
plt.xlabel("Data Clipping Scale",fontsize = 25)
plt.ylabel("MAE",fontsize = 25)
plt.legend(loc='upper left',fontsize = 15)
plt.grid(axis='both', alpha=0.3)


plt.savefig('C:/Users/chopin/Desktop/'+'SW_MAE.pdf', bbox_inches='tight')

plt.show()


