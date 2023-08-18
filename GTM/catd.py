# -*- coding: utf-8 -*-
"""
CATD.py
@author: Mengting Wan
"""

from __future__ import division

import csv
import time
import math
import numpy as np
import GTM.basic_functions as bsf
import numpy.linalg as la
from scipy.stats import chi2
from GTM.TruthFinder import TruthFinder

def update_w(claim, index, count, truth, m, n, eps=1e-15):
    rtn = np.ones(m)
    for i in range(n):
        rtn[index[i]] = rtn[index[i]] + (claim[i] - truth[i]) ** 2

    rtn[rtn == 0] = 0.00001
    # for j in range(m):
    #     rtn[j] = chi2.ppf(0.025, count[j]) / rtn[j]
    # rtn[rtn == 0] = 1e10

    rtn[rtn > 0] = chi2.cdf(0.025, count[rtn > 0]) / rtn[rtn > 0]
    # rtn[rtn>0] = chi2.interval(0.05, count[rtn>0])[0]/rtn[rtn>0]
    return (rtn)


def update_truth(claim, index, w_vec, m, n):
    rtn = np.zeros(n)
    for i in range(n):
        rtn[i] = np.dot(w_vec[index[i]], claim[i]) / np.sum(w_vec[index[i]])

    return (rtn)


def CATD(data, m, n, intl=[], tol=0.1, max_itr=10):
    index, claim, count = bsf.extract(data, m, n)
    w_vec = np.ones(m)
    if (len(intl) > 0):
        truth = update_truth(claim, index, w_vec, m, n)
    else:
        truth = np.copy(intl)
    err = 99
    itr = 0
    while (err > tol and itr < max_itr):
        w_old = np.copy(w_vec)
        w_vec = update_w(claim, index, count, truth, m, n)
        truth = update_truth(claim, index, w_vec, m, n)
        err = la.norm(w_old - w_vec) / la.norm(w_old)
        itr = itr + 1
    return ([truth, w_vec])


def CATD_discret(data, m, n, intl=[]):
    index, claim, count = bsf.extract(data, m, n)
    w_vec = np.ones(m)
    if (len(intl) > 0):
        truth = update_truth(claim, index, w_vec, m, n)
    else:
        truth = np.copy(intl)
    w_vec = update_w(claim, index, count, truth, m, n)
    truth = np.zeros(n)
    for i in range(n):
        truth[i] = claim[i][w_vec[index[i]].argmax()]
    return ([truth, w_vec])

def CATD_Output(datafile, truth_file):
    startTime = time.time()
    # datafile = "../datasets/Y.csv"
    # truth_file = "../datasets/T.csv"
    # datafile = "./crowdsoucre/newGoodReads2_309t_5105w50r/Y2.csv"
    # truth_file = "./crowdsoucre/newGoodReads2_309t_5105w50r/truth.csv"
    data = []
    f = open(datafile, 'r')
    reader = csv.reader(f)
    worker_set = set()
    task_set = set()
    truth_arr = {}
    next(reader)
    w2tl = {}
    task_index_max = -1

    for line in reader:
        worker, task, label, truth = line
        worker = int(worker)
        task = int(task)
        label = int(label)
        truth = float(truth)

        truth_arr[task] = truth

        if worker not in worker_set:
            worker_set.add(worker)

        if task > task_index_max:
            task_index_max = task

        if task not in task_set:
            task_set.add(task)

        if worker not in w2tl:
            w2tl[worker] = {}
        w2tl[worker][task] = label

    for task in task_set:
        arr = []
        for worker in w2tl:
            if task in w2tl[worker]:
                arr.append([worker, w2tl[worker][task]])
        data.append(np.array(arr))

    n = len(worker_set)
    m = task_index_max + 1

    # ni*2 array
    data_array = np.array(data)
    truth_set = []
    truth_set, tau_vec = TruthFinder.TruthFinder(data, n, m)
    truth_set = CATD(data, n, m, intl=truth_set)[0]
    f = open(truth_file, 'r')
    reader = csv.reader(f)
    next(reader)
    res = []
    i = 0
    tcount1 = 0
    tcount2 = 0
    truthRet = ""
    for item in truth_set:
        truthRet += str(item) + ","
    # print(truthRet)
    for line in reader:
        task, truth = line
        task = int(task)

        # if truth_set[task] > 5 or truth_set[task] <= 0:
        #     continue
        # if truth_set[task] > 5 or truth_set[task] <= 0:
            # print(truth_set[task], float(truth))
        tcount1 = tcount1 + math.fabs(float(truth) - truth_set[task])
        tcount2 = tcount2 + (float(truth) - truth_set[task]) ** 2
        i += 1
    # print("CATD---MAE:", tcount1 / i)
    # print("CATD---RMSE:", pow(tcount2 / i, 0.5))
    # for line in reader:
    #     task, truth = line
    #
    #     if int(truth) == int(truth_set[i]):
    #         res.append(1)
    #     else:
    #         res.append(0)
    #     i += 1
    # print(sum(res)/len(res))

    mae = tcount1 / i
    rmse = pow(tcount2 / i, 0.5)

    endTime = time.time()
    runtime = int(endTime - startTime)
    return rmse, mae, runtime
