# -*- coding: utf-8 -*-
"""
CRH.py

@author: Mengting Wan
"""

from __future__ import division
import time
import csv
import math

import numpy as np
import numpy.linalg as la
import GTM.basic_functions as bsf


def update_w(claim, index, truth, m, n, eps=1e-15):
    rtn = np.zeros(m)
    for i in range(n):
        rtn[index[i]] = rtn[index[i]] + np.fabs(claim[i] - truth[i]) / max(np.std(claim[i]), eps)
    tmp = np.sum(rtn)
    if (tmp > 0):
        rtn[rtn > 0] = np.copy(-np.log(rtn[rtn > 0] / tmp))
    return (rtn)


def update_truth(claim, index, w_vec, m, n):
    rtn = np.zeros(n)
    for i in range(n):
        rtn[i] = np.dot(w_vec[index[i]], claim[i]) / np.sum(w_vec[index[i]])

    return (rtn)


def CRH(data, m, n, tol=1e-3, max_itr=99):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    w_vec = np.ones(m)
    truth = np.zeros(n)
    while ((err > tol) & (itr < max_itr)):
        itr = itr + 1
        truth_old = np.copy(truth)
        truth = update_truth(claim, index, w_vec, m, n)
        w_vec = update_w(claim, index, truth, m, n)

        err = la.norm(truth - truth_old) / la.norm(truth_old)
    return ([truth, w_vec])


def CRH_discret(data, m, n, tol=1e-3, max_itr=99):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    w_vec = np.ones(m)
    truth = np.zeros(n)
    while ((err > tol) & (itr < max_itr)):
        itr = itr + 1
        truth_old = np.copy(truth)
        truth = update_truth(claim, index, w_vec, m, n)
        w_vec = update_w(claim, index, truth, m, n)
        err = la.norm(truth - truth_old) / la.norm(truth_old)
    truth = np.zeros(n)
    for i in range(n):
        truth[i] = claim[i][w_vec[index[i]].argmax()]
    return ([truth, w_vec])


# if __name__ == "__main__":
def CRH_Output(datafile, truth_file):
    startTime = time.time()
    # datafile = "../datasets/Y.csv"
    # truth_file = "../datasets/T.csv"
    # datafile = "./crowdsoucre/newDouban2_3r_0_202t_130w/Y2.csv"
    # truth_file = "./crowdsoucre/newDouban2_3r_0_202t_130w/truth.csv"
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

    truth_set = CRH(data, n, m)[0]

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
    # (truthRet)

    tmp = 0
    for line in reader:
        task, truth = line
        if np.isnan(truth_set[i]):
            i += 1
            tmp += 1
            continue
        tcount1 = tcount1 + math.fabs(float(truth) - truth_set[i])
        tcount2 = tcount2 + (float(truth) - truth_set[i]) ** 2
        i += 1
    # print("CRH---MAE:", tcount1 / (i - tmp))
    # print("CRH---RMSE:", pow(tcount2 / (i - tmp), 0.5))
    # # for line in reader:
    #     task, truth = line
    #
    #     if int(truth) == int(truth_set[i]):
    #         res.append(1)
    #     else:
    #         res.append(0)
    #     i += 1
    # print(sum(res)/len(res))
    rmse = pow(tcount2 / (i - tmp), 0.5)
    mae = tcount1 / (i - tmp)

    endTime = time.time()
    runtime = int(endTime - startTime)
    return rmse, mae, runtime
