# -*- coding: utf-8 -*-
"""
GTM.py

@author: Mengting Wan
"""
from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import csv
import math
import time

import numpy as np
import numpy.linalg as la
import GTM.basic_functions as bsf
from GTM.TruthFinder.TruthFinder import TruthFinder


def E_step(claim, index, m, n, sigma_vec, mu0, sigma0):
    truth = np.zeros(n)
    for i in range(n):
        tmp = mu0 / sigma0 ** 2 + sum(claim[i] / sigma_vec[index[i]] ** 2)
        tmp1 = 1 / sigma0 ** 2 + sum(1 / sigma_vec[index[i]] ** 2)
        truth[i] = tmp / tmp1
    return (truth)


def M_step(claim, index, m, n, truth, alpha, beta):
    sigma_vec = np.zeros(m)
    count = np.zeros(m)
    for i in range(n):
        sigma_vec[index[i]] = sigma_vec[index[i]] + 2 * beta + (claim[i] - truth[i]) ** 2
        count[index[i]] = count[index[i]] + 1
    sigma_vec = sigma_vec / (2 * (alpha + 1) + count)
    return (sigma_vec)


def Initialization(intl, claim, index, m, n, alpha, beta):
    if (len(intl) > 0):
        truth = np.copy(intl)
    else:
        truth = np.zeros(n)
        for i in range(n):
            truth[i] = np.median(claim[i])
    sigma_vec = M_step(claim, index, m, n, truth, alpha, beta)
    return ([truth, sigma_vec])


def GTM(data, m, n, intl=[], tol=1e-3, max_itr=99, alpha=10, beta=10, mu0=0, sigma0=1):
    err = 99
    index, claim, count = bsf.extract(data, m, n)

    itr = 0
    truth, sigma_vec = Initialization(intl, claim, index, m, n, alpha, beta)

    # truth, tau = TruthFinder.TruthFinder(data, m, n)
    while ((err > tol) & (itr < max_itr)):
        itr = itr + 1
        truth_old = np.copy(truth)
        truth = E_step(claim, index, m, n, sigma_vec, mu0, sigma0)
        sigma_vec = M_step(claim, index, m, n, truth, alpha, beta)
        err = la.norm(truth - truth_old) / la.norm(truth_old)
    return ([truth, sigma_vec])


def GTM_discret(data, m, n, intl=[], tol=1e-3, max_itr=99, alpha=10, beta=10, mu0=0, sigma0=1):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    truth, sigma_vec = Initialization(intl, claim, index, m, n, alpha, beta)
    # truth, tau = TruthFinder.TruthFinder(data, m, n)
    while ((err > tol) & (itr < max_itr)):
        itr = itr + 1
        truth_old = np.copy(truth)
        truth = E_step(claim, index, m, n, sigma_vec, mu0, sigma0)
        sigma_vec = M_step(claim, index, m, n, truth, alpha, beta)
        err = la.norm(truth - truth_old) / la.norm(truth_old)
        # print("err", err)
    truth = np.zeros(n)
    for i in range(n):
        truth[i] = claim[i][sigma_vec[index[i]].argmin()]
    return ([truth, sigma_vec])


# if __name__ == "__main__":
def GTM_Output(datafile , truthfile):
    startTime = time.time()
    # datafile = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_201t_26818w/Y_4conlums.csv'
    # truthfile = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_201t_26818w/truth.csv'

    # datafile = '../datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_122934w/Y_4conlums.csv'
    # truthfile = '../datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_122934w/truth.csv'


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
        label = float(label)
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

    truth_set, tau_vec = TruthFinder(data, n, m)
    truth_set = GTM(data, n, m, intl=truth_set)[0]
    f = open(truthfile, 'r')
    reader = csv.reader(f)
    next(reader)
    res = []
    i = 0
    tcount1 = 0
    tcount2 = 0
    for line in reader:
        task, truth = line
        tcount1 = tcount1 + math.fabs(float(truth) - truth_set[i])
        tcount2 = tcount2 + (float(truth) - truth_set[i]) ** 2
        i += 1
    # print("GTM---MAE:", tcount1 / i)
    # print("GTM---RMSE:", math.sqrt(tcount2 / i))
    endTime = time.time()
    runtime = int(endTime - startTime)
    return math.sqrt(tcount2 / i) , tcount1 / i , runtime

