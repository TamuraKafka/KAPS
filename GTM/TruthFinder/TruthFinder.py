# -*- coding: utf-8 -*-
"""
TruthFinder.py

@author: Mengting Wan
"""


from __future__ import division

import csv

import numpy as np
import numpy.linalg as la
import GTM.basic_functions as bsf


def update_source(claim, index, s_set, m, n):
    t_vec = np.zeros(m)
    tau_vec = np.zeros(m)
    count = np.zeros(m)
    for i in range(n):
        t_vec[index[i]] = t_vec[index[i]] + s_set[i]
        count[index[i]] = count[index[i]] + 1
    t_vec[count>0] = t_vec[count>0]/count[count>0]
    tau_vec[t_vec>=1] = np.log(1e10)
    tau_vec[t_vec<1] = -np.log(1-t_vec[t_vec<1])
    return(tau_vec)
    
def update_claim(claim, index, tau_vec, m, n, rho, gamma, base_thr=0):
    s_set= []
    for i in range(n):
        claim_set = list(set(claim[i]))
        sigma_i = np.zeros(len(claim_set))
        s_vec = np.zeros(len(claim[i]))
        for j in range(len(claim_set)):
            sigma_i[j] = sum(tau_vec[index[i]][claim[i]==claim_set[j]])
        tmp_i = np.copy(sigma_i)
        for j in range(len(claim_set)):
            tmp_i[j] = (1-rho*(1-base_thr))*sigma_i[j] + rho*sum((np.exp(-abs(claim_set-claim_set[j]))-base_thr)*sigma_i)
            #tmp_i[j] = (1+rho)*sigma_i[j] + rho*sum(-sigma_i)
            s_vec[claim[i]==claim_set[j]] = 1/(1 + np.exp(-gamma*tmp_i[j]))
        s_set.append(s_vec)
    return(s_set)

def TruthFinder(data, m, n, tol=0.1, max_itr=10):
    err = 99
    index, claim, count = bsf.extract(data, m, n)
    itr = 0
    tau_vec = -np.log(1-np.ones(m)*0.9)
    truth = np.zeros(n)
    rho = 0.5
    gamma = 0.3
    while((err > tol) & (itr < max_itr)):
        itr = itr+1
        tau_old = np.copy(tau_vec)
        s_set = update_claim(claim, index, tau_vec, m, n, rho, gamma)
        tau_vec = update_source(claim, index, s_set, m, n)
        err = 1 - np.dot(tau_vec,tau_old)/(la.norm(tau_vec)*la.norm(tau_old))
        # print(itr, err)
    truth = np.zeros(n)

    for i in range(n):
        truth[i] = claim[i][np.argmax(s_set[i])]

    return([truth, tau_vec])

if __name__ == "__main__":
    datafile = '../datasets/WeatherSentiment/WeatherSentiment_4_columns.csv'
    truth_file = '../datasets/WeatherSentiment/WeatherSentiment_truth.csv'

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
        truth = int(truth)

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
    # print(truth_set)
    f = open(truth_file, 'r')
    reader = csv.reader(f)
    next(reader)
    res = []
    i = 0
    taskToanswer = {}
    for line in reader:
        task, truth = line
        if task not in taskToanswer:
            taskToanswer[task] = truth

    for task_id in task_set:
        if int(taskToanswer[str(task_id)]) == int(truth_set[i]):
            res.append(1)
        else:
            res.append(0)
        i += 1
    #print(sum(res)/len(res))
