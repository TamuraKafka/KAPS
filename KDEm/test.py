import csv
import math
import time

import numpy as np

import KDEm.basic_functions as bsf
from KDEm import KDEm

def test_KDEm(data_raw, m, n, kernel, norm=True, outlier_thr=0, max_itr=99, argmax=False, h=-1):
    # print("Test with KDEm...")
    # print("Kernel:", kernel)
    if(norm):
        # print("Normalized: True")
        data, data_mean, data_sd = bsf.normalize(data_raw)
    else:
        # print("Normalized: False")
        data = data_raw[:]
    a = time.time()

    source_score, weights_for_each, itr = KDEm.KDEm(data, m, n, max_itr=max_itr, method=kernel, h=h)
    b = time.time() - a
    # print("Time cost for each iteration in KDEm: "+str(b)+"s")
    out, cluster_index, cluster_confidence = bsf.wKDE_twist(data, m, n, weights_for_each, kernel, argmax, outlier_thr, h)
    c = time.time() - a
    #print "Time cost for all: "+str(c)+"s"
    moments = get_moments(data, m, n, weights_for_each, method=kernel, h=h)
    if(norm):
        truth_out = bsf.normalize_ivr(out, data_mean, data_sd)
    else:
        truth_out = out[:]
    # print("End.")
    return([truth_out, cluster_index, cluster_confidence, source_score, weights_for_each, moments, [b/itr,c]])

def get_moments(data, m, n, w_M, method="gaussian", h=-1):
    moments = np.zeros((n,3))
    for i in range(n):
        x_i = np.copy(data[i][:,1])
        if(len(w_M)>0):
            moments[i,:] = bsf.get_moments(x_i, w_M[i], h)
        else:
            moments[i,:] = bsf.get_moments(x_i, np.ones(len(x_i))/len(x_i), h)
    return(moments)

# if __name__ == "__main__":
def KDEm_Output(datafile, truth_file):
    startTime = time.time()
    # datafile = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_201t_26818w/Y_4conlums.csv'
    # truthfile = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_201t_26818w/truth.csv'

    # datafile = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_2t_1057w/Y2.csv'
    # truthfile = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_2t_1057w/truth.csv'


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
    rtn = test_KDEm(data, n, m, kernel="gaussian", norm=True)
    out1, cluster_index, conf1, source_score, weights_for_each, moments1, time1 = rtn
    truth_set = []
    for item in out1:
        # print(item[0])
        truth_set.append(item[0])

    f = open(truth_file, 'r')
    reader = csv.reader(f)
    next(reader)
    res = []
    i = 0
    tcount1 = 0
    tcount2 = 0
    for line in reader:
        task, truth = line
        # print(float(truth), truth_set[i])
        tcount1 = tcount1 + math.fabs(float(truth) -truth_set[i])
        tcount2 = tcount2 + (float(truth) - truth_set[i]) ** 2
        i += 1
    # print("KDEm---MAE:", tcount1/i)
    # print("KDEm---RMSE:", pow(tcount2/i,0.5))
    endTime = time.time()
    runtime = int(endTime - startTime)
    return pow(tcount2/i,0.5) , tcount1/i, runtime

    # i = 0
    # for line in reader:
    #     task, truth = line
    #
    #     if int(truth) == int(truth_set[i]):
    #         res.append(1)
    #     else:
    #         res.append(0)
    #     i += 1
    # print(sum(res) / len(res))

