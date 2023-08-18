import csv
import math
import sys
import time

sep = ","

def read_data(filename, workers, items):
    with open(filename) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(sep)
            value = float(parts[2])
            item_name = parts[1]
            worker_name = parts[0]
            workers.setdefault(worker_name, {'u': 0, 'data': [], 'weight': 0}).get('data').append((item_name, value))
            items.setdefault(item_name, {'data': [], 'truth': -1, 'old_truth': 0}).get('data').append((worker_name, value))

def initialise_truths(items):
    for item in items.values():
        tempdata = []
        for pair in item['data']:
            tempdata.append(pair[1])
        item['truth'] = sum(tempdata) / len(tempdata)

def compute_weights(workers, items):
    total_weight = 0.0
    for worker in workers.values():
        diffsum = 0.0
        for pair in worker['data']:
            item_name = pair[0]
            diffsum += (pair[1] - (items.get(item_name)['truth'])) ** 2
        diffsum /= len(worker['data'])
        if not diffsum:
            pass
        worker['weight'] = 1 / (diffsum + 1e-9)

def calculate_truths(workers, items):
    for item in items.values():
        total_weight = 0
        temp_truth = 0
        for pair in item['data']:
            temp_weight = workers[pair[0]]['weight']
            total_weight += temp_weight
            temp_truth += temp_weight * pair[1]
        item['truth'] = temp_truth / total_weight

def cal_variance(items):
    temp = 0.0
    for item in items.values():
        temp += (item['truth'] - item['old_truth']) ** 2
        item['old_truth'] = item['truth']
    return temp


# if __name__ == "__main__":
def l_LFCcont_Output(filename,truth_file):
    startTime = time.time()
    # if len(sys.argv) != 2:
    #     print "exit"
    #     exit(1)
    workers = {}
    items = {}
    # filename = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_201t_26818w/Y.csv'
    # truth_file = '../datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_201t_26818w/truth.csv'

    # filename = '../datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_122934w/Y.csv'
    # truth_file = '../datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_122934w/truth.csv'

    read_data(filename, workers, items)
    initialise_truths(items)
    iter = 50
    variance = float("inf")
    eps = 1e-6
    while iter > 0 and variance > eps:
        iter -= 1
        compute_weights(workers, items)
        calculate_truths(workers, items)
        variance = cal_variance(items)
    e2tr = {}
    for name, item in items.items():
        e2tr[name] = item['truth']

    w2q = {}
    for name, worker in workers.items():
        w2q[name] = worker['weight']

    # print(w2q)
    # print(e2tr)
    i = 0
    tcount1 = 0
    tcount2 = 0

    f = open(truth_file, 'r')
    reader = csv.reader(f)
    next(reader)
    e2T = {}
    for line in reader:
        task, truth = line
        e2T[task] = truth

    for e, tr in e2tr.items():
        tcount1 = tcount1 + math.fabs(float(e2T[e]) - tr)
        tcount2 = tcount2 + (float(e2T[e]) - tr) ** 2
        i += 1
    # print("l_LFCcont---MAE:", tcount1 / i)
    # print("l_LFCcont---RMSE:", pow(tcount2 / i, 0.5))

    endTime = time.time()
    runtime = int(endTime - startTime)
    return pow(tcount2 / i, 0.5) , tcount1 / i , runtime