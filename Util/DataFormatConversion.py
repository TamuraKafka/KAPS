import csv
import pandas as pd


def sort_truth():
    f = open('../datasets/WeatherSentiment/WeatherSentiment_truth.csv', mode='r', encoding='utf-8')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for row in reader:
        task_id, truth = row
        data.append([int(task_id), int(truth)])
    data.sort()

    df = pd.DataFrame(data, columns=['task_id', 'truth'])
    df.to_csv("./WeatherSentiment_truth.csv", index=False, header=True)


# 把 WeatherSentiment_amt.csv 文件的 worker_id, task_id 从数字1开始编号
def change_WeatherSentiment():
    date_file = '../datasets/WeatherSentiment/WeatherSentiment_amt.csv'
    f = open(date_file, 'r', encoding='utf-8')
    reader = csv.reader(f)
    reader.__next__()
    worker = set()
    task = set()
    for line in reader:
        worker.add(line[0])
        task.add(line[1])
    dic = {}
    dic2 = {}
    content = []
    count = 0
    for i in worker:
        dic[i] = count
        count = count + 1

    cnt = 0
    for j in task:
        dic2[j] = cnt
        cnt = cnt + 1

    # print(dic)
    # print(dic2)
    f = open(date_file, 'r', encoding='utf-8')
    reader = csv.reader(f)
    reader.__next__()
    for row in reader:
        worker_name, task_id, label, truth = row[0], row[1], row[2], row[3]
        worker_num = dic[worker_name]
        task_num = dic2[task_id]
        content.append([worker_num, task_num, label, truth])

    # print(content)
    df = pd.DataFrame(content, columns=['worker_id', 'task_id', 'label', 'truth'])
    df.to_csv("./WeatherSentiment_changed.csv", index=False, header=True)
    f.close()


def format_WeatherSentiment():
    date_file = '../datasets/WeatherSentiment/WeatherSentiment_changed.csv'
    f = open(date_file, 'r', encoding='utf-8')
    reader = csv.reader(f)
    content1 = []
    content2 = []
    content3 = []
    task_set = set()
    reader.__next__()

    for line in reader:
        worker_id, task_id, label, truth = line[0], line[1], line[2], line[3]
        if task_id not in task_set:
            task_set.add(task_id)
            content2.append([task_id, truth])
        content1.append([worker_id, task_id, label, truth])
        content3.append([worker_id, task_id, label])
    print(content2)

    df = pd.DataFrame(content1, columns=['worker_id', 'task_id', 'label', 'truth'])
    df.to_csv("./WeatherSentiment_4_columns.csv", index=False, header=True)

    df = pd.DataFrame(content2, columns=['task_id', 'truth'])
    df.to_csv('./WeatherSentiment_truth.csv', index=False, header=True)

    df = pd.DataFrame(content3, columns=['worker_id', 'task_id', 'label'])
    df.to_csv('./WeatherSentiment_3_columns.csv', index=False, header=True)


if __name__ == '__main__':
    sort_truth()

    # change_WeatherSentiment()

    #format_WeatherSentiment()
