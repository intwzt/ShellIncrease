# coding:utf-8
from sklearn.cluster import KMeans
import numpy as np
import random
import matplotlib.pyplot as plt
from math import fabs

num_of_point = 100
num_of_cluster = 4
draw_color = ['red', 'green', 'blue', 'black']
target = 100000
step = 0.00001
change_range = [0.4, 0.3, 0.2, 0.1]
error = 0.001

gaussian_param = {'left': 10000, 'right': 15000, 'sigma_left': 1000, 'sigma_right': 5000}


def draw_scatter(data):
    plt.subplot(2, 2, 2)
    for index, group in enumerate(data):
        plt.scatter(group, group, marker='s', color=draw_color[index])
    plt.legend()
    plt.title('2. kmeans cluster')


def get_experience_partition():
    return [0.02, 0.04, 0.07, 0.11]


def gaussian_distribution(n):
    mu = random.randint(gaussian_param.get('left'), gaussian_param.get('right'))
    sigma = random.randint(gaussian_param.get('sigma_left'), gaussian_param.get('sigma_right'))
    print 'mu =', mu, ' sigma =', sigma
    np.random.seed(0)
    s = np.random.normal(mu, sigma, n)
    plt.subplot(2, 2, 1)
    plt.hist(s, 30)
    plt.title('1. origin data')
    return s


def choose_param(org, current, cluster_num):
    list_num = sorted(cluster_num.items(), key=lambda x: x[1], reverse=True)
    for i in range(4):
        index = list_num[i][0]
        if fabs(float(org[index]) - float(current[index])) <= float(org[index]) * change_range[index]:
            cluster_num[index] -= 1
            return index, cluster_num
    print 'adjust param base range!'
    return -1, cluster_num


def get_gradient(item_target, param):
    if item_target > target:
        return param - step
    else:
        return param + step


def get_item_target(el, increase):
    tmp_target = 0.0
    for i in range(num_of_cluster):
        tmp_target += el[i] * float(increase[i])
    return tmp_target


def gradient_descent(param, p_sum, cluster_num):
    flag = 1
    org_param = []
    org_sum = []

    param_set = []
    target_set = []

    item_target = 0
    for i in range(3, -1, -1):
        item_target += param[i] * p_sum[i]

    for i in range(4):
        org_param.append(param[i])
        org_sum.append(p_sum[i])

    target_set.append(item_target)
    param_set.append(param)

    while fabs(item_target - target) > target * error:
        # print 'current target =', item_target, ' param = ', param
        param_index, cluster_num = choose_param(org_param, param, cluster_num)
        if param_index < 0:
            flag = 0
            break
        param[param_index] = get_gradient(item_target, param[param_index])
        item_target = get_item_target(p_sum, param)

        target_set.append(item_target)
        tmp = []
        for i in range(4):
            tmp.append(param[i])
        param_set.append(tmp)

    if flag == 0:
        return flag, flag

    draw_trace(param_set, target_set)

    return item_target, param


def draw_trace(param_set, target_set):
    x_target_set = []
    for i in range(1, len(target_set) + 1):
        x_target_set.append(i)
    plt.subplot(2, 2, 3)
    plt.plot(x_target_set, target_set)
    plt.title('3. target trace')
    x_param_set = []
    for i in range(1, len(param_set) + 1):
        x_param_set.append(i)
    plt.subplot(2, 2, 4)
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    for i in range(len(param_set)):
        p1.append(param_set[i][0])
        p2.append(param_set[i][1])
        p3.append(param_set[i][2])
        p4.append(param_set[i][3])
    l1 = plt.plot(x_param_set, p1, color=draw_color[0])
    l2 = plt.plot(x_param_set, p2, color=draw_color[1])
    l3 = plt.plot(x_param_set, p3, color=draw_color[2])
    l4 = plt.plot(x_param_set, p4, color=draw_color[3])
    plt.legend(handles=[l1, l2, l3, l4, ], label=['1', '2', '3', '4'], loc='best')
    plt.ylim((0, 0.2))
    plt.title('4. param trace')


if __name__ == '__main__':
    data = []
    for point in gaussian_distribution(num_of_point).tolist():
        data.append([point])
    X = np.array(data)
    kmeans = KMeans(n_clusters=num_of_cluster).fit(X)
    vision_data = []
    for i in range(num_of_cluster):
        vision_data.append([])
    for i in range(num_of_point):
        index = kmeans.labels_[i]
        vision_data[index].append(float(data[i][0]))
    sum_data = []
    cluster_volume = {}
    for i in range(num_of_cluster):
        sum_data.append(sum(vision_data[i]))
        cluster_volume[i] = len(vision_data[i])
    sum_data = sorted(sum_data)
    draw_scatter(vision_data)
    target_trace = 0
    param_trace = []
    while target_trace == 0:
        target_trace, param_trace = gradient_descent(get_experience_partition(), sum_data, cluster_volume)
        for i in range(4):
            change_range[i] *= 1.5
    print 'target =', target_trace, ' param =', param_trace
    plt.show()
