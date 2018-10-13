# -*- coding:utf-8 -*-
import numpy as np
'''
    y = sigmoid(- w * x)
    R_w = -sum(y * log(y_train) + (1-y) * log(1 - y_train)) / N
    dw = -sum(x * (y - y_train)) / N
'''


def sigmoid(train_set, w):
    result = []
    for data in train_set:
        data = 1 / (1 + np.exp(- w * data))
        result.append(data)
    return result


def logistic_regression(learning_rate = 0.1, epoch_num = 10):
    # init data set
    x = np.random.random_sample(30)*10
    labels = np.random.randint(0, 2, 30)
    number = len(labels)
    w = 0
    # y = sigmoid(w*x)
    for epoch in range(epoch_num):
        y_train = np.asarray(sigmoid(x, w))
        dw = -np.sum(labels - y_train)
        w -= learning_rate * dw
    return w


w = logistic_regression()

