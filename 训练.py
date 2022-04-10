import numpy as np
from numpy.random import randn
import random
import math
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def loss_L2(X, w1, w2, y, lambd):
    h = 1.0 / (1.0 + np.exp(-X.dot(w1)))  # 激活函数
    y_pred = h.dot(w2)
    loss = np.square(y_pred-y).sum()
    # L2正则化
    L2_regularization_cost = lambd * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    loss = loss + L2_regularization_cost  # loss计算
    # 反向传播及梯度计算
    dy_pred = 2.0*(y_pred-y)
    dw2 = h.T.dot(dy_pred) + lambd * w2 * 2.0
    dh = dy_pred.dot(w2.T)
    dw1 = X.T.dot(dh*h*(1-h)) + lambd * 2.0 * w1
    return loss, dw1, dw2

def predict(X, w1, w2):
    h = 1.0/(1.0+np.exp(-X.dot(w1)))
    y_pred = h.dot(w2)
    y_pred = (y_pred == y_pred.max(axis=1, keepdims=1)).astype(float)
    return y_pred


def count_loss(X, w1, w2, y, lambd):
    h = 1.0 / (1.0 + np.exp(-X.dot(w1)))  # 激活函数
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    # L2正则化
    L2_regularization_cost = lambd * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    loss = loss + L2_regularization_cost  # loss计算
    return loss


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签
    # load data
    Xtrain = mnist.train.images  # 训练集样本
    Xvalid = mnist.validation.images  # 验证集样本
    Xtest = mnist.test.images  # 测试集样本
    # labels
    Ytrain = mnist.train.labels  # 训练集标签
    Yvalid = mnist.validation.labels  # 验证集标签
    Ytest = mnist.test.labels  # 测试集标签

    # shape参数
    n, d = Xtrain.shape
    t = Xvalid.shape[0]
    t2 = Xtest.shape[0]

    # 添加偏置项
    Xtrain = np.c_[np.ones(n), Xtrain]
    d = d + 1
    Xvalid = np.c_[np.ones(t), Xvalid]
    Xtest = np.c_[np.ones(t2), Xtest]

    # nLabels
    nLabels = 10

    # Choose network structure
    nHidden = np.array([50])

    # Count number of parameters and initialize weights 'w'
    nParams = d * nHidden  # 参数数目
    for h in range(1, len(nHidden)):
        nParams = nParams + nHidden[h - 1] * nHidden[h]
    nParams = nParams + nHidden[-1] * nLabels

    # 随机确定初始w
    w1 = randn(d, nHidden[0])
    w2 = randn(nHidden[0], nLabels)

    # Train with stochastic gradient
    maxIter = 100000
    stepSize = 1e-3
    minibatch = 32
    lamda = 0.032
    gama = 0.9
    # print('Training iteration = {}, train error = {}'.format(iter, error1))
    # print('Training iteration = {}, validation error = {}'.format(iter, error2))
    # Train with stochastic gradient
    train_accuracy = []
    valid_accuracy = []
    train_loss = []
    valid_loss = []
    for iter in range(maxIter):
        if iter % round(maxIter / 20) == 0:
            yhat_train = predict(Xtrain, w1, w2)
            yhat_valid = predict(Xvalid, w1, w2)
            error1 = (yhat_train != Ytrain).sum() / n / 2.0
            error2 = (yhat_valid != Yvalid).sum() / t / 2.0
            train_accuracy.append(1 - error1)
            valid_accuracy.append(1 - error1)
            # 计算loss
            valid_loss.append(count_loss(Xvalid, w1, w2, Yvalid, lamda))

            train_loss.append(count_loss(Xtrain, w1, w2, Ytrain, lamda))
            stepSize = stepSize * gama  # 学习率下降策略
            print('Training iteration = {}, train error = {}'.format(iter, error1))
            print('Training iteration = {}, validation error = {}'.format(iter, error2))
            # Train with stochastic gradient

        # 优化器SGD
        i = [math.floor(random.random() * n) for k in range(minibatch)]
        l, dw1, dw2 = loss_L2(Xtrain[i], w1, w2, Ytrain[i], lamda)
        w1 = w1 - stepSize * dw1
        w2 = w2 - stepSize * dw2

    yhat = predict(Xtest, w1, w2)
    print('Test error with final model = {}'.format((yhat != Ytest).sum() / t2 / 2.0))
