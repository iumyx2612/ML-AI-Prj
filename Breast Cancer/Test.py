import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import sys
np.set_printoptions(threshold=sys.maxsize)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def processAgeinX(X_age):
    temp = X_age.copy()
    age_dict = {}
    for _ in range(len(X_age)):
        if X_age[_] not in age_dict:
            temp_string = X_age[_].split("-")
            value = (int(temp_string[1]) + int(temp_string[0])) / 2
            age_dict[X_age[_]] = value
    for _ in range(len(X_age)):
        temp[_] = age_dict[temp[_]]
    return temp


def processTumorSizeinX(X_tumor):
    temp = X_tumor.copy()
    tumor_dict = {}
    for _ in range(len(X_tumor)):
        if X_tumor[_] not in tumor_dict:
            temp_string = X_tumor[_].split("-")
            value = (int(temp_string[1]) + int(temp_string[0])) / 2
            tumor_dict[X_tumor[_]] = value
    for _ in range(len(X_age)):
        temp[_] = tumor_dict[temp[_]]
    return temp


def processInvNodesinX(X_inv_nodes):
    temp = X_inv_nodes.copy()
    inv_nodes_dict = {}
    for _ in range(len(X_inv_nodes)):
        if X_inv_nodes[_] not in inv_nodes_dict:
            temp_string = X_inv_nodes[_].split("-")
            value = (int(temp_string[1]) + int(temp_string[0])) / 2
            inv_nodes_dict[X_inv_nodes[_]] = value
    for _ in range(len(X_age)):
        temp[_] = inv_nodes_dict[temp[_]]
    return temp


def labelEncode(y):
    temp = y.copy()
    label_dict = {}
    counter = 0
    for _ in range(y.size):
        if y[_] not in label_dict:
            label_dict[y[_]] = counter
            counter += 1
    for _ in range(y.size):
        temp[_] = label_dict[temp[_]]
    return temp


def costFunction(theta, X, y, lambda_=0.1):
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    for i in range(m):
        minus = sigmoid(np.dot(theta.T, X[i])) - y[i]
        grad += minus * X[i]
    for i in range(m):
        hx = sigmoid(np.dot(theta.T, X[i]))
        J += (y[i] * np.log(hx)) + (1 - y[i]) * np.log(1 - hx)
    grad *= 1 / m
    J = -1 / m * J
    # reg term
    J = J + lambda_/(2*m) * np.sum(theta[1:] ** 2)
    grad[1:] = grad[1:] + lambda_/m * theta[1:]
    # =============================================================
    return J, grad


def lrCostFunction(theta, X, y, lambda_=0.1):
    m = y.size
    if y.dtype == bool:
        y = y.astype(int)
    J = 0
    grad = np.zeros(theta.shape)
    one = np.dot(y, np.log(sigmoid(np.dot(X, theta))))
    two = np.dot((1-y), np.log(sigmoid(1-np.dot(X, theta))))
    J = -1/m * (one + two)
    grad = 1 / m * np.dot(sigmoid(np.dot(X, theta.T)) - y, X)

    #add regularization
    J = J + lambda_/(2*m) * np.sum(theta[1:] ** 2)
    grad[1:] = grad[1:] + lambda_/m * theta[1:]
    return J, grad


if __name__ == '__main__':
    # create a proper csv file for pandas to read
    with open("breast-cancer.data") as f:
        data = f.read()
    new_data = "Class,age,menopause,tumor-size,inv-nodes,node-caps,deg-malig,breast,breast-quad,irradiat\n" + data
    with open("Breast-Data.csv", 'w') as file:
        file.write(new_data)

    # pre-processing data
    data = pd.read_csv("Breast-Data.csv", delimiter=',')
    data = data[
        data["node-caps"] != "?"]  # sau khi inspect node-caps ở dưới, cta thấy có giá trị "?" nên cần phải drop đi
    data = data.reset_index()  # reset lại index vì sau khi bỏ đi thì hàng bị bỏ k tự nhảy index
    features = ["age", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "irradiat"]
    X = data[features]
    # print(X["node-caps"].isnull().sum()) # check if any data is missing
    y = data["Class"]
    y = np.array(y)
    y = labelEncode(y)

    # process X data
    # age
    X_age = X["age"]
    X_age = processAgeinX(X_age)
    X_age = np.array(X_age).T
    # tumor-size
    X_turmor = X["tumor-size"]
    X_turmor = np.array(processTumorSizeinX(X_turmor)).T
    # inv-nodes
    X_inv_nodes = X["inv-nodes"]
    X_inv_nodes = np.array(processInvNodesinX(X_inv_nodes)).T
    # node-caps
    X_node_caps = X["node-caps"]
    X_node_caps = np.array(labelEncode(X_node_caps)).T
    # irrediat
    X_irrediat = X["irradiat"]
    X_irrediat = np.array(labelEncode(X_irrediat)).T
    # deg-malig
    X_deg_malig = np.array(X["deg-malig"]).T
    # final X
    X = np.array((X_age, X_turmor, X_inv_nodes, X_node_caps, X_deg_malig, X_irrediat), dtype=np.float).T
    m, n = X.shape
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    lambda_ = 0.1
    cost, grad = costFunction(initial_theta, X, y)
    other_cost, other_grad = lrCostFunction(initial_theta, X, y)
    print(cost, grad)
    print(grad.dtype)
    print(other_cost, other_grad)
    print(other_grad.dtype)
    # set options for optimize.minimize
    options = {'maxiter': 50}
    # see documention for scipy's optimize.minimize  for description about
    # the different parameters
    # The function returns an object `OptimizeResult`
    # We use truncated Newton algorithm for optimization which is
    # equivalent to MATLAB's fminunc
    # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
    res = optimize.minimize(costFunction,
                            initial_theta,
                            (X, y, lambda_),
                            jac=True,
                            method='TNC',
                            options=options)
    print(res.x)