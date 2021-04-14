import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import sklearn.linear_model
import os
import sys

np.set_printoptions(threshold=sys.maxsize)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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
    for _ in range(len(X_tumor)):
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
    for _ in range(len(X_inv_nodes)):
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


def logisticRegressionCostFunc(theta, X, y, lambda_=0.1):
    m = y.size
    J = 0

    first_term = np.dot(y, np.log(sigmoid(np.dot(X, theta.T))))
    second_term = np.dot(1 - y, np.log(sigmoid(1 - np.dot(X, theta.T))))
    J = -1 / m * (first_term + second_term)
    reg_term = lambda_ / (2 * m) * np.sum(theta[1:] ** 2)
    J += reg_term
    return J


def logisticRegressionGradient(theta, X, y, lambda_=0.1):
    m = y.size
    grad = 1 / m * np.dot(sigmoid(np.dot(X, theta.T)) - y, X)
    grad[1:] = grad[1:] + lambda_ / m * theta[1:]
    return np.array(grad, np.float)


def calculateTheta(X, y, lambda_):
    if X.ndim == 1:
        X = X[None]
    options = {'maxiter': 50}
    thetas = np.zeros(X.shape[1]).T
    res = optimize.minimize(logisticRegressionCostFunc,
                            thetas,
                            (X, y, lambda_),
                            jac=logisticRegressionGradient,
                            method='TNC',
                            options=options)
    theta = res.x
    return theta


def predict(theta, X):
    result = sigmoid(np.dot(theta, X.T))
    if result >= 0.5:
        return 1, result
    else:
        return 0, result


if __name__ == '__main__':
    # create a proper csv file for pandas to read
    with open("breast-cancer.data") as f:
        data = f.read()
    new_data = "Class,age,menopause,tumor-size,inv-nodes,node-caps,deg-malig,breast,breast-quad,irradiat\n" + data
    with open("Breast-Data.csv", 'w') as file:
        file.write(new_data)

    # pre-processing data
    data = pd.read_csv("Breast-Data.csv", delimiter=',')
    data = data[data["node-caps"] != "?"]  # sau khi inspect node-caps ở dưới, cta thấy có giá trị "?" nên cần phải drop đi
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.reset_index()  # reset lại index vì sau khi bỏ đi thì hàng bị bỏ k tự nhảy index
    features = ["age", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "irradiat"]
    X = data[features]
    # print(X["node-caps"].isnull().sum()) # check if any data is missing
    y = data["Class"]
    y = labelEncode(y)
    y = np.array(y, dtype=np.float)

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
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # inspect data
    y0 = np.array(y[np.where(y == 0)]) # no = 0
    y1 = np.array(y[np.where(y == 1)]) # yes = 1
    plt.bar(['0', '1'], [len(y0), len(y1)])
    plt.show()

   # make train, validation and test sets
    m, n = X.shape
    X_train = X[:int(0.6 * m), :]
    X_validation = X[int(0.6 * m):int(0.8 * m), :]
    X_test = X[int(0.8 * m):, :]

    y_train = y[:int(0.6 * m)]
    y_validation = y[int(0.6 * m):int(0.8 * m)]
    y_test = y[int(0.8 * m):]

    theta = np.zeros((1, n))
    print(logisticRegressionCostFunc(theta, X_train, y_train))

    # train
    lambda_ = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 50]
    all_theta = np.zeros((len(lambda_), X.shape[1]))
    for i in range(len(lambda_)):
        theta = calculateTheta(X_train, y_train, lambda_[i])
        print("lambda uses:" + str(lambda_[i]))
        print("Theta found: " + str(theta))
        J_test = logisticRegressionCostFunc(theta, X_test, y_test, lambda_=0)
        print("Cost: " + str(J_test))
        all_theta[i] = theta
    best_J_cv = logisticRegressionCostFunc(all_theta[0], X_validation, y_validation, lambda_=0)
    chosen_theta = np.zeros((X.shape[1], 1))
    print("\n \n")
    for i in range(all_theta.shape[0]):
        temp = logisticRegressionCostFunc(all_theta[i], X_validation, y_validation, lambda_=0)
        print("Theta use: " + str(all_theta[i]))
        print("Cost calculated: " + str(temp))
        if temp <= best_J_cv:
            best_J_cv = temp
            chosen_theta = all_theta[i]
    print("Best validation cost: " + str(best_J_cv))
    print("Best theta: " + str(chosen_theta))
    print("\n \n")

    # evaluate model
    right = 0
    wrong = 0
    test_cost = logisticRegressionCostFunc(chosen_theta, X_test, y_test, lambda_=0)
    print("Test cost: " + str(test_cost))
    for i in range(y_test.size):
        result = predict(chosen_theta, X_test[i])
        print("Result by model: " + str(result) + "\n" + "Actual result: " + str(y_test[i]))
        if result[0] == y_test[i]:
            right += 1
        else:
            wrong += 1
    print("\n \n")
    print("right: " + str(right))
    print("Total: "+ str(y_test.size))
    print("accuracy: " + str(right/y_test.size))

    # solution by sklearn
    model = sklearn.linear_model.LogisticRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    print("coef: " + str(model.coef_))
    print("intercept: " + str(model.intercept_))
    print("Score: " + str(model.score(X_test, y_test)))
    for i in range(y_test.size):
        result = model.predict(np.array([X_test[i]]))
        print("Result by sklearn: " + str(result) + "\n" + "Actual result: " + str(y_test[i]))
