#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:15:26 2020

@author: Ahmet Dundar
"""

# TODO add cost function curve by iteration
# TODO add F1 score
# TODO add train/cv curve or learning curve

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from loading import load_variables
from sklearn.metrics import f1_score


def initialize_parameters(layer_dims: list) -> dict:
    """
    initialize weights and bias
    :param layer_dims: list
    :return: parameters : dict
    """
    if not isinstance(layer_dims, list):
        raise TypeError("layer_dims must be list")
    parameters = {}
    for number in range(1, len(layer_dims)):
        parameters["b" + str(number)] = np.zeros((layer_dims[number][0], 1))
        parameters["activation" + str(number)] = layer_dims[number][-1].lower()

        if parameters["activation" + str(number)] == "tanh":
            coefficient = np.sqrt(2 / (layer_dims[number - 1][0] + layer_dims[number][0]))
        elif parameters["activation" + str(number)] == "sigmoid":
            coefficient = 10e-3
        else:
            coefficient = np.sqrt(2 / (layer_dims[number - 1][0]))

        parameters["W" + str(number)] = np.random.randn(layer_dims[number][0], layer_dims[number - 1][0]) * coefficient
    return parameters


def __relu(z: np.ndarray) -> np.ndarray:
    """
    calculate ReLU from z
    :param z: np.ndarray
    :return: np.ndarray, return's shape is the same shape of z
    """
    return np.maximum(0.01 * z, z)


def __sigmoid(z: np.ndarray) -> np.ndarray:
    """
    calculate sigmoid from z
    :param z:
    :return: np.ndarray
    """
    return expit(z)


def forward_function(Z: np.ndarray, activation: str) -> np.ndarray:
    """
    calculate activation function result
    :param Z: np.ndarray
    :param activation: str
    :return: np.ndarray
    """
    if activation == "sigmoid":
        return __sigmoid(Z)
    elif activation == "relu":
        return __relu(Z)
    elif activation == "tanh":
        return np.tanh(Z)
    else:
        raise ValueError("Activation function must be one of them which are sigmoid, relu or tanh\n")


def forward_prop(parameters: dict, input_layer: np.ndarray, grads: dict, return_output: bool = False):
    """
    apply forward propagation
    :param grads:
    :param parameters: dict
    :param input_layer: np.ndarray
    :param return_output: bool
    :return: dict
    """

    grads.clear()
    grads["A0"] = input_layer
    layer_size = len(parameters.keys()) // 3

    for index in range(1, layer_size + 1):
        z = np.dot(parameters["W" + str(index)], grads["A" + str(index - 1)]) + parameters["b" + str(index)]
        a = forward_function(z, parameters["activation" + str(index)])
        grads["Z" + str(index)] = z
        grads["A" + str(index)] = a

    if return_output:
        return grads["A" + str(layer_size)]


def __derivative_of_relu(z: np.ndarray) -> np.ndarray:
    """
    calculate derivative of relu from z
    :param z: np.ndarray
    :return: np.ndarray
    """
    gradient = (z >= 0).astype(np.float128)
    gradient[gradient == 0] = 0.01
    return gradient


def __derivative_of_sigmoid(a: np.ndarray):
    """
    calculate derivative of sigmoid from a
    :param a: np.ndarray
    :return: np.ndarray
    """
    return np.multiply(a, (1 - a))


def __derivative_of_tanh(a: np.ndarray):
    """
    calculate derivative of tanh function at a
    :param a: np.ndarray
    :return: np.ndarray
    """
    return 1 - np.power(a, 2)


def backward_function(a: np.ndarray, activation: str):
    """
    :param a: np.ndarray 
    :param activation: str
    :return: np.ndarray
    """

    if activation == "sigmoid":
        return __derivative_of_sigmoid(a)
    elif activation == "tanh":
        return __derivative_of_tanh(a)
    elif activation == "relu":
        return __derivative_of_relu(a)
    else:
        raise ValueError("Activation function must be one of them which are sigmoid, relu or tanh\n")


def backward_prop(parameters: dict, output_layer: np.ndarray, grads: dict):
    """
    calculate backward pro
    :param parameters: 
    :param grads:
    :param output_layer:
    :return: 
    """

    length = len(parameters.keys()) // 3
    m = output_layer.shape[-1]

    dZ = grads["A" + str(length)] - output_layer
    dW = np.dot(dZ, grads["A" + str(length - 1)].transpose()) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    grads.update({
        "dZ" + str(length): dZ,
        "dW" + str(length): dW,
        "db" + str(length): db
    })

    for index in range(length - 1, 0, -1):
        dA_prew = np.dot(parameters["W" + str(index + 1)].transpose(), grads["dZ" + str(index + 1)])

        if parameters["activation" + str(index)] == "relu":
            derivative = __derivative_of_relu(grads["Z" + str(index)])
        else:
            derivative = backward_function(grads["A" + str(index)], parameters["activation" + str(index)])

        dZ = np.multiply(dA_prew, derivative)
        dW = np.dot(dZ, grads["A" + str(index - 1)].transpose()) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        grads.update({
            "dZ" + str(index): dZ,
            "dW" + str(index): dW,
            "db" + str(index): db
        })


def predict(_pre_predict: np.ndarray, limit: float = 0.5) -> np.ndarray:
    """

    :param _pre_predict:
    :param limit:
    :return:
    """

    return (_pre_predict > limit).astype(int)


def cost_function(output: np.ndarray, output_train: np.ndarray) -> float:
    """
    calculate cost function of a neural networks
    :param output_train: np.ndarray
    :param output: np.ndarray
    :return: float
    """

    inside = np.multiply(output_train, np.log(output + 10e-8)) + np.multiply((1 - output_train),
                                                                             np.log(1 - output + 10e-8))
    cost = np.sum(inside) / -output_train.shape[-1]
    cost = np.squeeze(cost)

    if cost.shape != ():
        raise Exception("Cost is: ", cost, "\ncost.shape: ", cost.shape)
    return float(cost)


def initialize_adam(parameters: dict) -> tuple:
    """
    
    :param parameters: 
    :return: 
    """
    length = len(parameters.keys()) // 3
    exp_weighted_ave = {}
    rms_prop = {}

    for index in range(1, length + 1):
        exp_weighted_ave["dW" + str(index)] = np.zeros(parameters["W" + str(index)].shape)
        exp_weighted_ave["db" + str(index)] = np.zeros(parameters["b" + str(index)].shape)
        rms_prop["dW" + str(index)] = np.zeros(parameters["W" + str(index)].shape)
        rms_prop["db" + str(index)] = np.zeros(parameters["b" + str(index)].shape)

    return exp_weighted_ave, rms_prop


def mini_batch_indexes(mini_batch_size: int, examples_size: int, seed: int):
    """

    :param seed:
    :param mini_batch_size:
    :param examples_size:
    :return:
    """
    np.random.seed(seed)
    permutation = np.random.permutation(examples_size)

    length = examples_size // mini_batch_size

    for index in range(length):
        yield permutation[(index * mini_batch_size) : ((index + 1) * mini_batch_size)]

    if examples_size % mini_batch_size != 0:
        yield permutation[(length * mini_batch_size):]


def gradient_descent_loop(grads: dict, parameters: dict, output: np.ndarray, input_array: np.ndarray,
                          dev_input: np.ndarray, dev_output: np.ndarray, iteration: int,
                          learning_rate: float = 0.1, mini_batch_size: int = 32, beta_1: float = 0.9,
                          beta_2: float = 0.999, epsilon: float = 10e-8, gradient_checking: bool = False) -> dict:
    """
    :param epsilon:
    :param beta_2:
    :param beta_1:
    :param mini_batch_size:
    :param grads:
    :param parameters:
    :param output:
    :param input_array:
    :param iteration:
    :param dev_input:
    :param dev_output:
    :param learning_rate:
    :param gradient_checking:
    :return:
    """
    holder = {
        "cost": [],
        "train_error": [],
        "dev_error": [],
        "f1_score": []
    }

    length = len(parameters.keys()) // 3

    exp_weight_ave, rms_prop = initialize_adam(parameters)
    power = 0

    if gradient_checking:
        forward_prop(parameters, input_array, grads)
        backward_prop(parameters, output, grads)
        _result = gradient_check(parameters, grads, input_array, output)
        return {"result": _result}

    for epoch in range(iteration):
        for dev_indexes in mini_batch_indexes(mini_batch_size, dev_input.shape[-1], power):
            dev_grap = {}
            forward_prop(parameters, dev_input[:, dev_indexes], dev_grap)
            holder["dev_error"].append(calculate_error(dev_output[:, dev_indexes], dev_grap["A" + str(length)]))

        for train_indexes in mini_batch_indexes(mini_batch_size, input_array.shape[-1], power):
            forward_prop(parameters, input_array[:, train_indexes], grads)
            holder["cost"].append(cost_function(grads["A" + str(length)], output[:, train_indexes]))
            holder["train_error"].append(calculate_error(output[:, train_indexes], grads["A" + str(length)]))
            holder["f1_score"].append(f1_score(output[:, train_indexes], predict(grads["A" + str(length)]), average="micro"))

            backward_prop(parameters, output[:, train_indexes], grads)
            # decaying_learning_rate = learning_rate * (1 / (1 + 0.01 * power))
            # and learning_rate = 0.001, f1 score is 0.97
            decaying_learning_rate = learning_rate * (1 / (1 + 0.011 * power))
            power += 1
            if power % 50 == 1:
                print("Cost at ", power, ":\t", holder["cost"][-1], end="\t")
                print("learning_rate: ", decaying_learning_rate)

            for index in range(1, length + 1):
                # update exponentially weighted averages
                exp_weight_ave["dW" + str(index)] = np.multiply(beta_1, exp_weight_ave["dW" + str(index)]) + np.multiply(
                    1 - beta_1, grads["dW" + str(index)])
                exp_weight_ave["db" + str(index)] = np.multiply(beta_1, exp_weight_ave["db" + str(index)]) + np.multiply(
                    1 - beta_1, grads["db" + str(index)])

                # update rms prop
                rms_prop["dW" + str(index)] = np.multiply(beta_2, rms_prop["dW" + str(index)]) + np.multiply(
                    1 - beta_2, np.power(grads["dW" + str(index)], 2))
                rms_prop["db" + str(index)] = np.multiply(beta_2, rms_prop["db" + str(index)]) + np.multiply(
                    1 - beta_2, np.power(grads["db" + str(index)], 2))

                # calculate exponentially weighted averages with bias correction
                exp_weight_ave_correct_dW = np.divide(exp_weight_ave["dW" + str(index)], 1 - beta_1 ** power)
                exp_weight_ave_correct_db = np.divide(exp_weight_ave["db" + str(index)], 1 - beta_1 ** power)

                # calculate rms prop with bias correction
                rms_prop_correct_dW = np.divide(rms_prop["dW" + str(index)], 1 - beta_2 ** power)
                rms_prop_correct_db = np.divide(rms_prop["db" + str(index)], 1 - beta_2 ** power)

                # update parameters
                parameters["W" + str(index)] -= np.multiply(decaying_learning_rate,
                                                            np.divide(
                                                                exp_weight_ave_correct_dW,
                                                                np.sqrt(rms_prop_correct_dW + epsilon)
                                                            ))
                parameters["b" + str(index)] -= np.multiply(decaying_learning_rate,
                                                            np.divide(
                                                                exp_weight_ave_correct_db,
                                                                np.sqrt(rms_prop_correct_db + epsilon)
                                                            ))

    return holder


def calculate_error(output: np.ndarray, output_predict: np.ndarray) -> list:
    """

    :param output:
    :param output_predict:
    :return:
    """

    m = output.shape[-1]

    return np.sum(np.power((output_predict - output), 2)) / (2 * m)


def gradient_check(parameters: dict, grads: dict, input_array: np.ndarray, output: np.ndarray, epsilon: float = 10e-7):
    length = len(parameters.keys()) // 3
    approx = {}

    for index in range(1, length + 1):
        approx["W" + str(index)] = np.zeros(parameters["W" + str(index)].shape)
        approx["b" + str(index)] = np.zeros(parameters["b" + str(index)].shape)

        for row in range(parameters["W" + str(index)].shape[0]):
            for column in range(parameters["W" + str(index)].shape[-1]):
                holder = parameters["W" + str(index)][row, column]
                parameters["W" + str(index)][row, column] = holder + epsilon
                cost_1 = cost_function(forward_prop(parameters, input_array, {}, True), output)

                parameters["W" + str(index)][row, column] = holder - epsilon
                cost_2 = cost_function(forward_prop(parameters, input_array, {}, True), output)

                parameters["W" + str(index)][row, column] = holder

                approx["W" + str(index)][row, column] = (cost_1 - cost_2) / (2 * epsilon)

        for row in range(parameters["b" + str(index)].shape[0]):
            for column in range(parameters["b" + str(index)].shape[-1]):
                holder = parameters["b" + str(index)][row, column]
                parameters["b" + str(index)][row, column] = holder + epsilon
                cost_1 = cost_function(forward_prop(parameters, input_array, {}, True), output)

                parameters["b" + str(index)][row, column] = holder - epsilon
                cost_2 = cost_function(forward_prop(parameters, input_array, {}, True), output)

                parameters["b" + str(index)][row, column] = holder

                approx["b" + str(index)][row, column] = (cost_1 - cost_2) / (2 * epsilon)

    lst_1 = []
    lst_2 = []

    for index in range(1, length + 1):
        lst_1.append(grads["dW" + str(index)].reshape(-1, 1))
        lst_1.append(grads["db" + str(index)].reshape(-1, 1))

        lst_2.append(approx["W" + str(index)].reshape(-1, 1))
        lst_2.append(approx["b" + str(index)].reshape(-1, 1))
    dQ = np.concatenate(lst_1, axis=0)
    approx_Q = np.concatenate(lst_2, axis=0)

    if dQ.shape != approx_Q.shape:
        raise IndexError("dQ.shape: ", dQ.shape, "\napprox_Q.shape: ", approx_Q.shape)

    nominator = np.linalg.norm(approx_Q - dQ)
    denominator = np.linalg.norm(approx_Q) + np.linalg.norm(dQ)
    difference = nominator / denominator

    if difference <= epsilon:
        # print("Congratulations!\nDifference: ", difference, flush=True)
        return 1
    elif difference <= epsilon * 10e2:
        # print("Meh!\nepsilon: ", epsilon, "\ndifference: ", difference, flush=True)
        return 0
    else:
        # print("There are some problems in backprop " + str(difference), flush=True)
        #
        # for index in range(1, length + 1):
        #     print("%" * 50)
        #     for appro, real in zip(approx["W" + str(index)], grads["dW" + str(index)]):
        #         print("-" * 55)
        #         print("appr_W" + str(index), ":", appro)
        #         print("real_W" + str(index), ":", real)
        #     print("%" * 50)
        #     for appro, real in zip(approx["b" + str(index)], grads["db" + str(index)]):
        #         print("-" * 55)
        #         print("appr_b" + str(index), ":", appro)
        #         print("real_b" + str(index), ":", real)
        #
        # input()
        return -1


def main(layer: list):
    """
    layer = [(number_of_units: int, actiavtion_function_name:str))
    :param layer:
    :return:
    """

    if not isinstance(layer, list):
        raise TypeError(
            "layer parameter must be list contains tuple like\n[(number_of_units: int, actiavtion_function_name:str)]")

    train_cv_test = load_variables()

    layer.insert(0, (train_cv_test["input_train"].shape[0],))
    layer.append((train_cv_test["output_train"].shape[0], "sigmoid"))

    parameters = initialize_parameters(layer)
    grads = {}

    print("Learning starting")
    __result = gradient_descent_loop(grads=grads,
                                     parameters=parameters,
                                     output=train_cv_test["output_train"].values,
                                     input_array=train_cv_test["input_train"].values,
                                     dev_input=train_cv_test["input_cv"].values,
                                     dev_output=train_cv_test["output_cv"].values,
                                     iteration=100,
                                     learning_rate=0.001,
                                     mini_batch_size=64,
                                     gradient_checking=False)

    try:
        for key in parameters.keys():
            parameters[key].to_csv(key + ".csv")
    except Exception as e:
        print("Parameters are not saved.")
        print(e)
    
    try:
        fig, ax = plt.subplots(5)
        ax[0].plot(__result["cost"], color="green", label="cost")
        ax[1].plot(__result["dev_error"], color="blue", label="dev_error")
        ax[2].plot(__result["train_error"], color="red", label="train_error")
        ax[3].plot(__result["train_error"], color="red", label="train_error")
        ax[3].plot(__result["dev_error"], color="blue", label="dev_error")
        ax[4].plot(__result["f1_score"], color="brown", label="f1_score")
        ax[0].set(xlabel="iteration", ylabel="cost")
        ax[1].set(xlabel="iteration", ylabel="cost")
        ax[2].set(xlabel="iteration", ylabel="cost")
        ax[3].set(xlabel="iteration", ylabel="cost")
        ax[4].set(xlabel="iteration", ylabel="f1_score")
        ax[0].set_title("Cost")
        ax[1].set_title("Dev_error")
        ax[2].set_title("train_error")
        ax[3].set_title("dev/train_error")
        ax[4].set_title("f1_score/iteration")
    except Exception as e:
        print(e)

    prediction = predict(forward_prop(parameters, train_cv_test["input_train"].values, {}, True))

    try:
        from sklearn.metrics import accuracy_score
        print("\n", "%" * 50)
        print("%" + "-" * 48 + "%")
        print("%f1 score for train is: ", f1_score(train_cv_test["output_train"].values, prediction, average="micro"))
        print("%accuracy_score for train is: ", accuracy_score(train_cv_test["output_train"].values, prediction))
        print("%" + "-" * 48 + "%")
        dev_prediction = predict(forward_prop(parameters, train_cv_test["input_cv"].values, {}, True))
        print("%f1 score for cv is:    ", f1_score(train_cv_test["output_cv"], dev_prediction, average="micro"))
        print("%accuracy_score for train is: ", accuracy_score(train_cv_test["output_cv"].values, dev_prediction))
        print("%" + "-" * 48 + "%")
        test_prediction = predict(forward_prop(parameters, train_cv_test["input_test"].values, {}, True))
        print("%f1 score for cv is:    ", f1_score(train_cv_test["output_test"], test_prediction, average="micro"))
        print("%accuracy_score for train is: ", accuracy_score(train_cv_test["output_test"].values, test_prediction))
        print("%" * 50)
    except Exception as e:
        print(e)

    return __result, grads, train_cv_test, parameters


#if __name__ == "__main__":
#    import time

#    tic = time.time()
#    layers = [(25, "tanh"), (10, "tanh")]
#    result = main(layers)
#    toc = time.time()

#    print("Total time in minute: ", (toc - tic) / 60)
