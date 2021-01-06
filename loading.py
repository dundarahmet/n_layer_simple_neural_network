#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 6 10:00:26 20201

@author: Ahmet Dundar
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def load_variables() -> dict:
    """
    Load train, cross validation and test examples
    :return: input_train, input_cv, input_test, output_train, output_cv, output_test
    """

    scaler = StandardScaler()

    input_train = pd.read_csv("x_train.csv", index_col=[0])
    input_train = pd.DataFrame(scaler.fit_transform(input_train), columns=input_train.columns.astype(int),
                               dtype=np.float128)

    input_cv = pd.read_csv("x_cross_validation.csv", index_col=[0])
    input_cv = pd.DataFrame(scaler.fit_transform(input_cv), columns=input_cv.columns.astype(int), dtype=np.float128)

    input_test = pd.read_csv("x_test.csv", index_col=[0])
    input_test = pd.DataFrame(scaler.fit_transform(input_test), columns=input_test.columns.astype(int),
                              dtype=np.float128)

    output_train = pd.read_csv("y_train.csv", index_col=[0])
    output_train = pd.DataFrame(output_train.values, columns=output_train.columns.astype(int), dtype=int)

    output_cv = pd.read_csv("y_cross_validation.csv", index_col=[0])
    output_cv = pd.DataFrame(output_cv.values, columns=output_cv.columns.astype(int), dtype=int)

    output_test = pd.read_csv("y_test.csv", index_col=[0])
    output_test = pd.DataFrame(output_test.values, columns=output_test.columns.astype(int), dtype=int)

    if input_train.shape[-1] != output_train.shape[-1]:
        raise Exception("Wrong Shape: \ninput_train: ({})\noutput_train: ({})".format(
            input_train.shape, output_train.shape))

    if input_cv.shape[-1] != output_cv.shape[-1]:
        raise Exception("Wrong Shape: \ninput_cv: ({})\noutput_cv: ({})".format(
            input_train.cv, output_cv.shape))

    if input_test.shape[-1] != output_test.shape[-1]:
        raise Exception("Wrong Shape: \ninput_test: ({})\noutput_test: ({})".format(
            input_test.shape, output_test.shape))

    train_cv_test = {
        "input_train": input_train,
        "input_cv": input_cv,
        "input_test": input_test,
        "output_train": output_train,
        "output_cv": output_cv,
        "output_test": output_test
    }

    return train_cv_test