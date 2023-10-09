#!/usr/bin/env python3

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

SENSOR_INDEX = 53 # PIT300

global SIZE_A, SIZE_B, model
SIZE_A = 512
SIZE_B = 512

model = None

loss_ = nn.MSELoss()

def loss(Ypred, Yactual, X):
    """
    PREDICTS JUST BY MULTIPLYING THE INPUT BY 10
    This is just the baseline we defined for a specific experiment.
    """
    return loss_(X[:, SENSOR_INDEX] * 10, Yactual[:, SENSOR_INDEX])

def loss_flatten(l):
    return [l]

def loss_labels():
    return ["loss"]

def summary(*a, **kw):
    pass

def configure(props):
    global SIZE_A, SIZE_B, model
    if "a" in props:
        SIZE_A = int(props["a"])

    if "b" in props:
        SIZE_B = int(props["b"])

    model = nn.Sequential(
        nn.Linear(SIZE_A, SIZE_B), nn.ReLU(),
        nn.Linear(SIZE_B, SIZE_B), nn.ReLU(),
        nn.Linear(SIZE_B, SIZE_A))

    logger.info(f"Set layer sizes to {SIZE_A} -> {SIZE_B} -> {SIZE_B} -> {SIZE_A}")

def next_state_prediction(X, Ypred):
    """
    X: original state
    Ypred: network prediction [we currently want it to be the gradient]
    """
    # TODO: check performance difference between using Ypred and X + Ypred
    # return X + Ypred
    return Ypred