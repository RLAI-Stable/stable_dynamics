#!/usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
import torch
from util import (DynamicLoad, latest_file, loadDataFile, setup_logging, to_variable)

from pathlib import Path

PIT300_COLUMN = "ZW_PIT300.DATA"
SKIP=20

logger = setup_logging(os.path.basename(__file__))

def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    h = args.skip

    data = load_water_data()
    sensor_data = list(data[PIT300_COLUMN])
    current_step = get_starting_point(data)

    X_nn = current_step
    total_error = 0
    for i in range(1, args.steps):
        current_step = predict_next_step(model, h, X_nn)
        total_error += calculate_prediction_error(current_step.cpu().numpy(), sensor_data, i, skip=SKIP)

    save_error(args.save_to, total_error)


def predict_next_step(model, h, X_nn):
    # TODO: RK4 could be interesting here
    X_nn.requires_grad = True
    k1 = model(X_nn) # H can be 1
    k1 = k1.detach()
    return k1

def calculate_prediction_error(model_prediction, sensor_data, step, skip=20):
    return abs(sensor_data[step * skip] - model_prediction[0][53].item())

def load_water_data():
    cache_path = Path("sensor-cache") / f"test_nov22_no_trace.pkl"
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_starting_point(data):
    start = to_variable(torch.tensor(np.array(data.iloc[0:1])), cuda=torch.cuda.is_available())
    return start

def save_error(path, total_error):
    rollout_error = {"rollout_error": total_error}
    with open(path, 'wb') as f:
        pickle.dump(rollout_error, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--skip', type=float, default=20, help="Downscaling of the rollout of data")
    parser.add_argument('--save-to', type=str, help="Destination of the file containing evaluation loss")

    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, help="number of steps to evaluate over")

    main(parser.parse_args())
