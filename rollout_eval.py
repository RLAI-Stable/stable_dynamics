#!/usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
import torch
from util import (DynamicLoad, latest_file, loadDataFile, setup_logging, to_variable)

from pathlib import Path
import matplotlib.pyplot as plt

PIT300_COLUMN = "ZW_PIT300.DATA"
PIT300_INDEX = 53

logger = setup_logging(os.path.basename(__file__))

def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    all_sensors_data = load_water_data()
    PIT300_data = list(all_sensors_data[PIT300_COLUMN])

    current_step = get_starting_point(all_sensors_data)
    starting_PIT300 = current_step[0][PIT300_INDEX].item()

    total_error = 0
    predicted_rollout = [starting_PIT300]
    true_rollout = [starting_PIT300]
    for i in range(1, args.rollout_steps):
        current_step = predict_next_step(model, current_step)

        current_PIT300 = current_step[0][PIT300_INDEX].cpu().numpy()
        true_PIT300 = PIT300_data[i * args.skip]
        
        predicted_rollout.append(current_PIT300)
        true_rollout.append(true_PIT300)

        total_error += calculate_prediction_error(current_PIT300, true_PIT300)

    avg_error = total_error / args.rollout_steps
    save_error(args.save_to, total_error, avg_error)
    save_rollout(args.save_image_to, predicted_rollout, true_rollout, total_error)


def predict_next_step(model, X_nn):
    # TODO: RK4 could be interesting here
    X_nn.requires_grad = True
    k1 = model(X_nn) # H can be 1
    k1 = k1.detach()
    return k1

def calculate_prediction_error(model_prediction, true_data):
    return abs(true_data - model_prediction)

def load_water_data():
    cache_path = Path("sensor-cache") / f"test_nov22_no_trace.pkl"
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_starting_point(data):
    start = to_variable(torch.tensor(np.array(data.iloc[0:1])), cuda=torch.cuda.is_available())
    return start

def save_error(path, total_error, avg_error):
    rollout_error = {"rollout_error": total_error, "avg_error": avg_error}
    print(rollout_error)
    with open(path, 'wb') as f:
        pickle.dump(rollout_error, f)

def save_rollout(path, predicted_rollout, true_rollout, total_error):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(predicted_rollout, label='Model predictions')
    plt.title('Model Predictions (Skip 20, stable)')
              
    plt.subplot(1, 2, 2)
    plt.plot(true_rollout, label='PIT300')
    plt.title('PIT300 - Test Set')

    plt.tight_layout()

    # Show the plots
    plt.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--skip', type=int, default=20, help="Downscaling of the rollout of data")
    parser.add_argument('--save-to', type=str, help="Destination of the file containing evaluation loss")
    parser.add_argument('--save-image-to', type=str, help="Destination of the file containing the predicted rollout plot")
    parser.add_argument('--rollout-steps', type=int, help="number of steps to evaluate over")
    parser.add_argument('--model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('--weights', type=latest_file, help='model weights to load')

    main(parser.parse_args())
