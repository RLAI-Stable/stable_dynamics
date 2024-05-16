#!/usr/bin/env python3

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from util import DynamicLoad, latest_file, setup_logging, to_variable

from pathlib import Path
import matplotlib.pyplot as plt

PIT300_COLUMN = "ZW_PIT300.DATA"

logger = setup_logging(os.path.basename(__file__))


def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    all_sensors_data = load_water_data(args.rollout_dataset)
    PIT300_index = all_sensors_data.columns.get_loc(PIT300_COLUMN)
    PIT300_data = all_sensors_data[PIT300_COLUMN]

    current_step = get_starting_point(all_sensors_data)
    starting_PIT300 = current_step[0][PIT300_index].item()

    total_error = 0
    predicted_rollout = [starting_PIT300]
    true_rollout = [starting_PIT300]
    cumulative_error = [0]
    for i in range(1, args.rollout_steps):
        current_step = predict_next_step(model, current_step)

        current_PIT300 = current_step[0][PIT300_index].cpu().numpy()
        if i * args.skip > len(all_sensors_data):
            break
        true_PIT300 = PIT300_data[i * args.skip]

        predicted_rollout.append(current_PIT300)
        true_rollout.append(true_PIT300)

        current_error = calculate_prediction_error(current_PIT300, true_PIT300)
        cumulative_error.append(current_error + cumulative_error[-1])

    save_error(args.save_to, cumulative_error)
    save_rollout_plots(
        args.save_image_to, predicted_rollout, true_rollout, total_error, args.skip
    )
    save_rollout_error_plot(args.save_error_plot_to, cumulative_error)


def predict_next_step(model, X_nn):
    # TODO: RK4 could be interesting here
    X_nn.requires_grad = True
    X_nn = X_nn.to(torch.float64)
    k1 = model(X_nn)  # H can be 1
    k1 = k1.detach()
    return k1


def calculate_prediction_error(model_prediction, true_data):
    return abs(true_data - model_prediction)


def load_water_data(rollout_dataset):
    # TODO: Maybe think about loading multiple datasets?
    data = pd.read_pickle(rollout_dataset)
    return data


def get_starting_point(data):
    """
    Returns the first step of the trajectory
    """
    start = to_variable(
        torch.tensor(np.array(get_first_row(data))), cuda=torch.cuda.is_available()
    )
    return start


def get_first_row(data):
    # If the data is a Pandas dataframe
    return data.iloc[:1]


def save_error(path, cumulative_error):
    """
    Saves the numerical total_error and avg_error to a file
    """
    total_error = cumulative_error[-1]
    avg_error = total_error / len(cumulative_error)
    rollout_error = {"rollout_error": total_error, "avg_error": avg_error}
    print(rollout_error)
    with open(path, "wb") as f:
        pickle.dump(rollout_error, f)


def save_rollout_plots(path, predicted_rollout, true_rollout, total_error, skip=20):
    """
    Plots the real rollout against the predicted rollout
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(predicted_rollout, label="Model predictions")
    plt.title(
        f"Model Predictions (Skip - {skip})"
    )  # TODO: make this adaptable to a SKIP parameter, for now we always assume that we are running SKIP 20 to simplify the code

    plt.subplot(1, 2, 2)
    plt.plot(true_rollout, label="PIT300")
    plt.title("PIT300 - Test Set")

    plt.tight_layout()

    # Show the plots
    plt.savefig(path)
    plt.close()


def save_rollout_error_plot(path, cumulative_error):
    """
    Plots the (cumulative) rollout error per timestep
    """
    plt.figure(figsize=(12, 4))
    plt.title("Error over the rollout")
    plt.plot(cumulative_error, label="Cumulative error")
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error of .")
    parser.add_argument(
        "--rollout-dataset",
        type=str,
        help="Dataset from which to load the rollout steps.",
    )
    parser.add_argument(
        "--skip", type=int, default=20, help="Downscaling of the rollout of data"
    )
    parser.add_argument(
        "--save-to", type=str, help="Destination of the file containing evaluation loss"
    )
    parser.add_argument(
        "--save-error-plot-to",
        type=str,
        help="Destination of the file containing the cumulative rollout error plot",
    )
    parser.add_argument(
        "--save-image-to",
        type=str,
        help="Destination of the file containing the predicted rollout plot",
    )
    parser.add_argument(
        "--rollout-steps", type=int, help="number of steps to evaluate over"
    )
    parser.add_argument("--model", type=DynamicLoad("models"), help="model to load")
    parser.add_argument("--weights", type=latest_file, help="model weights to load")

    main(parser.parse_args())
