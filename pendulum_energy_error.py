#!/usr/bin/env python3

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from plot_data import plot_data, plot_data_args
from torchvision.utils import save_image
from util import (DynamicLoad, latest_file, loadDataFile, setup_logging, to_variable)
from scipy.integrate import odeint

from models import pendulum_energy

from pathlib import Path

from datetime import datetime

logger = setup_logging(os.path.basename(__file__))

def main(args):
    model = args.model.model
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    n = args.data._n

    logger.info(f"Loaded physics simulator for {n}-link pendulum")

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y|%H:%M:%S")

    # Load the test dataset to calculate energies from
    test_set_path = Path("pendulum-cache") / f"p-{n}-train.npz"

    # Energy functions
    true_energy_function = pendulum_energy.pendulum_energy(n)
    model_energy_function = model.V

    pendulum_positions = np.load(test_set_path)
    pendulum_positions = pendulum_positions["X"]   

    errors = []
    for i in range(len(pendulum_positions)):
        pos = torch.FloatTensor([pendulum_positions[i]])
        real_energy = true_energy_function(pos)
        model_energy_pred = model_energy_function(pos).detach().item()
        errors.append((real_energy - model_energy_pred) ** 2)

    mse = (np.sum(errors) / len(errors))
    
    # Register RMSE for this model
    print(mse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--number', type=int, default=1000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.1, help="duration of each timestep")

    parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, help="number of steps to evaluate over")

    main(parser.parse_args())
