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

    physics = args.data._pendulum_gen
    n = args.data._n
    redim = args.data._redim
    h = args.timestep

    logger.info(f"Loaded physics simulator for {n}-link pendulum")

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y|%H:%M:%S")

    cache_path = Path("pendulum-cache") / f"p-physics-{n}.npy"
    predictions_path = Path("pendulum-cache") / f"prediction-{n}-{dt_string}.npy"

    # Energy functions
    energy = pendulum_energy.pendulum_energy(n)

    if not cache_path.exists():
        logger.info(f"Generating trajectories for {cache_path}")
        # Initialize args.number initial positions:
        X_init = np.zeros((args.number, 2 * n)).astype(np.float32)
        X_init[:,:] = (np.random.rand(args.number, 2*n).astype(np.float32) - 0.5) * np.pi # Pick values in range [-pi/8, pi/8] radians, radians/sec

        X_phy = np.zeros((args.steps, *X_init.shape), dtype=np.float32)
        X_phy[0,...] = X_init
        for i in range(1, args.steps):
            logger.info(f"Timestep {i}")
            k1 = h * physics(X_phy[i-1,...])
            k2 = h * physics(X_phy[i-1,...] + k1/2)
            k3 = h * physics(X_phy[i-1,...] + k2/2)
            k4 = h * physics(X_phy[i-1,...] + k3)
            X_phy[i,...] = X_phy[i-1,...] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            assert not np.any(np.isnan(X_phy[i,...]))

        np.save(cache_path, X_phy)
        logger.info(f"Done generating trajectories for {cache_path}")

    else:
        X_phy = np.load(cache_path).astype(np.float32)
        logger.info(f"Loaded trajectories from {cache_path}")

    X_nn = to_variable(torch.tensor(X_phy[0,:,:]), cuda=torch.cuda.is_available())
    errors = np.zeros((args.steps,))


    # Variable holding all our predictions to be saved
    X_pred = np.zeros((args.steps, *X_nn.shape)) 
    for i in range(1, args.steps):
        X_nn.requires_grad = True
        k1 = h * model(X_nn)
        k1 = k1.detach()
        k2 = h * model(X_nn + k1/2)
        k2 = k2.detach()
        k3 = h * model(X_nn + k2/2)
        k3 = k3.detach()
        k4 = h * model(X_nn + k3)
        k4 = k4.detach()
        X_nn = X_nn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        X_nn = X_nn.detach()

        y = X_nn.cpu().numpy()

        # Variable to save all our predictions
        X_pred[i,...] = y

        # TODO: Update error calculation
        vel_error = np.sum((X_phy[i,:,n:] - y[:,n:])**2)
        ang_error = (X_phy[i,:,:n] - y[:,:n]).astype('float64')
        ang_error = scaleAngErr(ang_error)

        ang_error = np.sum(ang_error**2)

        errors[i] = (vel_error + ang_error)

    for i in range(args.steps):
        if i % 100 == 0:
            print(f"Step {i}  |  Cum. error: {np.sum(errors[0:i])}  |  Error: {errors[i]}")

    np.save(predictions_path, X_pred)


def scaleAngErr(ang_error):
    # Scales errors to the range [0, 2pi)
    mod = ang_error//(2*np.pi)
    ang_error -= mod*2*np.pi
    # assert all([np.all(ang_error <= 2*np.pi), np.all(ang_error >= 0)]), f"Angles not in range [0, 2pi]: {ang_error[ang_error > 2*np.pi], ang_error[ang_error < 0]}"

    # Then moves them to the range (-pi, pi]
    if np.any(ang_error > np.pi):
        ang_error[ang_error > np.pi] -= 2*np.pi
    assert all([np.all(ang_error <= np.pi), np.all(ang_error >= -np.pi)]), f"Angles not in range [-pi, pi]: {ang_error[ang_error > np.pi], ang_error[ang_error < -np.pi]}"

    return ang_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--number', type=int, default=1000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.1, help="duration of each timestep")

    parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, help="number of steps to evaluate over")

    main(parser.parse_args())
