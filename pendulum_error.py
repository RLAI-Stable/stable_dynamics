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

import time

from models import pendulum_energy

from pathlib import Path

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

    cache_path = Path("pendulum-cache") / f"p-physics-{n}.npy"

    if not cache_path.exists():
        logger.info(f"Generating trajectories for {cache_path}")
        # Initialize args.number initial positions:
        X_init = np.zeros((args.number, 2 * n)).astype(np.float32)
        X_init[:,:] = (np.random.rand(args.number, 2*n).astype(np.float32) - 0.5) * np.pi/4 # Pick values in range [-pi/8, pi/8] radians, radians/sec

        X_phy = np.zeros((args.steps, *X_init.shape), dtype=np.float32)
        X_phy[0,...] = X_init
        for i in range(1, args.steps):
            logger.info(f"Generating Trajectories: Timestep {i}")
            k1 = h * physics(X_phy[i-1,...])
            X_phy[i,...] = X_phy[i-1,...] + k1
            assert not np.any(np.isnan(X_phy[i,...]))

        np.save(cache_path, X_phy)
        logger.info(f"Done generating trajectories for {cache_path}")

    else:
        X_phy = np.load(cache_path).astype(np.float32)
        logger.info(f"Loaded trajectories from {cache_path}")

    X_nn = to_variable(torch.tensor(X_phy[0,:,:]), cuda=torch.cuda.is_available())
    errors = np.zeros((args.steps,))
    for i in range(1, args.steps):
        logger.info(f"Calculating error: Timestep {i}")

        start = time.time()
        X_nn.requires_grad = True
        k1 = h * model(X_nn)
        k1 = k1.detach()
        X_nn = X_nn + k1
        X_nn = X_nn.detach()
        print(f"{i}: Time to predict: {time.time()-start}")
        
        start = time.time()
        y = X_nn.cpu().numpy()
        print(f"{i}: Time to convert to np: {time.time()-start}")

        # TODO: Update error calculation
        start = time.time()
        vel_error = np.sum((X_phy[i,:,n:] - y[:,n:])**2)
        ang_error = (X_phy[i,:,:n] - y[:,:n])
        print(f"{i}: Time to calculate error: {time.time()-start}")

        start = time.time()
        # while np.any(ang_error >= np.pi):
        #     ang_error[ang_error >= np.pi] -= 2*np.pi
        # while np.any(ang_error < -np.pi):
        #     ang_error[ang_error < -np.pi] += 2*np.pi
        mod = ang_error//(2*np.pi)
        ang_error -= mod*2*np.pi
        ang_error -= np.pi
        assert all(np.all(ang_error <= np.pi), np.all(ang_error >= -np.pi))
        print(f"{i}: Time to wrap error: {time.time()-start}")

        start = time.time()
        ang_error = np.sum(ang_error**2)
        errors[i] = (vel_error + ang_error)
        print(f"{i}: Time to sum error: {time.time()-start}")

    for i in range(args.steps):
        print(f"{i}\t{np.sum(errors[0:i])}\t{errors[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--number', type=int, default=1000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.01, help="duration of each timestep")

    parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, help="number of steps to evaluate over")

    main(parser.parse_args())
