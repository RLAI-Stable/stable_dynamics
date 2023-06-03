#!/usr/bin/env python3

import argparse
import os

import numpy as np
import torch
from util import (DynamicLoad, latest_file, loadDataFile, setup_logging, to_variable)

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
        X_phy = generate_trajectories(cache_path, n, args.number, args.steps, physics, h)
    else:
        X_phy = np.load(cache_path).astype(np.float32)
        logger.info(f"Loaded trajectories from {cache_path}")

    X_nn = to_variable(torch.tensor(X_phy[0,:,:]), cuda=torch.cuda.is_available())
    errors = np.zeros((args.steps,))
    for i in range(1, args.steps):
        if i % 50 == 0:
            logger.info(f"Calculating error: Timestep {i}")

        X_nn.requires_grad = True
        X_nn = trajectory_step(X_nn , model, h)
        X_nn = X_nn.detach()
        
        y = X_nn.cpu().numpy()

        errors[i] = compute_error_at_timestep(X_phy, y, i, n)

    for i in range(args.steps):
        if i % 50 == 0:
            logger.info(f"Step{i} Cumulative_error: {np.sum(errors[0:i])} Current: {errors[i]}")

    import os
    filename = "experiments/errors/pendulum-error.npz"
    i = 0
    while os.path.exists(filename):
        i += 1
        filename = f"experiments/errors/pendulum-error-{i}.npz"
    np.savez(filename, errors=errors)

    print("{},{},{}".format(i, np.mean(errors[0:]), errors[-1]), end="")

    generatePlot(filename[:-4], errors, n)


def generatePlot(filename, errors, n):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.plot(errors)
    plt.title(f"Error per timestep ({n}-link pendulum)")
    plt.savefig(filename + ".png")


def compute_error_at_timestep(X_phy, y, i, n):
    vel_error = np.sum((X_phy[i,:,n:] - y[:,n:])**2)
    ang_error = (X_phy[i,:,:n] - y[:,:n]).astype('float64')

    # Scales errors to the range [0, 2pi)
    mod = ang_error//(2*np.pi)
    ang_error -= mod*2*np.pi

    # Then moves them to the range (-pi, pi]
    if np.any(ang_error > np.pi):
        ang_error[ang_error > np.pi] -= 2*np.pi
    assert all([np.all(ang_error <= np.pi), np.all(ang_error >= -np.pi)]), f"Angles not in range: {ang_error[ang_error > np.pi], ang_error[ang_error < -np.pi]}"

    ang_error = np.sum(ang_error**2)
    return vel_error + ang_error


def generate_trajectories(cache_path, n, number, steps, physics, h):
        logger.info(f"Generating trajectories for {cache_path}")
        # Initialize args.number initial positions:
        X_init = np.zeros((number, 2 * n)).astype(np.float32)
        X_init[:,:] = (np.random.rand(number, 2*n).astype(np.float32) - 0.5) * np.pi/4 # Pick values in range [-pi/8, pi/8] radians, radians/sec

        X_phy = np.zeros((steps, *X_init.shape), dtype=np.float32)
        X_phy[0,...] = X_init
        for i in range(1, steps):
            logger.info(f"Generating Trajectories: Timestep {i}")
            X_phy[i,...] = trajectory_step(X_phy[i-1,...], physics, h)
            assert not np.any(np.isnan(X_phy[i,...]))

        np.save(cache_path, X_phy)
        logger.info(f"Done generating trajectories for {cache_path}")
        return X_phy


def trajectory_step(X, gradient, h):
    # Applies RK4 to generate X_t+1 from X_t
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    k1 = h * gradient(X)
    k2 = h * gradient(X + k1/2)
    k3 = h * gradient(X + k2/2)
    k4 = h * gradient(X + k3)
    rk4 = X + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return rk4 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Error of .')
    parser.add_argument('--number', type=int, default=1000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.01, help="duration of each timestep")

    parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to load')
    parser.add_argument('weight', type=latest_file, help='model weight to load')
    parser.add_argument('steps', type=int, help="number of steps to evaluate over")

    main(parser.parse_args())
