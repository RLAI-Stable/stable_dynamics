#!/usr/bin/env python3

import argparse
import os

import numpy as np
from util import (DynamicLoad, setup_logging)

from datasets import pendulum

from pathlib import Path

logger = setup_logging(os.path.basename(__file__))


def main(args):
    #physics = args.data._pendulum_gen
    n = args.n_links
    physics = pendulum.pendulum_gradient(n)
    h = args.timestep
    n_datasets = args.n_datasets

    logger.info(f"Loaded physics simulator for {n}-link pendulum")
    cache_folder = f"pendulum-cache/{n}"
    for i in range(n_datasets):
        cache_path = Path(cache_folder) / f"p-physics-{n}-{i}.npy"
        if not cache_path.exists():
            X_phy = generate_trajectories(cache_path, n, args.number, args.steps, physics, h)


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
    parser = argparse.ArgumentParser(description='Pendulum trajectories')
    parser.add_argument('--number', type=int, default=1000, help="number of starting positions to evaluate from")
    parser.add_argument('--timestep', type=float, default=0.01, help="duration of each timestep")

    #parser.add_argument('data', type=DynamicLoad("datasets"), help='the pendulum dataset to load the simulator from')
    parser.add_argument('n_links', type=int, default=2, help="number of links for the simulated pendulum")
    parser.add_argument('steps', type=int, default=1000, help="number of steps to evaluate over")
    parser.add_argument('n_datasets', type=int, default=30, help="number of datasets to generate")

    main(parser.parse_args())
