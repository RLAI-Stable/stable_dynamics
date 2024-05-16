#!/usr/bin/env python3
import os

# Usage ./train_many_water.py <num_runs>

import sys

if len(sys.argv) != 2:
    print("Usage: {} <skip_steps>".format(sys.argv[0]))
    sys.exit(1)

SKIP = int(sys.argv[1])
EXP_ID = "stable"
INNER = "PSD-REHU"
DT = "1"
SAVE = "1"

FILE_INDEX = 1
SCRIPT_FOLDER = "scripts"  # Define the folder path

# Create the folder if it doesn't exist
os.makedirs(SCRIPT_FOLDER, exist_ok=True)

# Define arrays for parameters with multiple values
ALPHA_VALUES = [0.001, 0.0001]
INNER_EPSILON_VALUES = [0.01, 0.005, 0.001]
SMOOTH_V_VALUES = [0]
REHU_VALUES = [0.005]  # 0.001 #0.0005
LAYERSIZE_VALUES = {2: [32, 64]}
LR_VALUES = [0.01, 0.001]
INPUT_DIM_VALUES = [2]
N_RUNS = 3  # Number of runs per combination

# Relative path to train_water_stable script
STABLE_SCRIPT_PATH = "./train_water_stable"

train_dataset = "./sine_data/train_sine.pkl"
test_dataset = "./sine_data/test_sine.pkl"

# TODO: adapt this script to include many runs in one .sh

# Iterate over parameter values
for run in range(N_RUNS):
    for input_dim in INPUT_DIM_VALUES:
        for ALPHA in ALPHA_VALUES:
            for INNER_EPSILON in INNER_EPSILON_VALUES:
                for SMOOTH_V in SMOOTH_V_VALUES:
                    for REHU in REHU_VALUES:
                        for LR in LR_VALUES:
                            for LAYERSIZE in LAYERSIZE_VALUES[input_dim]:
                                # Generate script file with parameters
                                SCRIPT_FILE = os.path.join(
                                    SCRIPT_FOLDER,
                                    "train_script_{}.sh".format(FILE_INDEX),
                                )
                                print("Generating script file: {}".format(SCRIPT_FILE))
                                with open(SCRIPT_FILE, "w") as f:
                                    f.write("#!/bin/bash\n")
                                    f.write(
                                        '{} "{}" "{}" "{}" "{}" "{}" "{}" "{}" "{}" "{}" "{}" "{}" "{}" "{}" "{}"\n'.format(
                                            STABLE_SCRIPT_PATH,
                                            EXP_ID,
                                            ALPHA,
                                            INNER,
                                            INNER_EPSILON,
                                            SMOOTH_V,
                                            REHU,
                                            DT,
                                            SKIP,
                                            LR,
                                            LAYERSIZE,
                                            input_dim,
                                            train_dataset,
                                            test_dataset,
                                            test_dataset,  # For now we use test as the rollout dataset
                                        )
                                    )
                                os.chmod(SCRIPT_FILE, 0o755)  # chmod +x
                                # Increment the file index
                                FILE_INDEX += 1
