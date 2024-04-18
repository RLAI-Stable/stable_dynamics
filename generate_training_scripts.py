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
ALPHA_VALUES = [0.001]  # 0.0001
INNER_EPSILON_VALUES = [0.01]  # 0.005 0.001
SMOOTH_V_VALUES = [0]
REHU_VALUES = [0.005]  # 0.001 #0.0005
LAYERSIZE_VALUES = {128: [300], 8: [100]}  # 100 200 300 #900 1000 #300 500 700 900 1100
LR_VALUES = [0.0003, 0.0005]  # Define two LR values here
INPUT_DIM_VALUES = [8, 128]  # Define the input dimension values
N_RUNS = 10  # Number of runs per combination

# Relative path to train_water_stable script
STABLE_SCRIPT_PATH = "./train_water_stable"

datasets_8_sensors = [
    "./water_data/train_april2022_8_sensors.pkl",
    "./water_data/train_may2022_8_sensors.pkl",
    "./water_data/train_july2022_8_sensors.pkl",
    "./water_data/test_12m_april2023_8_sensors.pkl",
    "./water_data/test_12m_may2023_8_sensors.pkl",
    "./water_data/test_12m_july2023_8_sensors.pkl",
]

datasets_128_sensors = [
    "./water_data/train_april2022_128_sensors.pkl",
    "./water_data/train_july2022_128_sensors.pkl",
    "./water_data/train_may2022_128_sensors.pkl",
    "./water_data/test_12m_april2023_128_sensors.pkl",
    "./water_data/test_12m_may2023_128_sensors.pkl",
    "./water_data/test_12m_july2023_128_sensors.pkl",
]

# Define dataset filenames for each input dimension
INPUT_DIM_DATASETS = {
    8: datasets_8_sensors,
    128: datasets_128_sensors,
}

# Iterate over parameter values
for run in range(N_RUNS):
    for input_dim in INPUT_DIM_VALUES:
        for ALPHA in ALPHA_VALUES:
            for INNER_EPSILON in INNER_EPSILON_VALUES:
                for SMOOTH_V in SMOOTH_V_VALUES:
                    for REHU in REHU_VALUES:
                        for LR in LR_VALUES:
                            for LAYERSIZE in LAYERSIZE_VALUES[input_dim]:
                                available_datasets = INPUT_DIM_DATASETS[input_dim]
                                for dataset_idx in range(len(available_datasets) - 1):
                                    train_dataset = available_datasets[dataset_idx]
                                    for test_dataset_idx in range(
                                        dataset_idx + 1, len(available_datasets)
                                    ):
                                        test_dataset = available_datasets[
                                            test_dataset_idx
                                        ]
                                        # Generate script file with parameters
                                        SCRIPT_FILE = os.path.join(
                                            SCRIPT_FOLDER,
                                            "train_script_{}.sh".format(FILE_INDEX),
                                        )
                                        print(
                                            "Generating script file: {}".format(
                                                SCRIPT_FILE
                                            )
                                        )
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
