import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
import os

HEADERS = ["alpha", "inner_epsilon", "rehu", "mean_error", "n_links", "run"]
RESULTS_DIRECTORY = "experiments/results/error_stable/"
CSV_RESULTS_FILE = RESULTS_DIRECTORY + "run_results.csv"
IMAGES_DIRECTORY = "experiments/results/error_stable/images/"


def parse_hyperparameters(model_folder):
    model_name = model_folder.split("/")[3]
    hyperparameters = model_name.split("_")
    alpha, inner_epsilon, rehu = float(hyperparameters[0]), float(hyperparameters[2]), float(hyperparameters[4])
    return alpha, inner_epsilon, rehu

def load_error_file(model_folder):
    error_files = [f for f in os.listdir(model_folder) if f.endswith(".npy") and f[:-4].isdigit()]
    if error_files:
        highest_index_error = max(error_files, key=lambda f: int(os.path.splitext(f)[0]))
        highest_index_error_path = os.path.join(model_folder, highest_index_error)
        errors = np.load(highest_index_error_path)
    else:
        raise FileNotFoundError("No error files to be loaded in folder {}".format(model_folder))
    return errors


def aggregate_runs_data(data_directory, results_path):
    model_folders = [f.path for f in os.scandir(data_directory) if f.is_dir()] # One model folder for each hyperparameters configuration
    csv_exists = os.path.isfile(CSV_RESULTS_FILE)
    with open(CSV_RESULTS_FILE, "a", newline="") as run_results_file:
        run_results_writer = csv.writer(run_results_file)
        if not csv_exists:
            run_results_writer.writerow(HEADERS)

        for model_folder in model_folders:
            alpha, inner_epsilon, rehu = parse_hyperparameters(model_folder)

            errors = load_error_file(model_folder)
            mean_errors = np.mean(errors, axis=1)

            # Write a row with the mean_error for each run
            for run_idx, mean_error in enumerate(mean_errors):
                run_results_writer.writerow([alpha, inner_epsilon, rehu, mean_error, N_LINKS, run_idx])
            for run_idx, errors in enumerate(errors):
                # Plot the error against timesteps
                plt.plot(range(len(errors)), errors, label=f"Run {run_idx}")
            # Add legend to the plot
            plt.legend()

            # Set labels and title for the plot
            plt.xlabel("Timestep")
            plt.ylabel("Error")
            plt.title(f"Error for {model_folder.split('/')[-1]} - All Runs")

            # Save the plot as an image in the subfolder
            image_path = os.path.join(model_folder, "aggregated_image.png")
            plt.savefig(image_path)

            # Clear the current plot
            plt.clf()


def main(N_LINKS):
    data_directory = f"experiments/error_stable/{N_LINKS}/"
    data_directory_simple = f"experiments/error_simple/{N_LINKS}/"

    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    os.makedirs(IMAGES_DIRECTORY, exist_ok=True)
    aggregate_runs_data(data_directory, CSV_RESULTS_FILE)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py N_LINKS")
        sys.exit(1)
    N_LINKS = int(sys.argv[1])

    main(N_LINKS)