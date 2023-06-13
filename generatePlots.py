import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def main():
    if len(sys.argv) < 2:
        sys.argv.append("experiments/evaluation_error_stable/test_run")
        # print("Usage: python3 generatePlots.py <foldername>")
        # return

    filename = genFilename(sys.argv[1])

    # Load the data if it exists
    if not os.path.exists("experiments/tmp/0.npy"):
        raise Exception("No data to plot")
    else:
        data = np.load(f"experiments/tmp/0.npy").reshape((1, -1))

    # Concatenate all the data
    i = 1
    while os.path.exists(f"experiments/tmp/{i}.npy"):
        temp = np.load(f"experiments/tmp/{i}.npy").reshape((1, -1))
        print("temp shape:", temp.shape)
        data = np.vstack((data, temp))
        i += 1
    errors = np.mean(data, axis=0).reshape((-1, 1))
    print("data shape:", data.shape)


    # Load predicitons if they exist
    if os.path.exists(f"experiments/preds/0.npy"):
        X_pred = np.load(f"experiments/preds/0.npy")
        if X_pred.shape[0] != 1:
            X_pred = X_pred.reshape((1, *X_pred.shape))
        i = 1
        while os.path.exists(f"experiments/preds/{i}.npy"):
            temp = np.load(f"experiments/preds/{i}.npy")
            if temp.shape[0] != 1:
                temp = temp.reshape((1, *temp.shape))
            X_pred = np.vstack((X_pred, temp))
            i += 1
        print("X_pred shape:", X_pred.shape)
        np.save(filename+"_pred.npy", X_pred)
    else:
        print("No predictions found")


    plt.plot(range(len(errors)), errors)
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.title(f"Error for {filename.split('/')[-2]} ({data.shape[0]} runs)")
    plt.savefig(filename+".png")
    plt.close()


    np.save(filename+".npy", data)
    # clear the temporary folder
    shutil.rmtree("experiments/tmp")
    shutil.rmtree("experiments/preds")


def genFilename(folder):
    from os.path import exists as path_exists

    if not path_exists(folder):
        os.makedirs(folder)

    i = 1
    filename = f"{folder}/{i}"
    while path_exists(filename+".npy"):
        i += 1
        filename = f"{folder}/{i}"
    
    return filename



if __name__ == "__main__":
    main()










