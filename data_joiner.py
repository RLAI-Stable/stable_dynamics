import os
import pickle
import csv
from datetime import datetime

def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_to_csv(data, csv_file):
    outdir = "aggregated_experiments/experiment_{:%d_%m_%Y-%H:%M:%S}".format(datetime.now())
    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, csv_file)
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def process_experiment(experiment_path, model_parameters):
    model_eval_data = []
    model_epochs_data = []
    
    for root, dirs, files in os.walk(experiment_path):
        for uuid_dir in dirs:
            uuid_path = os.path.join(root, uuid_dir)
            epoch_losses_path = os.path.join(uuid_path, 'epoch_losses.pkl')
            rollout_eval_path = os.path.join(uuid_path, 'rollout_eval.pkl')
            
            if os.path.exists(epoch_losses_path) and os.path.exists(rollout_eval_path):
                # Read epoch_losses.pkl
                epoch_losses = read_pickle_file(epoch_losses_path)
                for item in epoch_losses:
                    item['model_parameters'] = model_parameters
                    item['uuid'] = uuid_dir
                    model_epochs_data.append(item)
                
                # Read rollout_eval.pkl
                rollout_eval = read_pickle_file(rollout_eval_path)
                rollout_eval['model_parameters'] = model_parameters
                rollout_eval['uuid'] = uuid_dir
                model_eval_data.append(rollout_eval)
    
    # Save data to CSV
    save_to_csv(model_eval_data, 'model_eval.csv')
    save_to_csv(model_epochs_data, 'model_epochs.csv')

# Main function to process all experiments
def process_all_experiments(experiments_root):
    for model_parameters in os.listdir(experiments_root):
        joined_path = os.path.join(experiments_root, model_parameters)
        if os.path.isdir(joined_path):
            process_experiment(joined_path, model_parameters)

experiments_root = 'experiments/water-stable-skip'
process_all_experiments(experiments_root)