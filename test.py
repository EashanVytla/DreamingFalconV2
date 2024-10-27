import torch
from world_model import WorldModel
import pandas as pd
import os
from tqdm import tqdm
from utils import AttrDict
import yaml
import csv

model_directory = "models/10-14-Synthetic"
data_directory = "data/10-14-Synthetic/train"
log_directory = "runs/10-14"

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def main():
    with open('config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)

    config = AttrDict.from_dict(config_dict)

    model = WorldModel(config)

    model_path = os.path.join("models", "10-14-Synthetic", "model.pt")
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print("Using GPU:", torch.cuda.get_device_name(0)) 
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")

    output_file = os.path.join("models", "10-14-Synthetic", "test.csv")

    states_df = pd.read_csv(os.path.join(data_directory, "states.csv"), header=None)
    actions_df = pd.read_csv(os.path.join(data_directory, "actions.csv"), header=None)
    states_df = states_df.transpose()
    actions_df = actions_df.transpose()

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for (_, state_row), (_, action_row) in zip(states_df.iterrows(), actions_df.iterrows()):
            # Convert row to tensor
            state_tensor = torch.tensor(state_row.values, dtype=torch.float).reshape((1, 12))
            action_tensor = torch.tensor(action_row.values, dtype=torch.float).reshape((1, 4))
            
            _, pred = model.predict(state_tensor, action_tensor)

            csv_writer.writerow(tensor_to_numpy(pred.flatten()))

if __name__ == "__main__":
    main()