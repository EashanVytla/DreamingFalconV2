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

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print("Using GPU:", torch.cuda.get_device_name(0)) 
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")

    model = WorldModel(config, device).to(device)

    model_path = os.path.join("models", "10-14-Synthetic", "model.pt")
    model.load_state_dict(torch.load(model_path))

    output_file = os.path.join("models", "10-14-Synthetic", "test.csv")

    states_df = pd.read_csv(os.path.join(data_directory, "states.csv"), header=None)
    actions_df = pd.read_csv(os.path.join(data_directory, "actions.csv"), header=None)

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Convert row to tensor
        state_tensor = torch.tensor(states_df.iloc[:, 0].values, dtype=torch.float, device=device).unsqueeze(0)
        action_tensor = torch.tensor(actions_df.values, dtype=torch.float, device=device).unsqueeze(0)

        print(state_tensor.shape)
        print(action_tensor.shape)
        
        traj = model.rollout(state_tensor, action_tensor, 800)

        data = tensor_to_numpy(traj.squeeze(0)).T

        for row in data:
            csv_writer.writerow(row)

if __name__ == "__main__":
    main()