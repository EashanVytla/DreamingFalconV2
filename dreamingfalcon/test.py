import torch
from dreamingfalcon.world_model import WorldModel
import pandas as pd
import os
from tqdm import tqdm
from dreamingfalcon.utils import AttrDict
import yaml
import csv

model_directory = "models/1-30-Synthetic"
data_directory = "data/1-27-Synthetic/train"
log_directory = "runs/1-30"

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
    
    model_path = os.path.join(model_directory, "model.pt")
    state = torch.load(model_path)
    model.load_state_dict(state["state_dict"])
    # model.states_mean = state["states_mean"]
    # model.states_std = state["states_std"]
    # model.actions_mean = state["actions_mean"]
    # model.actions_std = state["actions_std"]

    output_file = os.path.join(model_directory, "test.csv")
    forces_file = os.path.join(model_directory, "forces.csv")

    states_df = pd.read_csv(os.path.join(data_directory, "states.csv"), header=None)
    actions_df = pd.read_csv(os.path.join(data_directory, "actions.csv"), header=None)

    model.eval()

    with open(output_file, 'w', newline='') as trajfile, open(forces_file, 'w', newline='') as forcesfile:
        traj_writer = csv.writer(trajfile)
        force_writer = csv.writer(forcesfile)

        # Convert row to tensor
        state_tensor = torch.tensor(states_df.iloc[:, 0].values, dtype=torch.float, device=device).unsqueeze(0)
        action_tensor = torch.tensor(actions_df.values, dtype=torch.float, device=device).unsqueeze(0)

        print(state_tensor.shape)
        print(action_tensor.shape)
        
        forces, traj = model.rollout(state_tensor, action_tensor, 800)

        data = tensor_to_numpy(traj.squeeze(0)).T
        forces = tensor_to_numpy(forces.squeeze(0)).T

        for row in data:
            traj_writer.writerow(row)

        for row in forces:
            force_writer.writerow(row)

if __name__ == "__main__":
    main()