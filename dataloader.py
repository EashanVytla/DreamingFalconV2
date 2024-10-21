import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

class Pipeline:
    def __init__(self, csv_path_states, csv_path_actions, csv_path_rewards):
        self.csv_path_states = csv_path_states
        self.csv_path_actions = csv_path_actions

    def read_csv(self, batch_size=500):
        self.dataloader = DataLoader(SequenceDataset(self.csv_path_states, self.csv_path_actions), num_workers=os.cpu_count(), batch_size=batch_size, shuffle=True, pin_memory=True)

        return self.dataloader
    
class SequenceDataset(Dataset):
    def __init__(self, states_file, actions_file):
        self.states = pd.read_csv(states_file, header=None)
        self.actions = pd.read_csv(actions_file, header=None)
        print(f"States: {self.states.shape}, Actions: {self.actions.shape}")

    def __len__(self):
        return self.states.shape[1]

    def __getitem__(self, idx):
        states = torch.tensor(self.states.iloc[:, idx].values, dtype=torch.float32)
        actions = torch.tensor(self.actions.iloc[:, idx].values, dtype=torch.float32)
        return states, actions