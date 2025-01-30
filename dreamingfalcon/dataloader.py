import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

class Pipeline:
    def __init__(self, csv_path_states, csv_path_actions, csv_path_forces, seq_len=25):
        self.csv_path_states = csv_path_states
        self.csv_path_actions = csv_path_actions
        self.csv_path_forces = csv_path_forces
        self.seq_len = seq_len

    def read_csv(self, batch_size=500):
        self.dataloader = DataLoader(SequenceDataset(self.csv_path_states, self.csv_path_actions, self.csv_path_forces, seq_len=self.seq_len), num_workers=os.cpu_count(), batch_size=batch_size, shuffle=True, pin_memory=True)

        return self.dataloader
    
class SequenceDataset(Dataset):
    def __init__(self, states_file, actions_file, forces_file, seq_len=25):
        self.states = pd.read_csv(states_file, header=None)
        self.actions = pd.read_csv(actions_file, header=None)
        self.forces = pd.read_csv(forces_file, header=None, usecols=[0, 1, 2, 3, 4, 5]).T
        print(f"Forces shape: {self.forces.shape}")
        self.seq_len = seq_len
        # print(f"States: {self.states.shape}, Actions: {self.actions.shape}")

    def __len__(self):
        return self.states.shape[1] - self.seq_len + 1

    def __getitem__(self, idx):
        states = torch.tensor(self.states.iloc[:, idx:idx+self.seq_len].values, dtype=torch.float32)
        actions = torch.tensor(self.actions.iloc[:, idx:idx+self.seq_len].values, dtype=torch.float32)
        forces = torch.tensor(self.forces.iloc[:, idx:idx+self.seq_len].values, dtype=torch.float32)
        return states, actions, forces