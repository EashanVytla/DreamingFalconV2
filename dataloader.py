import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os

class Pipeline:
    def __init__(self, csv_path_states, csv_path_actions, csv_path_rewards):
        self.csv_path_states = csv_path_states
        self.csv_path_actions = csv_path_actions

    def read_csv(self, batch_size=500):
        self.dataloader = DataLoader(SequenceDataset(self.csv_path_states, self.csv_path_action), num_workers=os.cpu_count(), batch_size=batch_size, shuffle=True, pin_memory=True)

        return self.dataloader
    
class SequenceDataset(Dataset):
    def __init__(self, states_file, actions_file):
        self.states = pd.read_csv(states_file)
        self.actions = pd.read_csv(actions_file)

    def __len__(self):
        length = len(self.states) - self.seq_len + 1
        return length

    def __getitem__(self, idx):
        states_seq = self.states[idx]
        actions_seq = self.actions[idx]
        return states_seq, actions_seq