import unittest
import torch
import pandas as pd
from pathlib import Path
from dreamingfalcon.world_model import WorldModel
from dreamingfalcon.utils import AttrDict
import yaml
import math

class TestSixDOF(unittest.TestCase):
    def setUp(self):
        with open('config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)

        self.config = AttrDict.from_dict(config_dict)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use the GPU
            print("Using GPU:", torch.cuda.get_device_name(0)) 
        else:
            self.device = torch.device("cpu")  # Use the CPU
            print("Using CPU")
                
        self.model = WorldModel(self.config, self.device)
        
    def test_six_dof_integration(self):
        """Test 6dof integration against recorded data"""
        # Load test data
        excel_path = Path("tests/data/6dofTestData.csv")
        df = pd.read_csv(excel_path, header=None)
        
        # Extract forces and states
        forces = torch.tensor(df.iloc[:, :6].values, dtype=torch.float32, device=self.device)  # First 6 columns: Fx, Fy, Fz, Mx, My, Mz
        states = torch.tensor(df.iloc[:, 6:18].values, dtype=torch.float32, device=self.device)  # Next 12 columns are states
        angle_wrap = 0

        print("\nTesting 6dof integration:")
        print(f"Loaded {len(states)} timesteps of data")
        adjustment = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
        
        # Test integration for each consecutive pair of states
        for i in range(len(states) - 1):
            # Current state and forces
            current_state = states[i:i+1]  # Add batch dimension
            current_forces = forces[i:i+1]
            # current_forces[:, 2] += self.config.physics.g * self.config.physics.mass
            
            # Ground truth next state
            next_state_true = states[i+1:i+2]

            # diff = next_state_true[:, 6:9] - current_state[:, 6:9]
            # adjustment += torch.where(torch.abs(diff) > (2 * torch.pi - 1e-4), torch.sign(diff) * (2 * torch.pi), torch.zeros_like(diff))
            # next_state_true[:, 6:9] += adjustment
            
            # Integrate to get next state
            next_state_pred = self.model.six_dof(current_state, current_forces)
            
            # Compute relative errors (ignoring near-zero values)
            mask = torch.abs(next_state_true) > 1e-6
            relative_errors = torch.abs((next_state_pred[mask] - next_state_true[mask]) / next_state_true[mask])
            
            max_rel_error = torch.max(relative_errors).item()
            mean_rel_error = torch.mean(relative_errors).item()

            prev_state = current_state
            
            if i < 5 or max_rel_error > 0.1 or mean_rel_error > 0.1:  # Print first 5 steps
                print(f"\nTimestep {i}:")
                print(f"Forces: {current_forces.cpu().numpy()}")
                print(f"Current state: {current_state.cpu().numpy()}")
                print(f"Predicted next state: {next_state_pred.cpu().numpy()}")
                print(f"Actual next state: {next_state_true.cpu().numpy()}")
                print(f"Relative Error: {relative_errors.cpu().numpy()}")
                print(f"Max relative error: {max_rel_error:.6f}")
                print(f"Mean relative error: {mean_rel_error:.6f}")
                
                # Print specific components with large errors
                if max_rel_error > 0.1:
                    component_errors = (next_state_pred - next_state_true) / next_state_true
                    large_error_indices = torch.where(torch.abs(component_errors) > 0.1)[1]
                    print("\nLarge errors in components:")
                    state_names = ['Xe', 'Ye', 'Ze', 'U', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r']
                    for idx in large_error_indices:
                        print(f"{state_names[idx]}: error = {component_errors[0, idx].item():.6f}")

if __name__ == '__main__':
    unittest.main()