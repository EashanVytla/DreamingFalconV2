import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_states(state_data):
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    fig.suptitle('State Features Over Time', fontsize=10)
    
    axes = axes.flatten()
    
    if isinstance(state_data, pd.DataFrame):
        state_data = state_data.to_numpy()
    
    time_steps = np.arange(state_data.shape[1])
    
    for i in range(12):
        axes[i].plot(time_steps, state_data[i], 'b-', linewidth=2)
        axes[i].set_title(f'Feature {i+1}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

ground_truth = pd.read_csv('data/1-24-Synthetic/train/states.csv', header=None).to_numpy()
visualize_states(ground_truth)

plt.show()