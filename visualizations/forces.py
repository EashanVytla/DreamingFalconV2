import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Load both datasets
data1_path = 'output/1-31-2-Synthetic/forces.csv'
data2_path = "data/1-31-2-Synthetic/val/forces.csv"
data1 = pd.read_csv(data1_path, header=None, names=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
data2 = pd.read_csv(data2_path, header=None, usecols=[0, 1, 2, 3, 4, 5], names=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])

# Plot Fx
plt.figure()
plt.plot(data1['Fx'], label='Ground')
plt.plot(data2['Fx'], label='Predicted')
plt.title('Force in X direction (Fx)')
plt.xlabel('Index')
plt.ylabel('Fx')
plt.grid(True)
plt.legend()
plt.show()

# Plot Fy
plt.figure()
plt.plot(data1['Fy'], label='Ground')
plt.plot(data2['Fy'], label='Predicted')
plt.title('Force in Y direction (Fy)')
plt.xlabel('Index')
plt.ylabel('Fy')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(data1['Fz'], label='Ground')
plt.plot(data2['Fz'], label='Predicted')
plt.title('Force in Z direction (Fz)')
plt.xlabel('Index')
plt.ylabel('Fz')
plt.grid(True)
plt.legend()
plt.show()

# Plot Mx
plt.figure()
plt.plot(data1['Mx'], label='Ground')
plt.plot(data2['Mx'], label='Predicted')
plt.title('Moment (Mx)')
plt.xlabel('Index')
plt.ylabel('Mx')
plt.grid(True)
plt.legend()
plt.show()

# Plot My
plt.figure()
plt.plot(data1['My'], label='Ground')
plt.plot(data2['My'], label='Predicted')
plt.title('Moment (My)')
plt.xlabel('Index')
plt.ylabel('My')
plt.grid(True)
plt.legend()
plt.show()

# Plot Mz
plt.figure()
plt.plot(data1['Mz'], label='Ground')
plt.plot(data2['Mz'], label='Predicted')
plt.title('Moment (Mz)')
plt.xlabel('Index')
plt.ylabel('Mz')
plt.grid(True)
plt.legend()
plt.show()