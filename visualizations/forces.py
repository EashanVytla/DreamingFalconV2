import pandas as pd

import matplotlib.pyplot as plt

# Load the data
file_path = 'models/12-26-2-Synthetic/forces.csv'
data = pd.read_csv(file_path, header=None, names=['Fx', 'Fy', 'M'])

# Plot Fx
plt.figure()
plt.plot(data['Fx'])
plt.title('Force in X direction (Fx)')
plt.xlabel('Index')
plt.ylabel('Fx')
plt.grid(True)
plt.show()

# Plot Fy
plt.figure()
plt.plot(data['Fy'])
plt.title('Force in Y direction (Fy)')
plt.xlabel('Index')
plt.ylabel('Fy')
plt.grid(True)
plt.show()

# Plot M
plt.figure()
plt.plot(data['M'])
plt.title('Moment (M)')
plt.xlabel('Index')
plt.ylabel('M')
plt.grid(True)
plt.show()