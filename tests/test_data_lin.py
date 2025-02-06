import numpy as np
import pandas as pd

# Read the CSV files
actions_df = pd.read_csv('data/1-31-2-Synthetic/train/actions.csv', header=None)
tests_df = pd.read_csv('data/1-31-2-Synthetic/val/forces.csv', header=None)

print(f"Actions: {actions_df.shape}")
print(f"Tests: {tests_df.shape}")

actions_vectors = actions_df.values

test_vectors = tests_df.iloc[:, 2:6].values
L = 0.5
gamma = 0.1
transform_matrix = np.array([
    [-1.0, -1.0, -1.0, -1.0],
    [-L, L, L, -L],
    [L, -L, L, -L],
    [gamma, gamma, -gamma, -gamma]
])

print(f"Rank of A: {np.linalg.matrix_rank(transform_matrix)}")

transformed_vectors = np.dot(transform_matrix, actions_vectors).T

# Compare with test vectors
np.testing.assert_array_almost_equal(
    transformed_vectors,
    test_vectors,
    decimal=3,
    err_msg="Transformed vectors do not match test vectors"
)