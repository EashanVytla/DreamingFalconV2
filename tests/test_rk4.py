import unittest
import torch
import math
from dreamingfalcon.rk4_solver import RK4_Solver
import numpy as np
import matplotlib.pyplot as plt

class TestRK4Solver(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.dt = 0.01
        self.solver = RK4_Solver(self.dt)
        
    def test_constant_function(self):
        """Test RK4 with dx/dt = 0"""
        def deriv(x, *args):
            return torch.zeros_like(x)
            
        x0 = torch.tensor([[1.0, 2.0, 3.0]], device=self.device)
        x1 = self.solver.step(x0, deriv)
        
        print("\nConstant function test:")
        print(f"Initial state: {x0}")
        print(f"Final state: {x1}")
        
        torch.testing.assert_close(x1, x0, rtol=1e-6, atol=1e-6)
        
    def test_linear_function(self):
        """Test RK4 with dx/dt = 1"""
        def deriv(x, *args):
            return torch.ones_like(x)
            
        x0 = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        x1 = self.solver.step(x0, deriv)
        expected = x0 + self.dt
        
        print("\nLinear function test:")
        print(f"Initial state: {x0}")
        print(f"Final state: {x1}")
        print(f"Expected state: {expected}")
        
        torch.testing.assert_close(x1, expected, rtol=1e-6, atol=1e-6)
        
    def test_simple_harmonic_oscillator(self):
        """Test RK4 with simple harmonic oscillator: d²x/dt² = -x"""
        def deriv(state, *args):
            # state = [x, v]
            x, v = state[:, 0], state[:, 1]
            return torch.stack([v, -x], dim=1)
            
        # Initial conditions: x = 1, v = 0
        x0 = torch.tensor([[1.0, 0.0]], device=self.device)
        
        # Simulate for one period
        period = 2 * math.pi
        steps = int(period / self.dt)
        x = x0
        
        print("\nSimple harmonic oscillator test:")
        # print(f"Number of steps: {steps}")
        print(f"Initial state: {x0}")
        # ypoints = []
        # xpoints = np.linspace(0, steps, steps)
        
        for _ in range(steps):
            x = self.solver.step(x, deriv)
            # ypoints.append(x[:, 1].item())
            
        # ypoints = np.array(ypoints)
        print(f"Final state after one period: {x}")

        # plt.plot(xpoints, ypoints, label="x(t)")
        # plt.xlabel("Time (t)")
        # plt.ylabel("Position (x)")
        # plt.title("Simple Harmonic Oscillator (RK4)")
        # plt.legend()
        # plt.show()
        
        # After one period, should return to initial conditions
        torch.testing.assert_close(x, x0, rtol=1e-2, atol=1e-2)
        
    def test_exponential_decay(self):
        """Test RK4 with exponential decay: dx/dt = -kx"""
        k = 1.0
        def deriv(x, *args):
            return -k * x
            
        x0 = torch.tensor([[1.0]], device=self.device)
        t = self.dt
        expected = x0 * math.exp(-k * t)
        x1 = self.solver.step(x0, deriv)
        
        print("\nExponential decay test:")
        print(f"Initial state: {x0}")
        print(f"Final state: {x1}")
        print(f"Expected state: {expected}")
        
        torch.testing.assert_close(x1, expected, rtol=1e-4, atol=1e-4)
        
    def test_batch_processing(self):
        """Test RK4 with batch of initial conditions"""
        def deriv(x, *args):
            return -x
            
        x0 = torch.tensor([[1.0], [2.0], [3.0]], device=self.device)
        t = self.dt
        expected = x0 * math.exp(-t)
        x1 = self.solver.step(x0, deriv)
        
        print("\nBatch processing test:")
        print(f"Initial states: {x0}")
        print(f"Final states: {x1}")
        print(f"Expected states: {expected}")
        
        torch.testing.assert_close(x1, expected, rtol=1e-4, atol=1e-4)
        
    def test_with_parameters(self):
        """Test RK4 with additional parameters"""
        def deriv(x, k):
            return -k * x
            
        k = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        x0 = torch.tensor([[1.0], [1.0], [1.0]], device=self.device)
        t = self.dt
        expected = x0 * torch.exp(-1 * k * t)
        x1 = self.solver.step(x0, deriv, k)
        
        print("\nParameter passing test:")
        print(f"Initial states: {x0}")
        print(f"Parameters: {k}")
        print(f"Final states: {x1}")
        print(f"Expected states: {expected}")
        
        torch.testing.assert_close(x1, expected, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print("Using GPU:", torch.cuda.get_device_name(0)) 
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")

    torch.set_default_device(device=device)
        
    unittest.main()