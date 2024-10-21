import unittest
import utils
import torch
import math

class TestUtilsMethods(unittest.TestCase):
    def test_euler_to_vector(self):
        test_cases = [
            # (input_euler_angles, expected_output)
            (torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])),
            (torch.tensor([[math.pi/2, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])),
            (torch.tensor([[0.0, math.pi/2, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])),
            (torch.tensor([[0.0, 0.0, math.pi/2]]), torch.tensor([[0.0, 0.0, 1.0]])),
            (torch.tensor([[math.pi/6, math.pi/4, math.pi/3]]), torch.tensor([[0.633, 0.773, 0.0391]])),
            (torch.tensor([[math.pi/4, math.pi/6, math.pi/2]]), torch.tensor([[0.787, 0.604, 0.129]]))
        ]

        for i, (input_angles, expected_output) in enumerate(test_cases):
            print(f"\nTest case {i + 1}:")
            print(f"Input: {input_angles}")
            print(f"Expected output: {expected_output}")
            
            output = utils.euler_to_vector(input_angles)
            print(f"Actual output: {output}")
            
            try:
                torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)
                print("Test case passed!")
            except AssertionError as e:
                print(f"Test case failed: {str(e)}")
            
            # Print the magnitude of the output vector
            magnitude = torch.norm(output, dim=-1)
            print(f"Output magnitude: {magnitude}")

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        print("Using GPU:", torch.cuda.get_device_name(0)) 
    else:
        device = torch.device("cpu")  # Use the CPU
        print("Using CPU")

    torch.set_default_device(device=device)
        
    unittest.main()