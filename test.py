import unittest
import utils
import torch

class TestUtilsMethods(unittest.TestCase):
    def test_euler_to_unit_sphere(self):
        test_vars = [
            (torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])),
            (torch.tensor([[90.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])),
            (torch.tensor([[0.0, 90.0, 0.0]]), torch.tensor([[0.0, 0.0, 1.0]])),
            (torch.tensor([[0.0, 0.0, 90.0]]), torch.tensor([[0.0, -1.0, 0.0]])),
            (torch.tensor([[30.0, 45.0, 60.0]]), torch.tensor([[0.3536, 0.6124, -0.7071]])),
            (torch.tensor([[45.0, 30.0, 90.0]]), torch.tensor([[0.0, 0.8660, -0.5000]])),
            (torch.tensor([[30.0, 45.0, 60.0], [45.0, 30.0, 90.0]]), 
            torch.tensor([[0.3536, 0.6124, -0.7071], [0.0, 0.8660, -0.5000]])),
        ]

        for case in test_vars:
            output = utils.euler_to_unit_sphere(case[0])
            assert torch.testing.assert_close(output, case[1], rtol=1e-4, atol=1e-4)

    def test_output_shape(self):
        input_attitude = torch.rand(10, 3) * 360.0  # Random angles between 0 and 360 degrees
        output = utils.euler_to_unit_sphere(input_attitude)
        assert output.shape == (10, 3)

    def test_unit_vector(self):
        input_attitude = torch.rand(100, 3) * 360.0  # Random angles between 0 and 360 degrees
        output = utils.euler_to_unit_sphere(input_attitude)
        magnitudes = torch.norm(output, dim=1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_large_batch(self):
        input_attitude = torch.rand(1000000, 3) * 360.0  # Large batch of random angles
        output = utils.euler_to_unit_sphere(input_attitude)
        assert output.shape == (1000000, 3)
        assert torch.allclose(torch.norm(output, dim=1), torch.ones(1000000), atol=1e-6)

if __name__ == '__main__':
    unittest.main()