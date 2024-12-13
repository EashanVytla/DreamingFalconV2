import torch 
import math

# def euler_to_vector(euler_angles: torch.Tensor, device="cuda") -> torch.Tensor:
#     vector = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
#     rot = euler_angles_to_matrix(euler_angles, "XYZ")

#     print("Rotation matrix for the given Euler angles:\n", rot)

#     rotated_vector = torch.matmul(rot, vector)
#     # rotated_vector = rotated_vector / torch.norm(rotated_vector)

#     return rotated_vector

def unwrap(x):
    y = x % (2 * math.pi)
    return torch.where(y > math.pi, 2*math.pi - y, y)

def euclidean_distance(p, q):
    return torch.norm(p - q)

class AttrDict(dict):
    """Dictionary subclass that allows attribute access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_dict(data):
        """Recursively convert a dictionary into AttrDict."""
        if isinstance(data, dict):
            return AttrDict({key: AttrDict.from_dict(value) for key, value in data.items()})
        elif isinstance(data, list):
            return [AttrDict.from_dict(item) for item in data]
        else:
            return data