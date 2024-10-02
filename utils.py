import torch 

def rotation_matrix_3d(theta_x, theta_y, theta_z):
    """
    Creates a 3D rotation matrix for rotations about the X, Y, and Z axes by angles theta_x, theta_y, and theta_z (in radians).
    
    Args:
    - theta_x: Rotation angle around X-axis (in radians).
    - theta_y: Rotation angle around Y-axis (in radians).
    - theta_z: Rotation angle around Z-axis (in radians).
    
    Returns:
    - A combined 3x3 rotation matrix tensor.
    """
    # Rotation matrices for each axis
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta_x), -torch.sin(theta_x)],
                        [0, torch.sin(theta_x), torch.cos(theta_x)]])
    
    R_y = torch.tensor([[torch.cos(theta_y), 0, torch.sin(theta_y)],
                        [0, 1, 0],
                        [-torch.sin(theta_y), 0, torch.cos(theta_y)]])
    
    R_z = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                        [torch.sin(theta_z), torch.cos(theta_z), 0],
                        [0, 0, 1]])
    
    # Combine the rotations (R = Rz * Ry * Rx)
    rotation_matrix = R_z @ R_y @ R_x
    return rotation_matrix

# Input attitude in radians!
def euler_to_unit_sphere(attitude):
    pitch, roll, yaw = attitude[:, 0], attitude[:, 1], attitude[:, 2]

    # Compute sine and cosine values
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    # Create rotation matrices
    R_x = torch.stack([
        torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
        torch.zeros_like(pitch), cos_pitch, -sin_pitch,
        torch.zeros_like(pitch), sin_pitch, cos_pitch
    ], dim=1).reshape(-1, 3, 3)

    R_y = torch.stack([
        cos_roll, torch.zeros_like(roll), sin_roll,
        torch.zeros_like(roll), torch.ones_like(roll), torch.zeros_like(roll),
        -sin_roll, torch.zeros_like(roll), cos_roll
    ], dim=1).reshape(-1, 3, 3)

    R_z = torch.stack([
        cos_yaw, -sin_yaw, torch.zeros_like(yaw),
        sin_yaw, cos_yaw, torch.zeros_like(yaw),
        torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    ], dim=1).reshape(-1, 3, 3)

    # Combined rotation matrix
    R = torch.bmm(torch.bmm(R_z, R_y), R_x)

    # Extract the first column as our representative vector
    return R[:, :, 0]

def euclidean_distance(p, q):
    return torch.norm(p - q)