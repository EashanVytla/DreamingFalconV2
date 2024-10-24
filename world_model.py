import torch
import torch.nn as nn
import torch.optim as optim
import utils
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        
        layers = []
        
        hidden_dims = [input_dim] + hidden_dims
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class WorldModel(nn.Module):
    def __init__(self, config):
        super(WorldModel, self).__init__()
        # initialize the MLP
        hidden_dims = [config.force_model.hidden_size] * config.force_model.hidden_layers

        self._rate = config.physics.refresh_rate
        self._mass = config.physics.mass
        self._g = config.physics.g
        self._k = config.physics.k

        self.model = MLP(config.force_model.input_dim, hidden_dims, config.force_model.output_dim)

    def predict(self, x_t, actuator_input):
        # Run feedforward on MLP
        # Actuator Input: (4, batch_size)
        # x_t: (12, batch_size)
        # Output: (6, batch_size)
        attitude, velo, ang_velo, pose = x_t[:, :3], x_t[:, 3:6], x_t[:, 6:9], x_t[:, 9:12]


        norm_x_t = torch.zeros_like(x_t, dtype=torch.float32)
        norm_x_t[:, :3] = x_t[:, :3] / (2*math.pi)
        norm_x_t[:, 3:6] = x_t[:, 3:6] / 20
        norm_x_t[:, 6:9] = x_t[:, 6:9] / (2*math.pi)
        norm_x_t[:, 9:12] = x_t[:, 9:12] / 1000
        norm_act = (actuator_input - 1000)/1000

        # print(f"X_dim: {norm_x_t.shape}, Act: {norm_act.shape}")
        inp = torch.cat((norm_x_t, norm_act), dim=1)
        output = self.model(inp)
        force, ang_acel = output[:, :3], output[:, 3:]

        '''
        State:
        1)  theta_x (pitch)
        2)  theta_y (roll)
        3)  theta_z (yaw)
        4)  v_x
        5)  v_y
        6)  v_z
        7)  w_x (pitch)
        8)  w_y (roll)
        9)  w_z (yaw)
        10) p_x
        11) p_y
        12) p_z
        '''

        ang_vel_t1 = ang_acel * self._rate + ang_velo
        attitude_t1 = attitude + ang_velo * self._rate + 0.5 * ang_acel * self._rate ** 2

        with torch.no_grad():
            rot = euler_angles_to_matrix(attitude_t1, "XYZ")

        # print(f"Force Shape: {force.shape}, Rot Shape: {rot.shape}")

        rot_force = torch.bmm(force.unsqueeze(1), rot).squeeze(1)

        # print(f"Rotated Force: {rot_force.shape}, Velo Shape: {velo.shape}")

        a = (1/self._mass) * (rot_force - torch.tensor([0, 0, self._g], dtype=torch.float32, device=x_t.device) - self._k * torch.tensor([1, 0, 0], dtype=torch.float32, device=x_t.device) * torch.square(velo))
        velo_t1 = velo + a * self._rate
        pose_t1 = pose + velo * self._rate + 0.5 * a * self._rate ** 2

        x_t1 = torch.cat([attitude_t1, velo_t1, ang_vel_t1, pose_t1], dim=1)

        return x_t1

    def loss(self, pred, truth):
        #Compute the loss (Quaternion loss)
        #return utils.euclidean_distance(utils.euler_to_vector(pred), utils.euler_to_vector(truth))

        #MSE:
        criterion = torch.nn.MSELoss()
        return criterion(pred, truth)