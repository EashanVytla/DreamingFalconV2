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
            layers.append(nn.LayerNorm(hidden_dims[i]))
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
    
    def compute_normalization_stats(self, dataloader):
        """Compute means and stds from the dataset"""
        # Collect all states
        states_list = []
        actions_list = []
        
        for states, actions in dataloader:
            states_list.append(states)
            actions_list.append(actions)
        
        all_states = torch.cat(states_list, dim=0)
        all_actions = torch.cat(actions_list, dim=0)
        
        # Compute means and stds
        self.states_mean = all_states.mean(dim=0)
        self.states_std = all_states.std(dim=0) + 1e-6  # add epsilon to avoid division by zero
        
        self.actions_mean = all_actions.mean(dim=0)
        self.actions_std = all_actions.std(dim=0) + 1e-6

    def rollout(self, x_t, act_inps, seq_len):
        x_roll = [x_t]
        for i in range(1, seq_len):
            x_roll.append(self.predict(x_roll[i-1], act_inps[i]))

        x_traj = torch.stack(x_roll, dim=1)

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
        norm_x_t[:, 9:12] = x_t[:, 9:12] / 10
        norm_act = (actuator_input - 1000)/1000

        # norm_x_t = (x_t - self.states_mean) / self.states_std
        # norm_act = (actuator_input - self.actions_mean) / self.actions_std

        # print(f"Norm_x:\n{norm_x_t}")
        # print(f"Norm_act:\n{norm_act}")

        print(f"X_dim: {norm_x_t.shape}, Act: {norm_act.shape}")
        inp = torch.cat((norm_x_t, norm_act), dim=1)
        output = self.model(inp)
        force, ang_acel = output[:, :3], output[:, 3:]

        print(f"Force: {force}\nAng Acel: {ang_acel}")

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

        rot = euler_angles_to_matrix(attitude_t1, "XYZ")

        rot_force = torch.bmm(force.unsqueeze(1), rot).squeeze(1)

        a = (1/self._mass) * (rot_force - torch.tensor([0, 0, self._g], dtype=torch.float32, device=x_t.device) - self._k * torch.tensor([1, 0, 0], dtype=torch.float32, device=x_t.device) * torch.square(velo))
        velo_t1 = velo + a * self._rate
        pose_t1 = pose + velo * self._rate + 0.5 * a * self._rate ** 2

        x_t1 = torch.cat([attitude_t1, velo_t1, ang_vel_t1, pose_t1], dim=1)

        return x_t1

    def loss(self, pred, truth):
        #Compute the loss (Quaternion loss)
        #return utils.euclidean_distance(utils.euler_to_vector(pred), utils.euler_to_vector(truth))

        weights = torch.ones_like(pred)
        weights[:, :3] *= 1.0  # attitude weights
        weights[:, 3:6] *= 0.1  # velocity weights
        weights[:, 6:9] *= 1.0  # angular velocity weights
        weights[:, 9:12] *= 0.1  # position weights
        
        # Weighted MSE loss
        # return torch.mean(weights * (pred - truth) ** 2)
        return torch.mean((pred - truth) ** 2)