import torch
import torch.nn as nn
import torch.optim as optim
import utils
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
            layers.append(nn.Dropout(p=0.1))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class WorldModel(nn.Module):
    def __init__(self, config, device):
        super(WorldModel, self).__init__()
        # initialize the MLP
        hidden_dims = [config.force_model.hidden_size] * config.force_model.hidden_layers

        self._rate = config.physics.refresh_rate
        self._mass = config.physics.mass
        self._g = config.physics.g
        self._k = config.physics.k

        self.model = MLP(config.force_model.input_dim, hidden_dims, config.force_model.output_dim)
        self.device = device
    
    def compute_normalization_stats(self, dataloader):
        states_list = []
        actions_list = []
        
        for states, actions in dataloader:
            states_list.append(states)
            actions_list.append(actions)
        
        all_states = torch.cat(states_list, dim=0)
        all_actions = torch.cat(actions_list, dim=0)
        
        self.states_mean = all_states.mean(dim=0)
        self.states_std = all_states.std(dim=0) + 1e-6
        
        self.actions_mean = all_actions.mean(dim=0)
        self.actions_std = all_actions.std(dim=0) + 1e-6

    def rollout(self, x_t, act_inps, seq_len):
        x_roll = [x_t]
        for i in range(1, seq_len):
            x_roll.append(self.predict(x_roll[i-1], act_inps[:, :, i]))
            # print(f"X Roll: {x_roll[i]}")

        stacked = torch.stack(x_roll, dim=2)
        return stacked

    def predict(self, x_t, actuator_input):
        # Run feedforward on MLP
        # Actuator Input: (4, batch_size)
        # x_t: (12, batch_size)
        # Output: (6, batch_size)
        attitude, velo = x_t[:, :3], x_t[:, 3:6]

        norm_x_t = torch.zeros_like(x_t, dtype=torch.float32, device=self.device)
        norm_x_t[:, :3] = x_t[:, :3] / (2*math.pi)
        norm_x_t[:, 3:6] = x_t[:, 3:6] / 20
        norm_x_t[:, 6:9] = x_t[:, 6:9] / (2*math.pi)
        norm_x_t[:, 9:12] = x_t[:, 9:12] / 10
        norm_act = (actuator_input - 1000)/1000

        inp = torch.cat((norm_x_t, norm_act), dim=1)
        output = self.model(inp)
        
        wrapped_theta_dot = torch.atan2(torch.sin(output[:, 0:3]), torch.cos(output[:, 0:3]))
        wrapped_w_dot = torch.clamp(output[:, 3:6], -math.pi/2, math.pi/2)

        theta_x_dot = wrapped_theta_dot[:, 0]
        theta_y_dot = wrapped_theta_dot[:, 1]
        theta_z_dot = wrapped_theta_dot[:, 2]
        w_x_dot = wrapped_w_dot[:, 0]
        w_y_dot = wrapped_w_dot[:, 1]
        w_z_dot = wrapped_w_dot[:, 2]
        p_x_dot = output[:, 6]
        p_y_dot = output[:, 7]
        p_z_dot = output[:, 8]

        print(f"Theta dot: {wrapped_theta_dot[0,:]}")

        eps = 1e-6
        
        theta_x_t1 = attitude[:, 0] + (theta_x_dot + torch.sin(attitude[:, 0]) * torch.tan(attitude[:, 0]) * theta_y_dot + torch.cos(attitude[:, 0]) * torch.tan(attitude[:, 1]) * theta_z_dot) * self._rate
        theta_y_t1 = attitude[:, 1] + (theta_y_dot * torch.cos(attitude[:, 0]) - torch.sin(attitude[:, 0])) * self._rate
        theta_z_t1 = attitude[:, 2] + ((torch.sin(attitude[:, 0])/torch.cos(attitude[:, 1] + eps)) + torch.cos(attitude[:, 0])/torch.cos(attitude[:, 1] + eps) * theta_z_dot) * self._rate
        velo_x_t1 = velo[:, 0] + (w_x_dot - (theta_y_dot * p_z_dot - theta_z_dot * p_y_dot) + self._g * torch.sin(attitude[:, 1])) * self._rate
        velo_y_t1 = velo[:, 1] + (w_y_dot - (theta_z_dot * p_x_dot - theta_x_dot * p_z_dot) - self._g * torch.cos(attitude[:, 1]) * torch.sin(attitude[:, 0])) * self._rate
        velo_z_t1 = velo[:, 2] + (w_z_dot - (theta_x_dot * p_y_dot - theta_y_dot * p_x_dot) - self._g * torch.cos(attitude[:, 1]) * torch.sin(attitude[:, 0]) + self._g) * self._rate
        wp_t1 = x_t[:, 6:12] + output[:, 3:9] * self._rate

        x_t1 = torch.cat([theta_x_t1.unsqueeze(1), theta_y_t1.unsqueeze(1), theta_z_t1.unsqueeze(1), velo_x_t1.unsqueeze(1), velo_y_t1.unsqueeze(1), velo_z_t1.unsqueeze(1), wp_t1], dim=1)

        return x_t1
    
    def loss(self, pred, truth):
        angle_diff = torch.remainder(pred[:, :3] - truth[:, :3] + math.pi, 2*math.pi) - math.pi
        other_diff = pred[:, 3:] - truth[:, 3:]
        diff = torch.cat([angle_diff, other_diff], dim=1)

        weights = torch.ones_like(pred, device=self.device)
        weights[:, :3] *= 10.0  # attitude weights
        weights[:, 3:6] *= 1.0  # velocity weights
        weights[:, 6:9] *= 10.0  # angular velocity weights
        weights[:, 9:12] *= 1.0  # position weights
        
        # Weighted MSE loss
        return torch.mean(weights * torch.square(diff))