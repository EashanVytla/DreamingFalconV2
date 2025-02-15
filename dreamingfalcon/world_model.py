import torch
import torch.nn as nn
import torch.optim as optim
import dreamingfalcon.utils
import math
from dreamingfalcon.rk4_solver import RK4_Solver
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        
        layers = []
        
        hidden_dims = [input_dim] + hidden_dims
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            # layers.append(nn.LayerNorm(hidden_dims[i]))
            layers.append(nn.LeakyReLU())
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
        self._loss_scaler = config.training.loss_scaler
        self.epsilon = 1e-6
        self._beta = config.training.beta

        self.I = torch.tensor([[config.physics.I_xx, config.physics.I_xy, config.physics.I_xz],
                              [config.physics.I_yx, config.physics.I_yy, config.physics.I_yz],
                              [config.physics.I_zx, config.physics.I_zy, config.physics.I_zz]], device=device, dtype=torch.float32)
        
        self.I_inv = torch.inverse(self.I)

        self.model = MLP(config.force_model.input_dim, hidden_dims, config.force_model.output_dim)
        self.model.apply(self.init_weights)
        self.device = device

        self.solver = RK4_Solver(dt=self._rate)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # Example 1: Xavier Initialization (Good for Tanh/Sigmoid)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def compute_normalization_stats(self, dataloader):
        states_list = []
        actions_list = []
        
        for states, actions in dataloader:
            states_list.append(states)
            actions_list.append(actions)
        
        all_states = torch.cat(states_list, dim=0)
        all_actions = torch.cat(actions_list, dim=0)
        
        self.states_mean = all_states.mean(dim=0)
        self.states_std = all_states.std(dim=0) + self.epsilon
        
        self.actions_mean = all_actions.mean(dim=0)
        self.actions_std = all_actions.std(dim=0) + self.epsilon

        self.states_mean = self.states_mean.to(device=self.device)
        self.states_std = self.states_std.to(device=self.device)
        self.actions_mean = self.actions_mean.to(device=self.device)
        self.actions_std = self.actions_std.to(device=self.device)

        print(f"Data statistics: ")
        print(f"States mean: {self.states_mean}")
        print(f"States std: {self.states_std}")
        print(f"Actions mean: {self.actions_mean}")
        print(f"Actions mean: {self.actions_std}")

    def rollout(self, x_t, act_inps, seq_len):
        x_roll = [x_t]
        forces_roll = []
        prev_x = None
        for i in range(1, seq_len):
            forces, pred = self.predict(x_roll[i-1], act_inps[:, :, i])
            if torch.max(pred).item() > 1000 or torch.min(pred).item() < -1000:
                print(f"Warning: Large values detected at step {i}: {torch.max(pred)}")

            if prev_x is not None:
                delta = torch.abs(pred - prev_x).max().item()
                if delta > 1000:
                    print(f"Warning: Large state change detected at step {i}, delta: {delta}")
            prev_x = pred
            x_roll.append(pred)
            forces_roll.append(forces)

        stacked = torch.stack(x_roll, dim=2)
        stackedForces = torch.stack(forces_roll, dim=2)
        return stackedForces, stacked

    def get_L_EB(self, phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        phi = torch.as_tensor(phi, dtype=torch.float32)
        theta = torch.as_tensor(theta, dtype=torch.float32)
        psi = torch.as_tensor(psi, dtype=torch.float32)
        
        c_phi = torch.cos(phi)
        s_phi = torch.sin(phi)
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_psi = torch.cos(psi)
        s_psi = torch.sin(psi)
        
        m11 = c_theta * c_psi
        m12 = s_phi * s_theta * c_psi - c_phi * s_psi
        m13 = c_phi * s_theta * c_psi + s_phi * s_psi
        
        m21 = c_theta * s_psi
        m22 = s_phi * s_theta * s_psi + c_phi * c_psi
        m23 = c_phi * s_theta * s_psi - s_phi * c_psi
        
        m31 = -s_theta
        m32 = s_phi * c_theta
        m33 = c_phi * c_theta
        
        row1 = torch.stack([m11, m12, m13], dim=-1)
        row2 = torch.stack([m21, m22, m23], dim=-1)
        row3 = torch.stack([m31, m32, m33], dim=-1)
        
        L_EB = torch.stack([row1, row2, row3], dim=-2)
        
        return L_EB

    def _compute_derivatives(self, x, forces):
        """
        Compute state derivatives for RK4 integration
        State vector: Position: Xe, Ye, Ze (0:3)
                        Velocity: U, v, w (3:6)
                        Euler Rotation Angles: phi, theta, psi (6:9)
                        Body Rotation Rates: p, q, r (9:12)
        """
        V = x[:, 3:6]
        phi = x[:, 6]
        theta = x[:, 7]
        psi = x[:, 8]
        omega = x[:, 9:12]

        F = forces[:, 0:3]
        M = forces[:, 3:6]
        
        # Initialize derivative vector
        dx = torch.zeros_like(x, device=self.device)
        
        # Compute derivatives using equations of motion
        # Position derivatives (Earth Frame)
        dx[:, 0:3] = torch.matmul(self.get_L_EB(phi, theta, psi), V.unsqueeze(-1)).squeeze(-1)

        # Velocity derivative (Earth frame)
        dx[:, 3:6] = F/self._mass - torch.cross(omega, V)

        # print(dx[:, 3:6])
        
        # Rotation derivative (0 is phi, 1 is theta)
        J = torch.zeros((x.shape[0], 3, 3), device=self.device, dtype=torch.float32)
        J[:, 0, 0] = 1
        J[:, 0, 1] = torch.sin(phi) * torch.tan(theta)
        J[:, 0, 2] = torch.cos(phi) * torch.tan(theta)
        J[:, 1, 1] = torch.cos(phi)
        J[:, 1, 2] = -torch.sin(phi)
        J[:, 2, 1] = torch.sin(phi) / torch.clamp(torch.cos(theta), min=self.epsilon)
        J[:, 2, 2] = torch.cos(phi) / torch.clamp(torch.cos(theta), min=self.epsilon)

        dx[:, 6:9] = torch.matmul(J, omega.unsqueeze(-1)).squeeze(-1)

        # Rotation rate derivative (Body-fixed frame)
        Iw = torch.matmul(self.I, omega.unsqueeze(-1))
        coriolis = torch.cross(omega, Iw.squeeze(-1), dim=1)
        dx[:, 9:12] = torch.matmul(self.I_inv, (M - coriolis).unsqueeze(-1)).squeeze(-1)
        
        return dx
    
    def six_dof(self, x_t, forces):
        return self.solver.step(x_t, self._compute_derivatives, forces)

    def predict(self, x_t, actuator_input):
        """
        Predict next state using forces from MLP and RK4 integration
        State vector: [gamma, alpha, q, V, Xe, Ze]
        """
        # Normalize states for neural network input
        norm_x_t = torch.zeros_like(x_t, dtype=torch.float32, device=self.device)
        norm_x_t[:, 0] = x_t[:, 0] / (torch.pi/4)    # Flight path angle: ±45 degrees
        norm_x_t[:, 1] = x_t[:, 1] / 2.0     # Angle of attack: ±2 radians
        norm_x_t[:, 2] = x_t[:, 2] / (torch.pi/4)     # Pitch rate: ±45 deg/s
        norm_x_t[:, 3] = x_t[:, 3] / 10       # Velocity: ±10 m/s
        norm_x_t[:, 4:6] = x_t[:, 4:6] / 60          # Positions: ±60m range

        # print(f"x_t shape: {x_t.shape}, states shape: {self.states_mean.shape}")

        # norm_x_t = (x_t - self.states_mean.T)/self.states_std.T
        # norm_act = (actuator_input - self.actions_mean.T)/self.actions_std.T

        # Normalize actuator inputs
        norm_act = (actuator_input - 1500) / 500       # Actuator inputs: 1000-2000 range
        
        # Get forces from MLP
        inp = torch.cat((norm_x_t, norm_act), dim=1)
        forces_norm = self.model(inp)
        # print(torch.max(forces_norm))
        
        # Denormalize forces
        forces = torch.zeros_like(forces_norm, device=self.device)
        forces[:, 0] = forces_norm[:, 0] - 0.5        # Fx: -0.1 to 0.1
        forces[:, 1] = forces_norm[:, 1] * -0.4 - 10        # Fz: 9.6 to 10
        forces[:, 2] = forces_norm[:, 2] * 0.2 - 0.1  # Mr: -0.1 to 0.1
        
        return forces, self.three_dof(x_t, forces)
    
    def loss(self, pred, truth):
        weights = torch.ones_like(pred, device=self.device)
        weights[:, 0] *= (torch.pi/4)      # Flight path angle: ±45 degrees
        weights[:, 1] *= 2.0      # Angle of attack: ±30 degrees
        weights[:, 2] *= (torch.pi/4)      # Pitch rate: ±90 deg/s
        weights[:, 3] *= 10               # Velocity: 20 m/s
        weights[:, 4:6] *= 60             # Positions: ±100m range

        time_steps = pred.shape[2]
        time_weights = torch.linspace(256.0, 256 - time_steps, time_steps, device=self.device).view(1, 1, time_steps)

        # Compute Huber loss (smooth L1) for each element
        huber_loss = F.smooth_l1_loss(pred, truth, reduction='none', beta=self._beta)

        # Apply weighting and then take mean
        return torch.mean((huber_loss / weights) * time_weights)