import torch
import torch.nn as nn
import torch.optim as optim
import utils

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

class WorldModel():
    def __init__(self):
        # initialize the MLP
        input_dim = 4
        hidden_dims = [32, 32]
        output_dim = 6
        state_dim = 12

        self.rate = 0.02
        self.mass = 35 # g
        self.g = 9.81
        self.k = 0.25 # Drag coefficient

        self.model = MLP(input_dim, hidden_dims, output_dim) 

    def predict(self, x_t, actuator_input):
        #Run feedforward on MLP
        output = self.model(actuator_input)
        force, ang_acel = output[:3], output[3:]
        x_t1 = torch.zeros_like(x_t)

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

        attitude, velo, ang_velo, pose = x_t[:3], x_t[3:6], x_t[6:9], x_t[9:12]
        attitude_t1, velo_t1, ang_velo_t1, pose_t1 = x_t1[:3], x_t1[3:6], x_t1[6:9], x_t1[9:12]

        ang_velo_t1 = ang_acel * self.rate + ang_velo
        attitude_t1 = attitude + ang_velo * self.rate + 0.5 * ang_acel * self.rate^2

        rot = utils.rotation_matrix_3d(attitude_t1)

        a = (1/self.mass) * rot * force - [0, 0, self.g].T - (self.k/self.m) * velo
        velo_t1 = velo + a * self.rate

        pose_t1 = pose + velo * self.rate + 0.5 * a * self.rate^2

        return x_t1

    def loss(self, pred, truth):
        #Compute the loss (Quaternion loss)
        

    def train(self):
        #Train the model