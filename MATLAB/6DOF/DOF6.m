clc
clear

sample_time = 0.01;  % Explicitly define sample time
t = 0:sample_time:10;  % 0 to 10 seconds, 0.01s steps

% Define your forces and moments as regular vectors/arrays
Fx = 10*sin(2*pi*t);  % Example function
Fy = 5*cos(2*pi*t);   
Fz = -9.81 + sin(t);  
Mx = 2*sin(0.5*t);    
My = 1.5*cos(0.5*t);  
Mz = sin(0.25*t); 

% Create timeseries objects
Fx = timeseries(Fx', t');  % Note the transpose to make column vectors
Fy = timeseries(Fy', t');
Fz = timeseries(Fz', t');
Mx = timeseries(Mx', t');
My = timeseries(My', t');
Mz = timeseries(Mz', t');

% Set simulation time
set_param('mod_6Dof', 'StopTime', '10')
set_param('mod_6Dof', 'FixedStep', num2str(sample_time))

% Run the simulation
out = sim('mod_6Dof', 'StopTime', '10');

% Extract outputs
vel = out.vel;
pos = out.pos;
rot = out.rot;
vbod = out.vbod;
bod = out.bod;

% Save inputs for later if needed
input = [Fx.Data, Fy.Data, Fz.Data, Mx.Data, My.Data, Mz.Data]; 
input = transpose(input);


states = [pos.Data,vbod.Data,rot.Data,bod.Data];
states = transpose(states);

input = input(:, 1:min(1000, size(input, 2)));
states = states(:, 1:min(1000, size(states, 2)));

rando = [input;states];

rando = transpose(rando);

csvwrite('C:\Users\kbs_s\Documents\MATLAB\DREAMER\6DOF\input.csv', input);
csvwrite('C:\Users\kbs_s\Documents\MATLAB\DREAMER\6DOF\states.csv', states);
csvwrite('C:\Users\kbs_s\Documents\MATLAB\DREAMER\6DOF\test.csv', rando);