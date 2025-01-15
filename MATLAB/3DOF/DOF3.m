clc
clear

t = 0:0.01:10;  


scale_func = .475 * (-1/3000*t.^4 + 2/5*t);
motor_func = 1/10 * t;
%scale_func = 0.015*(-1/90*((t-5)/2).^7+2*((t-5)/2));
%scale_func = -1/5*(t-5).^2 + 1;
%scale_func = sin(t);

max_force = .5;

max_moment = .05;

max_altitude = .05;

distance = 1;

% Fx = max_force * scale_func;  
% Fz = 0 * t;    
% M = (max_moment * (.5 - scale_func));       


%Fx = max_force * ones(size(t));    % Constant forward force
%Fz = -9.81 + (.5- scale_func);                        % No vertical force
%Fz = -9.81 + 0.1*(scale_func - 0.5);  % Much smaller variation around gravity
%M = max_moment * (.5 - scale_func);
%Fz = -9.81 * ones(size(t));
Fx = max_force * scale_func;
Fz = -9.81 + 0.1*(scale_func);
M = max_moment * (scale_func);

pwm_change = 250 * motor_func;

motor_outputs = zeros(length(t), 4);
motor_outputs(:,1) = (1500 - pwm_change)';  % Front right (motor 1)
motor_outputs(:,2) = (1500 + pwm_change)';  % Back left (motor 2)
motor_outputs(:,3) = (1500 - pwm_change)';  % Front left (motor 3)
motor_outputs(:,4) = (1500 + pwm_change)';  % Back right (motor 4)

motor_outputs = transpose(motor_outputs);


Fx_ts = timeseries(Fx, t);
Fz_ts = timeseries(Fz, t);
M_ts = timeseries(M, t);


out = sim('Dreamer_3dof.slx', 'StopTime', '10');


time = out.time;
gamma = out.gamma;
q = out.get('q');
angaccel = out.get('angaccel');
pos = out.get('pos');
vel = out.get('vel');
acc = out.get('acc');
alpha = out.get('alpha');




Fx = transpose(Fx);
Fz = transpose(Fz);
M = transpose(M);

output = [gamma.Data,alpha.Data,q.Data,vel.Data,pos.Data];


output = transpose(output(:, [1:4 6:end]));



csvwrite('C:\Users\kbs_s\Documents\MATLAB\DREAMER\states.csv', output);
csvwrite('C:\Users\kbs_s\Documents\MATLAB\DREAMER\actions.csv', motor_outputs);

% Plot Boi
figure(1)
subplot(3,1,1)
plot(gamma)
title('Flight Path Angle')
xlabel('Time (s)')
ylabel('gamma (rad)')

subplot(3,1,2)
plot(q)
title('Pitch Rate')
xlabel('Time (s)')
ylabel('q (rad/s)')

subplot(3,1,3)
plot(alpha)
title('Angle of Attack')
xlabel('Time (s)')
ylabel('alpha (rad)')

figure(2)
plot(pos)
title('Flight Path')
xlabel('X Position (m)')
ylabel('Altitude (m)')
grid on