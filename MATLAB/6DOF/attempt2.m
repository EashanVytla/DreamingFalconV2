% Read the simulation data
data = readmatrix('test.csv');

% Calculate time step from the data
dt = 0.01; 
t = 0:dt:(size(data,1)-1)*dt;

% Parameters (from block parameters)
mass = 1.0;  % Initial mass parameter
I = eye(3);  % Inertia matrix (eye(3) shown in parameters)

% Initialize arrays to store our calculated derivatives
X_dot_calc = zeros(length(t)-1, 12);

% Calculate derivatives at each time step
for i = 1:length(t)-1
    % Get current states
    x = data(i,7); y = data(i,8); z = data(i,9);
    u = data(i,10); v = data(i,11); w = data(i,12);
    phi = wrapToPi(data(i,13));
    theta = data(i,14); 
    psi = data(i,15);
    p = data(i,16); q = data(i,17); r = data(i,18);
    
    % Get forces and moments
    Fx = data(i,1); Fy = data(i,2); Fz = data(i,3);
    Mx = data(i,4); My = data(i,5); Mz = data(i,6);
    
    % Current vectors
    omega = [p; q; r];
    V_b = [u; v; w];
    
    % Get DCM
    L_EB = get_L_EB(phi, theta, psi);
    
    % Cross product for translational dynamics
    omega_cross_V = cross(omega, V_b);
    
    % Translational dynamics (body frame)
    V_b_dot = [Fx/mass; Fy/mass; Fz/mass] - omega_cross_V;
    
    u_dot = V_b_dot(1);
    v_dot = V_b_dot(2);
    w_dot = V_b_dot(3);
    
    % Rotational dynamics with identity inertia matrix
    I_omega = I * omega;
    omega_cross_Iomega = cross(omega, I_omega);
    M = [Mx; My; Mz];
    omega_dot = I\(M - omega_cross_Iomega);
    
    p_dot = omega_dot(1);
    q_dot = omega_dot(2);
    r_dot = omega_dot(3);
    
    % Euler angle rates
    phi_dot = p + (q*sin(phi) + r*cos(phi))*tan(theta);
    theta_dot = q*cos(phi) - r*sin(phi);
    psi_dot = (q*sin(phi) + r*cos(phi))/cos(theta);
    
    % Position rates using transformation matrix
    pos_dot = L_EB * V_b;
    
    x_dot = pos_dot(1);
    y_dot = pos_dot(2);
    z_dot = pos_dot(3);
    
    % Store calculated derivatives
    X_dot_calc(i,:) = [x_dot, y_dot, z_dot, u_dot, v_dot, w_dot, ...
                       phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot];
end

% Calculate actual derivatives from data
wrapped_phi = wrapToPi(data(:,13));
data_wrapped = data;
data_wrapped(:,13) = wrapped_phi;
X_dot_actual = diff(data_wrapped(:,7:18))/dt;

% Create figure with better formatting
figure('Position', [100, 100, 1200, 800]);

state_names = {'X', 'Y', 'Z', 'U', 'V', 'W', 'Phi', 'Theta', 'Psi', 'P', 'Q', 'R'};
units = {'(m)', '(m)', '(m)', '(m/s)', '(m/s)', '(m/s)', '(rad)', '(rad)', '(rad)', '(rad/s)', '(rad/s)', '(rad/s)'};

for i = 1:12
    subplot(3,4,i);
    plot(t(1:end-1), X_dot_actual(:,i), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(t(1:end-1), X_dot_calc(:,i), 'r--', 'LineWidth', 1.5);
    grid on;
    title([state_names{i} ' Rate ' units{i}], 'FontWeight', 'bold');
    xlabel('Time (s)');
    if i == 1
        legend('Simulink', 'Calculated', 'Location', 'best');
    end
    hold off;
end

% Adjust subplot spacing
set(gcf, 'Color', 'w');
sgtitle('6-DOF Motion Derivatives Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% Calculate error metrics
abs_errors = abs(X_dot_actual - X_dot_calc);
rms_errors = sqrt(mean((X_dot_actual - X_dot_calc).^2));
max_errors = max(abs_errors);

% Create error metrics figure
figure('Position', [100, 100, 800, 400]);

subplot(1,2,1);
bar(rms_errors);
set(gca, 'XTick', 1:12, 'XTickLabel', state_names);
title('RMS Errors', 'FontWeight', 'bold');
xtickangle(45);
grid on;
ylabel('RMS Error');

subplot(1,2,2);
bar(max_errors);
set(gca, 'XTick', 1:12, 'XTickLabel', state_names);
title('Maximum Absolute Errors', 'FontWeight', 'bold');
xtickangle(45);
grid on;
ylabel('Max Error');

% Print numerical results
fprintf('\nError Analysis:\n');
fprintf('%-8s %-15s %-15s\n', 'State', 'RMS Error', 'Max Error');
fprintf('----------------------------------------\n');
for i = 1:12
    fprintf('%-8s %-15e %-15e\n', state_names{i}, rms_errors(i), max_errors(i));
end

% Function to compute body-to-Earth transformation matrix
function L_EB = get_L_EB(phi, theta, psi)
    % ZYX rotation sequence for NED
    c_phi = cos(phi); s_phi = sin(phi);
    c_theta = cos(theta); s_theta = sin(theta);
    c_psi = cos(psi); s_psi = sin(psi);
    
    L_EB = [c_theta*c_psi, s_phi*s_theta*c_psi-c_phi*s_psi, c_phi*s_theta*c_psi+s_phi*s_psi;
            c_theta*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi, c_phi*s_theta*s_psi-s_phi*c_psi;
            -s_theta, s_phi*c_theta, c_phi*c_theta];
end