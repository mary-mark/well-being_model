function [ alpha_new ] = get_overproduction_capacity(alpha, alpha_max, tau_alpha, scarcity_index,epsilon,dt )
% Inputs:   product_ratio: ratio of production to total demand
%           epsilon: used to estimate what is full recovery (in adaptation process)

for i = 1:length(alpha_max)
    if scarcity_index(i) > epsilon
        alpha_new(i) = alpha(i) + (alpha_max(i)-alpha(i))* scarcity_index(i)*dt/tau_alpha;
    else
        alpha_new(i) = alpha(i) + (1-alpha(i))*dt/tau_alpha;
    end    
end
alpha_new = alpha_new';
end

