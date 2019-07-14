function [ alpha_new ] = get_overproduction_capacity(alpha, alpha_max, tau_alpha, product_cap_ratio,epsilon,dt )
% Inputs:   product_ratio: ratio of production to total demand
%           epsilon: used to estimate what is full recovery (in adaptation process)
scarcity_index = 1-product_cap_ratio;
for i = 1:length(alpha_max)
    if product_cap_ratio(i) < (1-epsilon)
        alpha_new(i) = alpha(i) + (alpha_max(i)-alpha(i))* scarcity_index(i)*dt/tau_alpha;
    else
        alpha_new(i) = alpha(i) + (1-alpha(i))*dt/tau_alpha;
    end    
end
alpha_new = alpha_new';
end

