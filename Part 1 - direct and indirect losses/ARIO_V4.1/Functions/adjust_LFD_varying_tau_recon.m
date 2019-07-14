function [Local_demand_new,Exports_new,Reconstr_demand,Total_final_demand,Order_new] =...
    adjust_LFD_varying_tau_recon(M, Local_demand_pre_eq, Order, Exports,Reconstr_demand_mat,tau_recon)
% M: is the Macroseismic effect

% reduction in local demand by macro_effect
Local_demand_new = M*Local_demand_pre_eq';


% unchanged exports
Exports_new = Exports';

% demand for reconstruction by households does not depend on budget
% IMPORTANT (i.e., perfect access to credit for reconstruction)

% Loop over all the industries
n_ind = size(Reconstr_demand_mat,2);
Reconstr_demand = zeros(1,n_ind);
for ind = 1:n_ind
%this was for v7.3
% Reconstr_demand_rate = reshape(Reconstr_demand_mat(1,ind,:),1,n_ind)/tau_recon(ind) + Reconstr_demand_rate;
% Reconstruction demand to sector j (should be only for construction and
% manufacturing)
Reconstr_demand = Reconstr_demand_mat(ind,:,1)/tau_recon(ind) + Reconstr_demand;

end
%Reconstr_demand = Reconstr_demand/tau_recon; %????????????????


% Order(i,j) = order of j to i
Order_new = sum(Order,2)';

% Demand_total = sum of all demands
Total_final_demand= Exports_new + Local_demand_new + Reconstr_demand + Order_new;
Total_final_demand(Total_final_demand == 0) = 1e-6;

end

