function [Local_demand_new,Exports_new,Reconstr_demand_rate,Total_final_demand,Order_new] =...
    adjust_LFD(M, Local_demand_pre_eq, Order, Exports,Reconstr_demand_mat,tau_recon)
% M: is the Macroseismic effect

% reduction in local demand by macro_effect
Local_demand_new = M*Local_demand_pre_eq';


% unchanged exports
Exports_new = Exports';

% demand for reconstruction by households does not depend on budget
% IMPORTANT (i.e., perfect access to credit for reconstruction)
Reconstr_demand_rate = reshape(sum(Reconstr_demand_mat,2),1,length(Exports))/tau_recon;
%Reconstr_demand = Reconstr_demand/tau_recon; %????????????????


% Order(i,j) = order of j to i
Order_new = sum(Order,2)';

% Demand_total = sum of all demands
Total_final_demand= Exports_new + Local_demand_new + Reconstr_demand_rate + Order_new;
Total_final_demand(Total_final_demand == 0) = 1e-6;

end

