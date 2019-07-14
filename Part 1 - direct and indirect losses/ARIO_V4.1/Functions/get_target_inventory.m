function [ S_target] = get_target_inventory(total_demand,IO_norm, n_ij,prod_cap)
% Inputs: n_ij in years
% Outputs:  S_target target investories
N = length(prod_cap);
prod_lim_by_cap = min(prod_cap', total_demand);
S_target= repmat(prod_lim_by_cap,N,1).*IO_norm.*repmat(n_ij',1,N);

end

