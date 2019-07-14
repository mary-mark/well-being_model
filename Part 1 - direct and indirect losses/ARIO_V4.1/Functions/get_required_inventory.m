function [ S_required] = get_required_inventory(production,IO_norm, n_ij  )
% Inputs: n_ij in years
% Outputs:  S_target target investories
N = length(production);
S_required = repmat(production,N,1).*IO_norm.*repmat(n_ij',1,N);

end

