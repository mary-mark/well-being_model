function [ prod_lim_by_sup, constrain_idx] = limit_production_by_supplies(Psi,Stock,Stock_required, production)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
N = length(Psi);
Psi_mat = repmat(Psi',1,N);
product_ratio = Stock./(Psi_mat.*Stock_required);
if any(product_ratio < 1)
product_ratio;
end
product_constraint = repmat(production',1,N); % how much industry i can produce constrained on j
for i = 1:length(production)
   for j = 1:N
    if Stock(j,i)< Psi(j)*Stock_required(j,i)
        product_constraint(i,j) = product_constraint(i,j)*min(1,product_ratio(j,i));
    end
   end
   [prod_lim_by_sup(i), constrain_idx(i)] = min(product_constraint(i,:));
    if prod_lim_by_sup(i) == production(i)
        constrain_idx(i) = 16;
    
    end
end



end

