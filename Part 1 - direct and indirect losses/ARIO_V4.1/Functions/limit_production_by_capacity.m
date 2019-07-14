function [ prod_lim_by_cap,prod_cap ] = limit_production_by_capacity( product_cap_ratio, alpha_prod, product_pre_eq, production)
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
prod_cap = max(0,alpha_prod.*product_pre_eq'.*product_cap_ratio);

% if production without constraint is larger than max
% production, then production equals max prod
prod_lim_by_cap = min(prod_cap',production);

end

