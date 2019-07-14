function [ Order_new ] = update_orders( Stock, Stock_target,tau_stock,Order,IO_norm, production, dt,epsilon)
 N = length(production);
% Stock dynamics (Stock(i,j) = stock of goods i own by sector j)
Order_new = IO_norm.* repmat(production,N,1)+ dt/tau_stock*(;
Order_new = max(epsilon,Order_new);
end

