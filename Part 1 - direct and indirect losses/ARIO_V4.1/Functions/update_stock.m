function [ Stock_new ] = update_stock( Stock, Order,IO_norm, production, dt,epsilon)
 N = length(production);
% Stock dynamics (Stock(i,j) = stock of goods i own by sector j)
Stock_new = Stock + dt*(Order-IO_norm.* repmat(production,N,1));
Stock_new = max(epsilon,Stock_new);
end

