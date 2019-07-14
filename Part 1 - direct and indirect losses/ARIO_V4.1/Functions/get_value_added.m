function [ VA, Inter_purchases, Inter_sales ] =  get_value_added(production,imports, IO_norm,price)
%UNTITLED4 Summary of this function goes here
% Inter_purchases(i) = sum of purchase from sector i
 N = length(production);
VA = production - imports - sum(IO_norm.* repmat(production,N,1),1);
Inter_sales = sum(IO_norm.* repmat(production,N,1),1)';
Inter_purchases = sum(IO_norm.* repmat(production,N,1).* repmat(price',1,N));
end

