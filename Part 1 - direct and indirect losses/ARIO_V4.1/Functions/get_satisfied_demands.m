function [ reconstr_demand_sat, local_demand_sat, exports_sat,...
    order_sat,final_demand_unsatisfied] = ...
    get_satisfied_demands(reconstr_demand, exports, local_demand, orders, production, demand )
    % new rationing scheme: full proportional (final demand and interindustry demands)
N = length(production);
satisfied_ratio = production./demand;
reconstr_demand_sat = reconstr_demand.*satisfied_ratio;
local_demand_sat = local_demand.*satisfied_ratio;
exports_sat = exports.*satisfied_ratio;
order_sat =orders.* repmat(satisfied_ratio',1,N);
final_demand_unsatisfied= production-demand;

end

