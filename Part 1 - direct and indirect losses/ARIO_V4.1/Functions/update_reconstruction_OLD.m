function [reconstr_demand_matrix_new,reconstr_inv] = update_reconstruction_old(reconstr_demand_matrix, reconstr_demand_sat_rate,dt)
N = size(reconstr_demand_matrix,2);

reconstr_demand = reshape(sum(reconstr_demand_matrix,2),1,1,N);
reconstr_demand_sat_rate = reshape(reconstr_demand_sat_rate ,1,1,N);

satisfied_demand_fraction = reconstr_demand_matrix./repmat(reconstr_demand,1,N,1);
reconstr_demand_matrix_new = reconstr_demand_matrix - dt.*...
    repmat(reconstr_demand_sat_rate,1,N,1).*satisfied_demand_fraction;


%reconstr_demand_matrix_new(reconstr_demand_matrix == 0  ) = 0;
reconstr_demand_matrix_new = max(0,reconstr_demand_matrix_new);

reconstr_inv = max(0,sum(reconstr_demand_matrix,3) - sum(reconstr_demand_matrix_new,3));

end

