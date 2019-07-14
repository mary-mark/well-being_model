function [reconstr_demand_matrix_new,reconstr_inv] = update_reconstruction_v2(reconstr_demand_matrix, reconstr_demand_act, reconstr_demand,dt,tau_recon)
N = size(reconstr_demand_matrix,2);

RDM_ratio = reconstr_demand_matrix./repmat(tau_recon,1,N)./repmat(reconstr_demand,N,1);
RDM_ratio(isnan(RDM_ratio))=0;
reconstr_demand_matrix_new = reconstr_demand_matrix - dt.*...
    RDM_ratio.*repmat(reconstr_demand_act,N,1);


reconstr_demand_matrix_new(reconstr_demand_matrix_new < 0  ) = 0;
%reconstr_demand_matrix_new = max(0,reconstr_demand_matrix_new);

reconstr_inv = max(0,sum(reconstr_demand_matrix,2) - sum(reconstr_demand_matrix_new,2));

end

