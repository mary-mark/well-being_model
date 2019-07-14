function [ recovery_curve_ind,total_bldgs_ind  ind_code] = map_hazus2ind_recovery(nbldgs_recovered_hazus,total_bldgs_hazus )
load './Input data/Hazus2ind/hazus2ind_V2_HayWired.mat'
%loss_hazus = losses_byOcc;
recovery_curve_ind= nbldgs_recovered_hazus(:,:) * hazus2ind.mapping_scheme;
total_bldgs_ind = total_bldgs_hazus * hazus2ind.mapping_scheme;
ind_code = hazus2ind.ind_code;

% figure
% n_time = size(recovery_curve_ind,1);
% plot(1:n_time, recovery_curve_ind(:,:)')
% ylabel('recovery')
% xlabel('time')

end

