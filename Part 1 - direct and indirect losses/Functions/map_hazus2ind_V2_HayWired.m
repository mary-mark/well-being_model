function [ direct_loss_ind, ind_code,frac_loss_prod] = map_hazus2ind_V2_HayWired(loss_hazus,loss_HH_owned)
load './Input data/Hazus2ind/hazus2ind_V2_HayWired.mat'
%loss_hazus = losses_byOcc;
% nsims = size(loss_hazus,1);
% n_occ = size(loss_hazus,2);
% n_ind = length(hazus2ind.ind_code);

direct_loss_ind = loss_hazus * hazus2ind.mapping_scheme;
ind_code = hazus2ind.ind_code;
total_HH_owned = sum(loss_HH_owned,2);

frac_loss_prod = ones(size(direct_loss_ind));
frac_loss_prod(:,10) =1- total_HH_owned./direct_loss_ind(:,10);
% figure
% boxplot(direct_loss_ind/1e9)
% ylabel('Direct losses (billion USD)')
% xlabel('Industry')
% set(gca,'XTickLabel',hazus2ind.ind_code)
% set(gca,'XTickLabelRotation',45)
% ylim([0 90])
end

