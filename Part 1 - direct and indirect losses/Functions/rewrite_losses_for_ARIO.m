function rewrite_losses_for_ARIO( filepath,output_filename)
%'../Output/BayArea_agg_losses_SC119.mat'
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
load(filepath)
nsims = size(direct_losses_ind,1);
BayArea_damages_mean = zeros(15,15);
for i = 1:nsims
BayArea_damages_sims{i} = zeros(15,15);
BayArea_damages_sims{i}(:,4) = direct_losses_ind(i,:)'* 0.8/1e6;
BayArea_damages_sims{i}(:,5) = direct_losses_ind(i,:)' * 0.2/1e6;
BayArea_damages_mean = BayArea_damages_mean + BayArea_damages_sims{i};
end
BayArea_damages_mean = BayArea_damages_mean/nsims;
save(output_filename,'BayArea_damages_sims','BayArea_damages_mean','frac_loss_prod')
%save('./ARIO_V4.1/BayArea_damages_mean.mat','BayArea_damages_mean')
end

