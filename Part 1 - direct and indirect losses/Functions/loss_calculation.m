%function [ output_args ] = loss_calculation( occ_names, ds_data )
clear all; close all; clc;
addpath('../Input data/DS tables')
load Loss_tables.mat
load('../Output/DS_results_02152018.mat')
display('Loading files is complete')
nlocs = length(DS_reals);
nsims = size(DS_reals{1},1);
nbldgs = size(DS_reals{1},2);
noccs = length(Loss_tables.occCode);
losses_all = cell(1,nlocs);
losses_byOcc = zeros(nsims,noccs);
total_losses = zeros(nsims,nbldgs);
rng(1)
% to delete
bldg_occType = BayArea.bldg_occType;
bldg_num = BayArea.bldg_num;
RC_per_bldg = BayArea.RC_per_bldg; 
for j = 1:nlocs
    j
    for i = 1:length(bldg_occType)
        idx_occType=find(ismember(Loss_tables.occCode,bldg_occType(i))==1);
        % to fix
        loss_fraction = (Loss_tables.lossStr(idx_occType,:)...
            + Loss_tables.lossAccNS(idx_occType,:)...
            + Loss_tables.lossDriftNS(idx_occType,:))/100;
        loss_realization = loss_fraction(DS_reals{j}(:,i))*RC_per_bldg(j,i)*bldg_num(j,i);
        losses_byOcc(:,idx_occType) = losses_byOcc(:,idx_occType)+loss_realization';
        total_losses(:,i) =  total_losses(:,i)+loss_realization';
        losses_all{j}(:,i) =  loss_realization';
    end 
end
%end
%%
save '../Output/BayArea_losses_sc119.mat' 'total_losses' 'losses_all' 'losses_byOcc' -v7.3
