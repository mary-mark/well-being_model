% Create HAZUS probability of damage state tables
clear all; close all;
if 1
    import_HAZUS_data
    save('hazusData.mat','hazusData')
else
    load hazusData
end
%%
PGA = 0.001:0.001:6;
n_bldgTypes = length(hazusData.buildingTypeCode);
n_designLevels = length(hazusData.codeLevel);
n_PGA = length(PGA);
n_DS = 4;
PGA_mat = repmat(PGA',1,n_DS);
DS_tables.PGA = PGA;
DS_tables.bldgTypeCode = hazusData.buildingTypeCode;
DS_tables.bldgTypeDesc = hazusData.buildingTypeDesc;
DS_tables.designLevel = {'HC','MC','LC','PC'};

for i = 1:n_bldgTypes
    %Pds_tables{i,j} = zeros(n_PGA,n_DS,n_designLevels);
    for j = 1:n_designLevels
        medians = repmat(hazusData.medians{j}(i,:),n_PGA,1);
        betas = repmat(hazusData.betas{j}(i,:),n_PGA,1);
        Pexc_ds = normcdf(log(PGA_mat./medians)./betas);
        Pexc_ds_1 = [ones(n_PGA,1),Pexc_ds];
        Pexc_ds_2 = [Pexc_ds,zeros(n_PGA,1)];
        DS_tables.Pds{i,j}  = Pexc_ds_1 - Pexc_ds_2;
        DS_tables.Pexc_ds{i,j}  = Pexc_ds;
    end
end
Loss_tables.occCode = hazusData.occCode;
Loss_tables.occLabel = hazusData.occLabel;
Loss_tables.lossStr = [zeros(size(hazusData.lossStruct,1),1),hazusData.lossStruct];
Loss_tables.lossAccNS = [zeros(size(hazusData.lossAccNS,1),1),hazusData.lossAccNS];
Loss_tables.lossDriftNS = [zeros(size(hazusData.lossDriftNS,1),1),hazusData.lossDriftNS];
Recovery_tables.occCode = hazusData.occCode;
Recovery_tables.occLabel = hazusData.occLabel;
Recovery_tables.recoveryTimes = [zeros(size(hazusData.recoveryTimes,1),1),hazusData.recoveryTimes];
save('DS_tables.mat','DS_tables')
save('Loss_tables.mat','Loss_tables')
save('Recovery_tables.mat','Recovery_tables')