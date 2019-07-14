%% Write insurance penetration flags

data = csvread('./insurance_penetration_flags.csv',1,0);
ins_penetr_data.CensusTracts = data(:,1);
ins_penetr_data.poor_15 = data(:,2);
ins_penetr_data.poor_30 = data(:,3);
ins_penetr_data.poor_50 = data(:,4);
ins_penetr_data.rich_15 = data(:,5);
ins_penetr_data.rich_30 = data(:,6);
ins_penetr_data.rich_50 = data(:,7);
save('insurance_penetration.mat','ins_penetr_data')