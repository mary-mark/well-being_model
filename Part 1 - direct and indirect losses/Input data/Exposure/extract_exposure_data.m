% Load Exposure data
close all; clear all;

%Load and organize the header
fid      = fopen('BayArea_BldgCount_byBTandOcc.csv');
headers = textscan(fid,'%s',1,'HeaderLines',0);
fclose(fid);
headers = strsplit(headers{1,1}{1},',');
BayArea.bldg_id =cellfun(@(x) x(2:end-1) ,headers(3:end),'UniformOutput',false);

%Temp fix --> FIX THIS IN THE FUTURE
for i = 1:length(BayArea.bldg_id)
    
    temp_cell =  strsplit( BayArea.bldg_id{i},'_');
    BayArea.bldg_strType{i} = temp_cell{2}(1:end-1);
    BayArea.bldg_occType{i} = temp_cell{1}(4:7);
    BayArea.bldg_designLevel{i} = temp_cell{3};
    if BayArea.bldg_designLevel{i} == 'HS'
        BayArea.bldg_designLevel{i} = 'HC';
    elseif BayArea.bldg_designLevel{i} == 'LS'
        BayArea.bldg_designLevel{i} = 'LC';
    end
end

% Load the actual exposure data
raw_data = csvread('BayArea_BldgCount_byBTandOcc.csv',1,1);
BayArea.CensusTract = raw_data(:,1);
BayArea.bldg_num = raw_data(:,2:end);
% Load the dollar per bldg exposure data
raw_data = csvread('BayArea_2016DollarExposure_perBTandOcc.csv',1,1);
BayArea.RC_per_bldg = raw_data(:,2:end);

save('BayArea_exposure.mat','BayArea')



