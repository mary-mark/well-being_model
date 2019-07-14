% Load HAZUS parameters from spreadsheets and save to a data structure for
% later analysis
%
% Jack Baker
% June 13, 2016

clear; close all; clc;

filename = 'HAZUS data.xlsx';
probSheet = 1; % sheet number with probabilities of damage
lossSheet = 2; % sheet number with loss ratios
filename_recovery = 'HAZUS recovery times.xlsx';
recovery_timeSheet = 1;


%% read some building type labels

[~,hazusData.buildingTypeCode] = xlsread(filename, 3, 'C11:C46');
[~,hazusData.buildingTypeDesc] = xlsread(filename, 3, 'D11:D46');
[~,hazusData.buildingNumStoriesDesc] = xlsread(filename, 3, 'E11:E46');
[~,hazusData.buildingNumStoriesNumeric] = xlsread(filename, 3, 'F11:F46');
[hazusData.buildingTypStories] = xlsread(filename, 3, 'G11:G46');
[hazusData.buildingTypHeight] = xlsread(filename, 3, 'H11:H46');





%% read probability data

% high code data
[numHC,txt] = xlsread(filename, probSheet, 'A11:I46');

% moderate code data
[numMC,txt] = xlsread(filename, probSheet, 'B56:I91');

% low code data
[numLC] = xlsread(filename, probSheet, 'B99:I134');

% pre code data
[numPC] = xlsread(filename, probSheet, 'B145:I180');

%% format data structures
hazusData.codeLevel = {'High', 'Moderate', 'Low', 'Pre'};

% extract medians per damage state
idx = [1 3 5 7];
hazusData.medians{1} = numHC(:,idx);
hazusData.medians{2} = numMC(:,idx);
hazusData.medians{3} = numLC(:,idx);
hazusData.medians{4} = numPC(:,idx);

% extract medians per damage state
idx = [2 4 6 8];
hazusData.betas{1} = numHC(:,idx);
hazusData.betas{2} = numMC(:,idx);
hazusData.betas{3} = numLC(:,idx);
hazusData.betas{4} = numPC(:,idx);

%% read loss ratio data
[~,hazusData.occCode] = xlsread(filename, lossSheet, 'C9:C36');
[~,hazusData.occLabel] = xlsread(filename, lossSheet, 'D9:D36');

hazusData.lossStruct = xlsread(filename, lossSheet, 'E9:H36');
hazusData.lossAccNS = xlsread(filename, lossSheet, 'E42:H69');
hazusData.lossDriftNS = xlsread(filename, lossSheet, 'E75:H102');

%% read recovery data
hazusData.recoveryTimes = xlsread(filename_recovery, recovery_timeSheet, 'E9:H36');

save hazusData hazusData
