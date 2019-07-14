clear all;

global fault
global scenario_flag
fault = 'Hayward';
scenario_flag = 3;

% display('Starting damage simulation')
% MAIN_DS_simulation_mult_scenarios
% display('Ending damage simulation')
% 
% clearvars -except fault scenario_flag
% display('Starting HH damage simulation')
% MAIN_DS_to_HH_simulation_mult_scenarios
% display('Ending HH damage simulation')
% 
% clearvars -except fault scenario_flag
% display('Starting loss simulation')
% MAIN_loss_simulation_mult_scenarios
% display('Ending loss simulation')
% 
% clearvars -except fault scenario_flag
% display('Starting HH loss simulation')
% MAIN_loss_to_HH_simulation_mult_scenarios
% display('Ending HH loss simulation')
% 
% clearvars -except fault scenario_flag
% addpath('./ARIO_V4.1')
% display('Starting recovery simulation')
% ARIO_version_7_1_Maryia_code_BA_sims_M7_scen_4_welfare_model
% display('Ending recovery simulation')


display('Starting empoyment simulation')
MAIN_employment_recovery_4_socioeconomic
display('Ending employment simulation')

