function [SCENARIOS_descrip,SCENARIOS] = select_M7_scenario(fault,SCENARIOS_descrip,SCENARIOS)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
switch fault
    case 'Hayward'
        SCENARIOS_descrip = SCENARIOS_descrip(3,:);
        SCENARIOS = SCENARIOS(3);
    case 'Calaveras'
        SCENARIOS_descrip = SCENARIOS_descrip(3,:);
        SCENARIOS = SCENARIOS(3);
    case 'SanAndreas'
        SCENARIOS_descrip = SCENARIOS_descrip(1,:);
        SCENARIOS = SCENARIOS(1);
    otherwise
        error('FAULT DOESNT EXIST')
end
end

