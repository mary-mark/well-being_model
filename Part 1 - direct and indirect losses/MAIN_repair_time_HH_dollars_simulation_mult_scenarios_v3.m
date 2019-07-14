clear all; close all; clc;
addpath('./Input data/DS tables')
addpath('./Functions')
addpath('./Input data/Hazard')
load Recovery_tables.mat
load Loss_tables.mat

%%
load('./Input data/Exposure/BayArea_exposure.mat')
fault = 'Hayward';
scenerio_flag = 3;
suffix = '_retrofit';%option: '_baseline','_no_code','_retrofit'
load( sprintf('mult_scenenarios_%s_500sims.mat',fault),'SCENARIOS','SCENARIOS_descrip')
% to delete
bldg_occType = BayArea.bldg_occType;
bldg_num = BayArea.bldg_num;
nlocs   = size(bldg_num,1);
noccs = length(Recovery_tables.occCode);

display('Loading files is complete')
sim_start = 301;
nsims_shortened = 500;
time = 1:1:365*4;%960+1;
%%
rng(1)
for scen = scenerio_flag;%:(length(SCENARIOS))
    scen
    input_file =  sprintf('Output/Damage states/DS_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),suffix);
    load(input_file)
    HH_agg_occ = {};
    count_1 = 1;
    for i = 1:length(agg_occ)
        if ismember(agg_occ{i}, {'RES3A','RES3B','RES3C','RES3D','RES3E','RES3F'} )
            agg_occ{i}
            agg_occ{i} = 'RES3';
        end
        if ismember(agg_occ{i}, {'RES1','RES2','RES3'})
            HH_agg_occ{count_1} = agg_occ{i};
            count_1 = count_1+1;
        end
        
    end
    HH_occ_idx = ismember(agg_occ, {'RES1','RES2','RES3'});
    HH_RC_per_occ = RC_per_occ(:,HH_occ_idx);
    
    HH_RC_total = sum(agg_bldg_num(:,HH_occ_idx).*RC_per_occ(:,HH_occ_idx),2);
    
    nsims = size(DS_reals_agg_occ{1},1);
    nbldgs = size(DS_reals_agg_occ{1},2);
    nDS = 5;
    
    dollars_recovered_HH = zeros(nlocs,length(time),nsims_shortened);
    smoothed_dollars_recovered_HH = zeros(nlocs,length(time),nsims_shortened);
    
    recovery_curve_HH = zeros(nlocs,length(time),nsims_shortened);
    smoothed_recovery_curve_HH = zeros(nlocs,length(time),nsims_shortened);
    time_95_recovered = zeros(nlocs,nsims_shortened);

    for sim =sim_start:nsims_shortened
        sim
        dollars_recovered = zeros(nlocs,length(time));
        
        for i = 1:length(HH_agg_occ)
            idx_occType=find(ismember(Recovery_tables.occCode,HH_agg_occ{i}(1:4))==1);
            % to fix
            recovery_time = Recovery_tables.recoveryTimes(idx_occType,:);
            loss_fraction = (Loss_tables.lossStr(idx_occType,:)...
                + Loss_tables.lossAccNS(idx_occType,:)...
                + Loss_tables.lossDriftNS(idx_occType,:))/100;
            non_loss_fraction = 1-loss_fraction;
            for j = 1:nlocs
                %losses_byOcc(:,idx_occType) = losses_byOcc(:,idx_occType)+loss_realization';
                loss = DS_reals_agg_occ{j}(sim,5*(i-1)+1:5*i).*loss_fraction*HH_RC_per_occ(j,i);
                recovery_realization = recovery_time+1;%*bldg_num(j,i);
                for ds = 1:nDS
                    dollars_recovered(j,recovery_realization(ds)) = dollars_recovered(j,recovery_realization(ds)) + loss(ds);
                end
                % non damaged portion
                dollars_recovered(j,1) = dollars_recovered(j,1) + sum(non_loss_fraction.*DS_reals_agg_occ{j}(sim,5*(i-1)+1:5*i)*HH_RC_per_occ(j,i));
                
            end
        end
        for j = 1:nlocs
            dollars_recovered(j,:) =cumsum( dollars_recovered(j,:));
            
        end
        
        recovery_curve_HH(:,:,sim) = (dollars_recovered./repmat(HH_RC_total,1,length(time)));
        dollars_recovered_HH(:,:,sim) =dollars_recovered;
        for loc = 1:nlocs
            smoothed_recovery_curve_HH(loc,:,sim) = smooth(recovery_curve_HH(loc,:,sim),365/2,'moving');
            smoothed_dollars_recovered_HH(loc,:,sim)= smooth(dollars_recovered_HH(loc,:,sim),365/2,'moving');
        end
        
        for loc = 1:nlocs
            recovery_95_threshold = 0.95+recovery_curve_HH(loc,1,sim)*0.05;
            time_index = find(smoothed_recovery_curve_HH(loc,:,sim)>=recovery_95_threshold,1,'first');
            if isempty(time_index)
                time_95_recovered(loc,sim) = 1;
            else
                time_95_recovered(loc,sim) = time_index;
            end
        end
    end
    %%
    figure
    hist(time_95_recovered(:)/365,100);
    recovery_curve_HH(isnan(recovery_curve_HH)) = 0;
    %%
    
    
    figure
    plot(time,  smoothed_recovery_curve_HH(:,:,sim)')
    %% SAVE MATLAB FILE
    if 0
    output_file = sprintf('Output/Recovery/HH_loss_recovery_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),suffix);
    save(output_file, 'smoothed_recovery_curve_HH', 'recovery_curve_HH','time','HH_RC_total')
    end
    
    %% Save csv files
    if 1
        output_dir = sprintf('Output/Recovery/HH/Recovery_HH_%s_sc%i_%i_DEC2018%s',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),suffix);
        if exist(output_dir,'dir')
        else
            mkdir(output_dir)
        end
        
        
        headers = {'tract','total_asset_value','t(days)_95_recovered'};
        count = 3;
        sub_header = {'Day'};
        for t = 1:length(time)
            count = count +1;
            headers{count} = sprintf('%s%i',sub_header{1},t);
        end
        
        
        display('Writing .csv files')
        for sim = sim_start:nsims_shortened
            sim
            filename = sprintf('%s/Recovery_HH_%s_sc%i_%i_sim%i.csv',...
                output_dir,fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),sim);
            data = [BayArea.CensusTract,HH_RC_total,time_95_recovered(:,sim),smoothed_recovery_curve_HH(:,:,sim)];
            csvwrite_with_headers(filename,data,headers)
            
            
            %
        end
    end
    
end