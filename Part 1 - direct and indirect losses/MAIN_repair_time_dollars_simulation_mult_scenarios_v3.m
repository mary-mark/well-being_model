clear all; close all; clc;
addpath('./Input data/DS tables')
addpath('./Functions')
addpath('./Input data/Hazard')
load Recovery_tables.mat
load Loss_tables.mat

%%
load('./Input data/Exposure/BayArea_exposure.mat')
fault = 'Hayward';
scenario_flag = 3
suffix = '_no_code'; %options: '_baseline','_retrofit','_no_code'
            %          ''

load( sprintf('mult_scenenarios_%s_500sims.mat',fault),'SCENARIOS','SCENARIOS_descrip')

% to delete
bldg_occType = BayArea.bldg_occType;
bldg_num = BayArea.bldg_num;
nlocs   = size(bldg_num,1);
noccs = length(Recovery_tables.occCode);

RC_per_bldg = BayArea.RC_per_bldg;
BA_RC_byOcc =  zeros(1,noccs);
for i = 1:length(bldg_occType)
    idx_occType=find(ismember(Loss_tables.occCode,bldg_occType(i))==1);
    for j = 1:nlocs
        BA_RC_byOcc (1,idx_occType) = BA_RC_byOcc (1,idx_occType)+ RC_per_bldg(j,i)*bldg_num(j,i);
    end
end
clear RC_per_bldg

smoothing_factor = 365/3;
time = 1:1:960+round(smoothing_factor);
rng(1)

%% Choose scenarios based on the scenario_flag
SCENARIOS_descrip = SCENARIOS_descrip(scenario_flag,:);
SCENARIOS = SCENARIOS(scenario_flag);



%%
for scen = 1:(length(SCENARIOS))
    scen
    input_file =  sprintf('Output/Damage states/DS_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),suffix);
    load(input_file)
    display('Loading is complete')

    nsims = size(DS_reals_agg_occ{1},1);
    nbldgs = size(DS_reals_agg_occ{1},2);
    nDS = 5;
    %losses_byOcc = zeros(nsims,noccs);
    %  nbldgs_damaged = zeros(nsims,15);
    % nbldgs_damaged_total = zeros(nsims);
    recovery_curve_ind = zeros(length(time),15,nsims);
    smoothed_curve_ind = zeros(length(time),15,nsims);
    recovery_curve_total = zeros(length(time),nsims);
    smoothed_curve_total = zeros(length(time),nsims);
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
    
    for sim =1:nsims
        sim
        dollars_recovered = zeros(noccs,length(time));
        
        for i = 1:length(agg_occ)
            idx_occType=find(ismember(Recovery_tables.occCode,agg_occ(i))==1);
            % to fix
            recovery_time = Recovery_tables.recoveryTimes(idx_occType,:);
            loss_fraction = (Loss_tables.lossStr(idx_occType,:)...
                + Loss_tables.lossAccNS(idx_occType,:)...
                + Loss_tables.lossDriftNS(idx_occType,:))/100;
            non_loss_fraction = 1-loss_fraction;
            for j = 1:nlocs
                %losses_byOcc(:,idx_occType) = losses_byOcc(:,idx_occType)+loss_realization';
                loss = DS_reals_agg_occ{j}(sim,5*(i-1)+1:5*i).*loss_fraction*RC_per_occ(j,i);
                recovery_realization = recovery_time+1;%*bldg_num(j,i);
                for ds = 1:nDS
                    dollars_recovered(idx_occType,recovery_realization(ds)) = dollars_recovered(idx_occType,recovery_realization(ds)) + loss(ds);
                end
                % non damaged portion
                dollars_recovered(idx_occType,1) = dollars_recovered(idx_occType,1) + sum(non_loss_fraction.*DS_reals_agg_occ{j}(sim,5*(i-1)+1:5*i)*RC_per_occ(j,i));
            end
        end
        
        [dollars_recovered_ind,RC_by_ind,ind_code] = map_hazus2ind_recovery(cumsum(dollars_recovered,2)',BA_RC_byOcc);
        recovery_curve_ind(:,:,sim) = (dollars_recovered_ind./repmat(RC_by_ind,length(time),1));
        recovery_curve_ind(isnan(recovery_curve_ind)) = 1;
        for ind = 1:15
            smoothed_curve_ind(:,ind,sim) = smooth(recovery_curve_ind(:,ind,sim),smoothing_factor,'moving');
            
            recovery_ind_95_threshold = 0.95+smoothed_curve_ind(1,ind,sim)*0.05;
            time_index = find(smoothed_curve_ind(:,ind,sim)>=recovery_ind_95_threshold,1,'first');
         %   time_index_1 = find(smoothed_curve_ind(:,ind,sim)>=0.99999,1,'first');
            
            if isempty(time_index)
                time_95_recovered(ind,sim) = 1;
            else
                time_95_recovered(ind,sim) = time_index;
            end
            
          %  if isempty(time_index_1)
          %      time_100_recovered(ind,sim) = 1;
          %  else
          %      time_100_recovered(ind,sim) = time_index_1;
          %  end
        end
        % nbldgs_damaged(sim,:) = bldg_num_by_ind - nbldgs_recovered_ind(1,:);
        if sim == 1
            norm_RC_byInd = RC_by_ind./sum(RC_by_ind);
        end
        
    end
    
    
    %%
    for sim =1:nsims
        sim
        recovery_curve_total(:,sim) = recovery_curve_ind(:,:,sim) * norm_RC_byInd';
        smoothed_curve_total(:,sim) = smooth( recovery_curve_total(:,sim),smoothing_factor,'moving');
    end
    
    %%
    recovery_curve_ind = recovery_curve_ind(1:7:end,:,:);
    recovery_curve_total = recovery_curve_total(1:7:end,:);
    smoothed_curve_ind = smoothed_curve_ind(1:7:end,:,:);
    smoothed_curve_total = smoothed_curve_total(1:7:end,:);
    
    output_file = sprintf('Output/Recovery/Industry/loss_recovery_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),suffix);
    save(output_file, 'time_95_recovered','recovery_curve_ind', 'recovery_curve_total','smoothed_curve_ind', 'smoothed_curve_total','RC_by_ind')
    %%
    t_recovery_output_file = sprintf('Output/Recovery/Industry/t_95_ind_recovery_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),suffix);
    save(t_recovery_output_file, 'time_95_recovered')
end