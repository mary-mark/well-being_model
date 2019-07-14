
%%
clear all; close all; clc;
addpath('./Input data/DS tables')
addpath('./Functions')
addpath('./Input data/Hazard')
load Loss_tables.mat


fault = 'Hayward'; %'SanAndreas';
scanario_flags = [3];
for scenario_flag= scanario_flags 

input_suffix = '_baseline';
output_suffix = '_ins40_15';

%rng(1)

%%%%%%%  INSIRANCE PARAMETERS%%%%%%%%%%%

% Baseline insurance parameters 
% ins_penetration = .15;
% ins_deduct = 0.15;

ins_penetration = 0.40;
ins_deduct = 0.15;

load( sprintf('mult_scenenarios_%s_500sims.mat',fault),'SCENARIOS','SCENARIOS_descrip')
load('./Input data/Exposure/BayArea_exposure.mat')



%%
for scen = scenario_flag%1%:(length(SCENARIOS))
    scen
    input_file =  sprintf('Output/Damage states/DS_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),input_suffix);
    load(input_file)
    output_file = sprintf('Output/Losses/Loss_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),output_suffix);
    
    display('Loading files is complete')
    %%
    nlocs = length(DS_reals_agg_occ);
    nsims = size(DS_reals_agg_occ{1},1);
    n_DS = 5;
    n_agg_occ = size(DS_reals_agg_occ{1},2)/n_DS;
    noccs = length(Loss_tables.occCode);
    losses_byOcc_HH_owned = zeros(nsims,noccs);
        census_losses_HH = zeros(nsims,nlocs);
        census_losses_HH_insured = zeros(nsims,nlocs);
    
    % losses_all = cell(1,nlocs);
    HH_losses_perBldg_perOcc_perDS = zeros(nlocs,n_agg_occ*n_DS,nsims);
    %HH_losses_perOcc_perDS = zeros(nlocs,n_agg_occ*n_DS,nsims);
    HH_losses_perDS = zeros(nlocs,n_DS,nsims);
    losses_perBldg_perOcc_perDS = zeros(nsims,n_agg_occ*n_DS,noccs);
    %losses_perOcc_perDS = zeros(nsims,noccs,n_agg_occ*n_DS);
    losses_perDS = zeros(nsims,n_DS,noccs);
    
    for i = 1:length(agg_occ)
        if ismember(agg_occ{i}, {'RES3A','RES3B','RES3C','RES3D','RES3E','RES3F'} )
            agg_occ{i}
            agg_occ{i} = 'RES3';
        end
    end
    
    for i = 1:length(agg_occ)
        i
        target_occ = agg_occ{i}(1:4)
        idx_occType=find(ismember(Loss_tables.occCode,target_occ)==1);
        % to fix
        loss_fraction = (Loss_tables.lossStr(idx_occType,:)...
            + Loss_tables.lossAccNS(idx_occType,:)...
            + Loss_tables.lossDriftNS(idx_occType,:))/100;
        
        for j = 1:nlocs
            for k = 1:nsims
                
                 if ismember(idx_occType,[1,2,3])
                     
                HH_losses_perBldg_perOcc_perDS(j,5*(i-1)+1:5*i,k) = loss_fraction*RC_per_occ(j,i);
                %HH_losses_perOcc_perDS(j,5*(i-1)+1:5*i,k) = HH_losses_perBldg_perOcc_perDS(j,5*(i-1)+1:5*i,k)...
                %    .*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i);
                 HH_losses_perDS(j,:,k) = HH_losses_perDS(j,:,k)+HH_losses_perBldg_perOcc_perDS(j,5*(i-1)+1:5*i,k)...
                    .*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i);
                
                losses_byOcc_HH_owned(k,idx_occType) = losses_byOcc_HH_owned(k,idx_occType)...
                    +sum(HH_losses_perBldg_perOcc_perDS(j,5*(i-1)+1:5*i,k).*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i))...
                    *(1-BayArea.renters(j));
                
                % CALCULATE INSURANCE PAID FOR EACH CENSUS TRACT
                non_ins_loss = 0;
                total_loss = 0;
                ins_loss_perDS = 0;

                total_loss =  sum(HH_losses_perBldg_perOcc_perDS(j,5*(i-1)+1:5*i,k).*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i));
                non_ins_loss =  sum(HH_losses_perBldg_perOcc_perDS(j,5*(i-1)+1:5*i,k).*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i))*(1-ins_penetration);
                ins_loss_perDS = HH_losses_perBldg_perOcc_perDS(j,5*(i-1)+1:5*i,k).*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i)*ins_penetration;
               % temp2 = non_ins_loss +sum(ins_loss_perDS)
                deduct_perDS = RC_per_occ(j,i)*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i)*ins_penetration*ins_deduct;
                ins_loss = 0;
                for ds = 1:n_DS
                ins_loss = ins_loss+min(ins_loss_perDS(ds),deduct_perDS(ds));
                end
                % temp3 = non_ins_loss +ins_loss
                census_losses_HH_insured (k,j) = census_losses_HH_insured(k,j)+ins_loss+non_ins_loss;
                census_losses_HH (k,j) = census_losses_HH(k,j)+total_loss;
            
                 end
         
                losses_perBldg_perOcc_perDS(k,5*(i-1)+1:5*i,idx_occType) = loss_fraction*RC_per_occ(j,i);
                %losses_perOcc_perDS(k,idx_occType,5*(i-1)+1:5*i) = losses_perBldg_perOcc_perDS(k,idx_occType,5*(i-1)+1:5*i)...
                %    .*reshape(DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i),1,1,n_DS);
                
                losses_perDS(k,:,idx_occType) = losses_perDS(k,:,idx_occType)+losses_perBldg_perOcc_perDS(k,5*(i-1)+1:5*i,idx_occType)...
                    .*DS_reals_agg_occ{j}(k,5*(i-1)+1:5*i);
                 
                
                
            end
        end
    end
    HH_losses_total = sum(HH_losses_perDS,2);
    losses_byOcc = reshape(sum(losses_perDS,2),nsims,28);
%%
    for sim = 1:nsims
        for loc = 1:nlocs
        HH_losses_total_2(sim,loc) = HH_losses_total(loc,:,sim);
        end
    end

    [direct_losses_ind, ind_codes,frac_loss_prod] = map_hazus2ind_V2_HayWired(losses_byOcc,losses_byOcc_HH_owned);
    direct_losses_hazus = losses_byOcc;
    hazus_codes = Loss_tables.occCode;
    %%
    save(output_file,  'losses_byOcc', 'direct_losses_hazus', 'hazus_codes', 'direct_losses_ind', 'ind_codes','frac_loss_prod','census_losses_HH','census_losses_HH_insured')
    ARIO_output_filename =  sprintf('./Output/For ARIO/Losses_4ARIO_%s_sims_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),output_suffix);
    filepath = sprintf('./Output/Losses/Loss_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),output_suffix);
    rewrite_losses_for_ARIO(filepath,ARIO_output_filename)

end

%% Save csv files
if 1
    output_dir = sprintf('Output/Losses/Loss_HH_%s_sc%i_%i_DEC2018%s',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),output_suffix);
    if exist(output_dir,'dir')
    else
        mkdir(output_dir)
    end
    
    
%     headers = {'Full_tract'};
%     count = 1;
%     sub_header = {'n_dmg','loss_perBldg','loss_total'};
%     for k = 1:3
%         for i = 1:n_agg_occ
%             for   j = 1:n_DS
%                 count = count +1;
%                 headers{count} = sprintf('%s_%s_DS%i',sub_header{k},HH_agg_occ{i},j);
%             end
%         end
%     end
    headers2 = {'tract','HH_loss_DS1','HH_loss_DS2','HH_loss_DS3','HH_loss_DS4',...
        'HH_loss_DS5','HH_loss_total','HH_loss_insured'}%s_%s_DS%i',sub_header{k},HH_agg_occ{i},j);
        
    
    
%     % Rewrite DS into accepable format
%     for i = 1:nlocs
%         for j = 1:n_agg_occ*n_DS
%             for k = 1:nsims
%                 DS_perOcc_perDS(i,j,k) = DS_reals_agg_occ{i}(k,j);
%             end
%         end
%     end
%     
    for sim = 1:nsims
        sim

        filename2 = sprintf('%s/Loss_HH_%s_sc%i_%i_sim%i_agg.csv',...
            output_dir,fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),sim);
         data2 = [BayArea.CensusTract,HH_losses_perDS(:,:,sim),census_losses_HH(sim,:)',census_losses_HH_insured(sim,:)'];
        csvwrite_with_headers(filename2,data2,headers2)
        
        %
    end
end
end