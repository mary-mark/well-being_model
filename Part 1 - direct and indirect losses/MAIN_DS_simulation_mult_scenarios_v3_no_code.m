clear all; 
fault = 'Hayward';
scenario_flag = 3;
suffix = '_no_code';
%%
close all; clc;

addpath('./Input data/Hazard')
addpath('./Input data/Exposure')
addpath('./Input data/DS tables')
addpath('./Functions')

if 1
    % Load hazard
    load(sprintf('mult_scenenarios_%s_500sims.mat',fault))
    % Load exposure
    load BayArea_exposure
    load IM_tract_mapping
    % Load DS tables
    load DS_tables
    nsims = nsims_within;
    nlocs = length(BayArea.CensusTract);
    %nbldgs = length(BayArea.bldg_id);
    
    bldg_occType = BayArea.bldg_occType;
    bldg_num = BayArea.bldg_num;
    RC_per_bldg = BayArea.RC_per_bldg;
    
    names = {'RES3A','RES3B','RES3C','RES3D','RES3E','RES3F'};
    for i = 1:6
        for j = 0:13
        bldg_occType(14*(i-1)+8+j) = names(i);
        end
    end
    
    count_1 = 1;
    agg_occ = {};
    for i = 1:length(bldg_occType)
        if ismember({bldg_occType{i}},agg_occ )
           count_1 = count_1; 
        else
            agg_occ{count_1} = bldg_occType{i};
            agg_occ{count_1}
            count_1 = count_1+1; 
        end
    end
    
    n_agg_occ = length(agg_occ);
    nbldg_type = size(bldg_num,2);
    RC_per_occ= zeros(nlocs,n_agg_occ);
    agg_bldg_num=zeros(nlocs,n_agg_occ);
    n_DS = 5;
    rng(1)
    
%     HH_bldg_occType = cellstr(bldg_occType(1:91));% exclude dorminories and nursing homes
%     names = {'RES3A','RES3B','RES3C','RES3D','RES3E','RES3F'};
%     for i = 1:6
%         for j = 0:13
%         HH_bldg_occType(14*(i-1)+8+j) = names(i);
%         end
%     end
%     HH_bldg_num = bldg_num(:,1:91);
%     HH_RC_per_bldg = RC_per_bldg(:,1:91);
%     HH_bldg_strType = BayArea.bldg_strType(1:91);
%     nbldgs_type = size(HH_bldg_num,2);
% 
%     
%     HH_agg_occ = unique(HH_bldg_occType);
%     n_agg_occ = length(HH_agg_occ);
%     HH_RC_per_occ = zeros(nlocs,n_agg_occ);
%     HH_agg_bldg_num = zeros(nlocs,n_agg_occ);
    
    for scen = scenario_flag%:(length(SCENARIOS))
        output_file = sprintf('Output/Damage states/DS_results_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),suffix);
        IM_sims = round(IM_reals{scen},3);
        tic
        IM_sims(IM_sims(:)>6) = 6;
        toc
        for i = 1:nlocs
            %DS_reals{i} = zeros(nsims,nbldgs_type*n_DS);
            DS_reals_agg_occ{i} = zeros(nsims,n_agg_occ*n_DS);
        end
        nlocs_part = 800;
        for i = 1:nlocs%:nlocs_part;
            i
            [~,idx_PGA] = ismember(round(IM_sims(IM_tract_mapping(i),:),3),round(DS_tables.PGA,3));
            if any(idx_PGA == 0)
                error('ZERO')
            end
            for j = 1:nbldg_type
                idx_designLevel =find(ismember(DS_tables.designLevel,BayArea.bldg_designLevel{j})==1);
                idx_strType=find(ismember(DS_tables.bldgTypeCode,BayArea.bldg_strType{j})==1);
                if ismember(BayArea.bldg_occType{j},{'RES1'}) || ismember(BayArea.bldg_occType{j},{'RES3'})
                    if idx_designLevel == 1
                       idx_designLevel = 2;
                    end
                end
                
                P_ds = DS_tables.Pds{idx_strType, idx_designLevel}(idx_PGA,:);
                
                n_bldgs_to_sim = round(bldg_num(i,j));
                if n_bldgs_to_sim == 0
                    n_bldgs_to_sim = 1;
                end
                ds_reals = mnrnd(n_bldgs_to_sim,P_ds);
                ds_reals = ds_reals/n_bldgs_to_sim * bldg_num(i,j);
                [~,occ_idx] = ismember(bldg_occType{j},agg_occ);
                DS_reals_agg_occ{i}(:,5*(occ_idx-1)+1:5*occ_idx) = ...
                    DS_reals_agg_occ{i}(:,5*(occ_idx-1)+1:5*occ_idx) + ds_reals;
                RC_per_occ(i,occ_idx) = RC_per_bldg(i,j);
                agg_bldg_num(i,occ_idx) = agg_bldg_num(i,occ_idx) + bldg_num(i,j);
               
            end      
        end
        
%         parfor i = nlocs_part+1:nlocs;
%             i
%             [~,idx_PGA] = ismember(round(IM_sims(i,:),3),round(DS_tables.PGA,3));
%             if any(idx_PGA == 0)
%                 error('ZERO')
%             end
%             for j = 1:nbldgs_type;
%                 idx_designLevel =find(ismember(DS_tables.designLevel,BayArea.bldg_designLevel{j})==1);
%                 idx_strType=find(ismember(DS_tables.bldgTypeCode,BayArea.bldg_strType{j})==1);
%                 P_ds = DS_tables.Pds{idx_strType, idx_designLevel}(idx_PGA,:);
%                 
%                 n_bldgs_to_sim = round(HH_bldg_num(i,j));
%                 if n_bldgs_to_sim == 0
%                     n_bldgs_to_sim = 1;
%                 end
%                 ds_reals = mnrnd(n_bldgs_to_sim,P_ds);
%                 ds_reals = ds_reals/n_bldgs_to_sim * HH_bldg_num(i,j);
%                % DS_reals{i}(:,5*(j-1)+1:5*j) = ds_reals;
%                 [~,occ_idx]=ismember(HH_bldg_occType{j},HH_agg_occ);
%                 DS_reals_agg_occ{i}(:,5*(occ_idx-1)+1:5*occ_idx) = ...
%                     DS_reals_agg_occ{i}(:,5*(occ_idx-1)+1:5*occ_idx) + ds_reals;
%             end           
%        end
        %%
        save(output_file,'DS_reals_agg_occ','agg_occ','bldg_occType',...
        'agg_bldg_num','RC_per_occ','-v7.3')
    end
else
    0
end