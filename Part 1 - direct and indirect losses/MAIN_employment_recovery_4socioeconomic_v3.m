clear all; close all; clc;


face_colors = [[0.2 0.2 0.2];[0, 0.4470, 0.7410];[0.7350, 0.0780, 0.1840]];
addpath('./Functions')
%%
ind_code = {'Agriculture',...
    'Mining',...
    'Utilities',...
    'Construction',...
    'Manufacturing',...
    'Wholesale',...
    'Retail',...
    'Transport/warehousing',...
    'Information',...
    'Finance/real estate',...
    'Professional/business services',...
    'Education/health',...
    'Arts/recreation/accom./food',...
    'Other services',...
    'Government'};

ind_code_short = {'AGR',...
    'MIN',...
    'UTI',...
    'CON',...
    'MAN',...
    'WHO',...
    'RET',...
    'TRA',...
    'INF',...
    'FIN',...
    'PRO',...
    'EDU',...
    'ART',...
    'OTH',...
    'GOV'};
%% Figures to draw indicators
f9 = 1; axis_fig9 = [0 5 -25 2];
faults = {'Calaveras','Hayward','SanAndreas'};
fault_ind =  2;
scen_flag = 3;

extra_output_suffix = '_baseline';%options:'_retrofit','_alpha35
%%
fault = faults{fault_ind};
c_ind = fault_ind;

load(sprintf('./Input data/Hazard/mult_scenenarios_%s_500sims.mat',fault),'SCENARIOS','SCENARIOS_descrip')
for scen = scen_flag%:(length(SCENARIOS))
    input_filename =  sprintf('./Output/From ARIO/ARIO_v7_4_welfare_results_%s_sims_sc%i_%i_FEB2018_RIMS%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),extra_output_suffix);
    load(input_filename)
    scen
    
    %%
    nsims = length(Direct_losses);
    nt =size(VA_results,1);
    t = t_results/365;
    
    if f9
        
        % Load tract employment data
        emp_tract_raw = csvread('Input data/Employment/BA_2016_employment_tract.csv',1,0);
        BayArea.CensusTract = emp_tract_raw(:,1);
%         output_dir = sprintf('Output/Employment/Employment_%s_sc%i_%i',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2));
%         if exist(output_dir,'dir')
%         else
%             mkdir(output_dir)
%         end
%         headers = {'tract'};
%         count = 1;
%         sub_header = {'Day'};
%         for x = 1:length(t_results)
%             count = count +1;
%             headers{count} = sprintf('%s%i',sub_header{1},t_results(x));
%         end
        
        for ind =1:15
            Jobs_total = reshape(Jobs_results(:,ind,:),nt,nsims);
            Jobs_total_delta = (Jobs_total-repmat(Jobs_total(1,:),nt,1))./repmat(Jobs_total(1,:),nt,1)*100;
            Jobs_fraction = Jobs_total./repmat(Jobs_total(1,:),nt,1);
            Unemp_fraction = 1-Jobs_fraction;
            Jobs_mean_delta = mean(Jobs_total_delta,2);
            Jobs_sorted_delta = sort(Jobs_total_delta,2);
            CI = .95;
            lb_idx = round((1-CI)/2*nsims+1);
            ub_idx = round((1-(1-CI)/2)*nsims);
            % VA with CI
            f = figure(9);
            subplot(3,5,ind)
            %plot(x, curve1, 'r', 'LineWidth', 2);
            hold on;
            x2 = [t, fliplr(t)];
            inBetween = [Jobs_sorted_delta(:,lb_idx)', fliplr(Jobs_sorted_delta(:,ub_idx)')];
            p_H_Jobs_CI=fill(x2, inBetween,face_colors(c_ind,:),'EdgeColor',face_colors(c_ind,:),'FaceAlpha',0.2);
            
            %plot(t, VA_total_delta, 'LineWidth', 1,'Color',[0.5 0.5 0.5]);
            p_H_Jobs_mu= plot(t, Jobs_mean_delta, 'LineWidth', 2,'Color',face_colors(c_ind,:));
            xlabel('Years')
            ylabel('Change in employment (%)')
            box on
            axis(axis_fig9)
            title(sprintf('Employment: %s',ind_code{ind}))
            set(findall(f,'-property','FontSize'),'FontSize',15)
            
%             % Save the results for processing
%             if 0
%                 for sim = 1:nsims
%                     %sim
%                     fprintf('Industry %i: %.2f percent\n',ind,sim/nsims*100);
%                     filename = sprintf('%s/Employment_tract_%s_%s_sc%i_%i_sim%i.csv',...
%                         output_dir,ind_code_short{ind},fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),sim);
%                     emp_ind = emp_tract_raw(:,4+ind)*Jobs_fraction(:,sim)';
%                     data = [BayArea.CensusTract,emp_ind];
%                     csvwrite_with_headers(filename,data,headers)
%                 end
%             end
            
            
            
            output_dir = sprintf('Output/Employment/Employment_ind_%s_sc%i_%i_DEC2018_RIMS%s',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),extra_output_suffix);
            if exist(output_dir,'dir')
            else
                mkdir(output_dir)
            end
            headers = {'sim'};
            count = 1;
            sub_header = {'Day'};
            for x = 2:length(t_results)
                count = count +1;
                headers{count} = sprintf('%s%i',sub_header{1},t_results(x));
            end
            
            if 1
                
                %sim
                fprintf('Industry %i \n',ind);
                filename = sprintf('%s/Employment_%s_%s_sc%i_%i.csv',...
                    output_dir,ind_code_short{ind},fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2));
                data = [[1:nsims]',Unemp_fraction(2:end,:)'];
                csvwrite_with_headers(filename,data,headers)
                
            end
        end
    end
end


