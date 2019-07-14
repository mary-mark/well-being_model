%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   ARIO_version_4_1.m ARIO version 4.1 model main loop
%    Copyright (C) 2007-2010 Stéphane Hallegatte, Météo-France
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    Author: Stephane Hallegatte, hallegatte@centre-cired.fr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Maryia's record of adjustments:
%
%   V7.1, September 4, 2018:    - limiting to one scenario for three faults
%                               - adding of physical reconstruction time constraint
%   V7.3, December 4, 2018:    - adding the non-productive housing sector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Adaptive Regional Input-Ouput (ARIO) model by Stéphane Hallegatte
%   Adjusted by Maryia Markhvida 2018
%
%  Assess the economic cost of natural disasters
%  Simulate the reconstruction phase
%  Economy with N sectors
%  Applied on the Louisiana economy
%
%  changes of version 4.1 with respect to version 1.0
%  1- introduction of household budget and limited insurance penetration
%          i.e. households and companies pay a fraction of reconstruction
%          important: assumption of unlimited access to credit
%  2- household losses are included in the real estate sector
%          not treated separately (no more private reconstruction)
%  3- introduction of inventories (additional flexilibity and resilience),
%          changes in rationing rules (full proportionnal), reduced
%          production when inventories are insufficient
%  4- no price modeling in this version (no demand surge)
%  5- time step = 1 day
%
%  Structured as a function Res_VA = ARIO_version_4_1()
%
%  Can also be called with parameters (to carry out senstivity analysis)
%
%  List of parameters:
%    sens_ana = 1 if sensitivity analysis; 0 otherwise
%    ampl = multiplies the amount of direct losses by ampl with respect to
%               Katrina
%    maxmax_surcapa = maximum capacity for overproduction
%    tau_alpha = timescale for overproduction
%    NbJourStockU = nb of days of inventories for "normal" sectors
%    Tau_Stock = timescale of inventory restoration
%    Adj_prod = parameters for impact heterogeneity
%
%  Inputs: file 'Katrina_CBO_f' with the table BayArea_damages describing the disaster
%     'BayArea_damages' (N,N)
%       where BayArea_damages(i,j) is reconstruction demand by sector i to sector j
%
%  Output: Res_VA is a table (NStep+1,N) providing the value added by
%            each sector for a given day
%
%%%%%%%%%%%%%
%  Other needed input:
%  Inputs: File "Louisiana" with tables describing the local economy:
%     'T_CA': total production per sector
%     'L_CA': employment per sector
%     'IO_Table_CA': Input-Ouput table (caution: the matrix is transposed in the code)
%                       this table includes only what is produced AND consumed in the region
%                       this table does not include housing services
%     'Imports_CA': Imports per sector (i.e. imports from all sectors together, needed by each sector to produce)
%         (to produce T(i), the sector i needs to import a value Imports(i) of goods and services from all sectors)
%     'Exports_CA': exports by each sector
%     'Local_Dem_CA': local demand toward each sector
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%function Res_VA=ARIO_version_4_1_NO(sens_ana,ampl,alpha_prod_max,tau_alpha,NbJourStockU,Tau_Stock,Adj_prod)

disp('*******************************************************************')
disp('Adaptive Regional Input-Ouput (ARIO) model by Stéphane Hallegatte, version 4.1')
disp('Modified by Maryia Markhvida, 2018')
disp('Copyright (C) 2007-2009 Stéphane Hallegatte and Météo-France')
disp('Applied on the Bay Area earthquake scenarios')
disp('*******************************************************************')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MODEL PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% IMPORTANT: the time step is ONE DAY
% All time unit are YEARS
% Economic fluxes are in Euro per Year
%%
close all; clear all;
addpath('./Functions/')

% Chose fault and scenario
fault = 'Hayward';
scenario_flags = 3;
for scenario_flag = scenario_flags

extra_output_suffix = '_retrofit';%options: '_baseline','_alpha35','_retrofit'
extra_input_suffix = '_retrofit';% options: '_baseline', '_retrofit'
defined_alpha = 1.25;

%%%%%%%%%%%%% Control variables
% Default reconstruction timescale (in years)
tau_recon_default = 1/2;
%%%%%%%%%%%%%%%%%%%%%%

% time step: one day
dt = 1/365.;
% simulation length
NStep=365.*10;
% number of sectors
N=15;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MODEL PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if no parameters then no sensitivity analysis
if not(exist('sens_ana'))
    sens_ana = false;
end

% if no sensitivity analysis then reference parameters
if not(sens_ana) %
    % production over-capacity parameters
    alpha_prod_max = defined_alpha*ones(1,N);%1.25*ones(1,N);
    tau_alpha = 1; % years
    % Nb of days of stock
    NbJourStockU = 90; % 60
    % timescale of stock building (year)
    Tau_Stock = 30/365.;
    % size of the direct losses
    ampl = 1;
    % parameter of production smoothing
    Adj_prod = .8;
    
end
% inventories
NbJourStock = NbJourStockU*ones(1,N);
% production smoothing parameter - heterogeneity coefficient per industry
Psi =  Adj_prod*ones(1,N);

% timescale of debt reimbursement (years)
tauR=2;
% epsilon used to estimate what is full recovery (in adaptation process)
epsilon=1.e-6;

if sens_ana % if sensitivity analysis then write parameters
    disp('Sensitivity analysis - Parameters:');
    [ampl,alpha_prod_max(1),tau_alpha,NbJourStock(1),Tau_Stock,Adj_prod]
end


% DELAY TO SUBSTITUTE THE PRODUCT OF THIS COMPANY WITH IMPORTS
% If =0 then, no substitution possible (e.g., electricity, water)
Sub_Del = zeros(N,1);
for i=1:N
    Sub_Del(i)=1;
    NoStock(i)=0;
end

% INVENTORIES FOR NON STOCKABLE GOODS & SERVICES
% Utilities
NbJourStock_Short = 3.*NbJourStockU/60;
NoStock(3)=1;
NbJourStock(3)=NbJourStock_Short;
% Transportation
NoStock(6)=1;
NbJourStock(6)=NbJourStock_Short;
% Specifics of the construction sector
NbJourStock(4)=365*100000; % construction sector production is not urgent (in the IO table)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ECONOMIC DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Household insurance penetration rate
penetration = 1.0;
% Business insurance penetration rate
penetrationf= 1.0 ;
%%%
% Household and transportation % non operational yet
% Total_Household = ?? ;
% Fact_Housing_Loss_Workers = ??;
%%%
% alpha = fraction of capital that is owned locally - ad hoc assumption, no data
% sources on Mumbai - used to know how decrease in business profit affects
% household income.
alpha = 0.5;
% wage = numeraire
wage = 1.;
%%%%%%%%%%%%%%%%%%%

load('Bay Area IO inputs/ratio_K2Y_adjusted','ratio_K2Y');% MARYIA
load('Bay Area IO inputs/IO_Data_BA_RIMS_v1'); %Maryia

exchange_rate = 1;
T = T_BA*exchange_rate; % prod totale
Labor = Labour_BA*exchange_rate; %
IO = IO_Table_BA*exchange_rate;


% Assign pre-EQ economic variables
Imports_pre_eq = Imports_BA*exchange_rate;
Exports_pre_eq = Exports_BA*exchange_rate;
Local_demand_pre_eq = Local_demand_BA*exchange_rate;
Jobs_pre_eq = Job_BA;
% Intermediate purchases or consumption
Inter_purchases_pre_eq = sum(IO,1);
% Total sales to other industries
Inter_sales_pre_eq = sum(IO,2);

Production_pre_eq = Exports_pre_eq' + Local_demand_pre_eq' + Inter_sales_pre_eq';
VA_pre_eq = Production_pre_eq - Inter_purchases_pre_eq - Imports_pre_eq;
Total_local_demand_pre_eq=sum(Local_demand_pre_eq);

% Normalized IO table
IO_norm = IO./repmat(Production_pre_eq,N,1);
IO_for_checking = IO./repmat(Production_pre_eq',1,N);
% Asset calculation
Assets_pre_eq = VA_pre_eq.*ratio_K2Y*exchange_rate;
disp('Economy data loaded');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIAL SITUATION (to check consistency and variable initialization)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

openness_rate=0.5;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CREATION OF THE DISASTER %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Choose scenario parameters


load(sprintf('../Input data/Hazard/mult_scenenarios_%s_500sims.mat',fault),'SCENARIOS','SCENARIOS_descrip')
rng(1)
t_results = [0,1:7:NStep];
nt = length(t_results);

% Choose scenarios based on the scenario_flag
SCENARIOS_descrip = SCENARIOS_descrip(scenario_flag,:);
SCENARIOS = SCENARIOS(scenario_flag);


for scen = 1:(length(SCENARIOS))
    
    fprintf('This is scenario: %s %i',fault,scen)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Start the big simulation loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    damages_input_filename =  sprintf('../Output/For ARIO/Losses_4ARIO_%s_sims_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),extra_input_suffix);
    load(damages_input_filename,'BayArea_damages_sims','frac_loss_prod'); % already in euros !
    
    nsims = length(BayArea_damages_sims);
    
    %%%% Set up reconstruction time parameters
    t_recovery_input_filename =  sprintf('../Output/Recovery/Industry/t_95_ind_recovery_%s_sc%i_%i_DEC2018%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),extra_input_suffix);
    load(t_recovery_input_filename);
    tau_recon_sims = time_95_recovered/7/52;
    tau_recon_sims(tau_recon_sims == 1/7/52) = tau_recon_default;
    %tau_recon_sims(:) = tau_recon_default;
    %%%%%%%%%%%%%%%
    
    
    
    VA_results = zeros(nt,N,nsims);
    Jobs_results = zeros(nt,N,nsims);
    Employee_comp_results = zeros(nt,N,nsims);
    Profit_results = zeros(nt,N,nsims);
    Reconstr_needs_results = zeros(nt,N,nsims);
    Direct_losses = zeros(1,nsims);
    Indirect_losses = zeros(1,nsims);
    Loss_amplification = zeros(1,nsims);
    data_4csv = zeros(nsims,nt+2);
    parfor sim = 1:nsims
        fprintf('This is simulation %i\n',sim)
        %%%%%%%%%%%%%%%%%%%%%% RESET VARIABLES %%%%%%%%%%%%%%%%%%%
        % initial surcapacity situation % can be used to include a production gap
        alpha_prod= ones(N,1)*1;
        % Imports for consumption, assumed equal to local_dem
        Imports_C = Local_demand_pre_eq;
        Inter_purchases = Inter_purchases_pre_eq;
        
        % Re-initialize key vairables
        %OK=zeros(NStep+1,N);
        actual_housing_loss = zeros(NStep+1,1);
        actual_total_production = zeros(NStep+1,1);
        actual_Imports = zeros(NStep+1,N);
        actual_Exports = zeros(NStep+1,N);
        actual_Local_demand = zeros(NStep+1,N);
        actual_Dem_Exports = zeros(NStep+1,N);
        actual_Dem_Imports = zeros(NStep+1,N);
        actual_final_cons = zeros(NStep+1,N);
        actual_inter_sales=zeros(NStep+1,N);
        Final_demand_unsatisfied=zeros(NStep+1,N);
        actual_Labour = zeros(NStep+1,N);
        actual_Jobs = zeros(NStep+1,N);
        Total_Labour=zeros(NStep+1,1);
        Debt=zeros(NStep+1);
        price = ones(NStep+1,N);
        payback=zeros(NStep+1);
        macro_effect=ones(NStep,1);
        actual_reconstr = zeros(NStep+1,1);
        Reconstr_demand_matrix= zeros(N,N,NStep+1);
        Reconstr_needs = zeros(NStep+1,N);
        Budget=zeros(1,NStep+1);
        Production=zeros(NStep+1,N);
        Profit=zeros(NStep+1,N);
        Prof_rate=zeros(NStep+1,N);
        Demand = zeros(NStep+1,N);
        actual_Reconstr_inv=zeros(NStep+1,N);
        actual_reconstr_demand_sat=zeros(NStep, N);
        VA = zeros(NStep+1,N);
        cout = 0;
        Stock = zeros(N,N);
        Long_ST = zeros(N,N);
        Order = zeros(N,N);
        Total_profit=zeros(NStep+1,1);
        
        
        Production(1,:) = Production_pre_eq;
        % sector profit
        Profit(1,:)= Production(1,:) - (Inter_purchases_pre_eq+wage*Labor+Imports_pre_eq);
        % profit rate
        Prof_rate(1,:) = Profit(1,:)./T';
        % sector value added
        VA(1,:)=VA_pre_eq;
        Total_Labour(1)=sum(Labor);
        actual_Labour(1,:)=Labor;
        actual_Jobs(1,:) = Jobs_pre_eq;
        actual_Imports(1,:)=Imports_pre_eq;
        actual_Exports(1,:)=Exports_pre_eq;
        actual_Dem_Imports(1,:)=Imports_pre_eq;
        actual_Local_demand(1,:)=Local_demand_pre_eq;
        plus=zeros(N,N);
        Order_prod = Production(1,:);
        
        macro_effect(1) = 1;
        Demand(1,:) = Production(1,:);
        Total_profit(1) = sum(Profit(1,:));
        earnings_pre_eq = Total_profit(1)+Total_Labour(1);
        
        % Profits from business outside the affected region
        % Assumption 1: Profits that leave the region are equal to Profits that enter
        % the region
        Pi = (1-alpha)*sum(Profit(1,:));
        % Assumption 2: Profit as needed to balance the local economy
        %Pi = sum(Local_demand_pre_eq)+sum(Imports_C) - (sum(alpha*Profit(1,:))+sum(Labor));        %%%%%%%%%%%%%%%%%%%%%%%CLARIFY
        
        % Initial household consumption and investment
        % Assumption: Investments are made by households, not by businesses
        DL_ini = wage*sum(Labor(:))+ alpha*sum(Profit(1,:)) + Pi;
        % Alternative: DL_ini = wage*sum(L(:))+ (alpha*sum(Profit(1)) + Pi)*Redistr;
        % where Redistr gives the amount of business profits that is redistributed
        % to household. In this case, macro_effect must be modified, and the
        % Final demand has to be modified to make a difference between business
        % investment on the one hand, and household consumption and investment on
        % the other hand.
        for i=1:N
            for j=1:N
                Stock(i,j) = IO(i,j)*NbJourStock(i)/365;
                Long_ST(i,j) = Stock(i,j);
                Order(i,j) = IO(i,j);
                % if needed to be recorded
                %        mem_Order(1,i,j)=Order(i,j);
                %        mem_Stock(1,i,j)=Stock(i,j);
            end
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        BayArea_direct_losses = BayArea_damages_sims{sim}*ampl;
        Destr_capital_ini = sum(BayArea_direct_losses,2).*frac_loss_prod(sim,:)';
        Destr = Destr_capital_ini./(ratio_K2Y.*VA(1,:))';
        %mem_Destr(1,:) = Destr;
        tau_recon =  tau_recon_sims(:,sim);
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% PRODUCTION POST-DISASTER: ECONOMIC MODEL
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % initialisation
        %OK(1,:)=1-Destr;
        for i=1:N
            for j=1:N
                Reconstr_demand_matrix(i,j,1)=BayArea_direct_losses(i,j);
            end
        end
        
        Reconstr_needs(1,:) = sum(BayArea_direct_losses,2)';
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% loop on days (k=number of days)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %disp('Start simulation');
        for k=1:NStep
            
            if (mod(k,365)==364) % every year
                %  disp(strcat('year:',int2str((k-1)/365),' completed'));
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % FINAL DEMANDS CALCULATIONS (A.4)
            % Assess demand size constraints %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % calculation of demand as a function of budget; demand for reconstruction by households does not depend on budget
            % IMPORTANT (i.e., perfect access to credit for reconstruction)
            [Local_demand_new,Exports_new,Reconstr_demand,Demand_new, Order_agg_new]...
                = adjust_LFD_varying_tau_recon(macro_effect(k), Local_demand_pre_eq,Order,Exports_pre_eq, Reconstr_demand_matrix(:,:,k),tau_recon);
            
            Imports_C_new = macro_effect(k)* Imports_C';
            %actual_Local_demand(k+1,:) = Local_demand_new;
            %actual_Exports(k+1,:) = Exports_new;
            actual_Imports_C = Imports_C_new;
            Demand(k+1,:) = Demand_new;
            
            
            %      mem_dem_export(k+1,i)= actual_Exports(k+1,i);
            %      mem_dem_local(k+1,i) = actual_Local_demand(k+1,i);
            %  mem_demand_total(k+1,:)=  Demand_new;
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Assess supply-side constraint (production capacity)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Production(k+1,:) = Demand(k+1,:);
            
            % Limitation by production capacity
            % assess production limits based on production capacity
            [prod_lim_by_cap, product_cap]  =  limit_production_by_capacity(1-Destr, alpha_prod, Production_pre_eq,Production(k+1,:));
            Production(k+1,:) = prod_lim_by_cap;
            % Update the productive capital of next step
            %OK(k+1,:)=min(1,Production(k+1,:)./Demand(k+1,:));
            
            % Limitation by supplies
            % assess production limits due to insufficient stocks
            %       Stock(i,j)=stock of goods i owned by j
            %       Normalized IO : IO(i,j)/Produc(1,j)
            Stock_target = get_required_inventory(Production(k+1,:),IO_norm, NbJourStock/365);
            [ prod_lim_by_sup, constraint_idx] = limit_production_by_supplies(Psi,Stock,Stock_target,   Production(k+1,:));
            Production(k+1,:) = prod_lim_by_sup;
            % Save results
            %  mem_qui_out_sec(k+1,:) = constraint_idx;
            %   mem_out_sec(k+1,:) =  prod_lim_by_sup;
            actual_total_production(k+1)=sum(Production(k+1,:));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Rationing of the supplies (2013 paper section 3.1.4) %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Satisfied demands
            % new rationing scheme: full proportional (final demand and interindustry demands)
            [ reconstr_demand_sat, local_demand_sat, exports_sat,order_sat,Final_demand_unsatisfied(k+1,:)] ...
                = get_satisfied_demands(Reconstr_demand,Exports_new, Local_demand_new, Order, Production(k+1,:),Demand_new);
            
            actual_reconstr_demand_sat(k+1,:) = reconstr_demand_sat;
            actual_final_cons(k+1,:) = local_demand_sat;
            %actual_Exports(k+1,:) = exports_sat;
            Supply = order_sat;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update stock for next time step (k+1) (2013 paper eq 20) %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stock dynamics (Stock(i,j) = stock of goods i own by sector j)
            [Stock] = update_stock( Stock, Supply,IO_norm, Production(k+1,:), dt,epsilon);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update orders for next time step (k+1) (2013 paper eq 4) %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Stock_target = get_target_inventory(Demand_new,IO_norm, NbJourStock/365,product_cap);
            Long_ST_pre = Long_ST;
            %%%%%%%%%%%%%%%% FIX THIS!!!!! %%%%%%%%%%%%%%%%%%%%%%%
            % intro  duction of smoothing to reduce numerical instabilities
            % %%%%%%%?
            tau_stock_target = 60./365.;
            Long_ST = Long_ST + dt/tau_stock_target*(Stock_target - Long_ST); % TEST
            %delta_Long_ST(k) =  sum(sum(Long_ST_pre-Long_ST))/sum(sum(Long_ST_pre))*100;  %       update_order
            
            %       update_order
            for i=1:N
                for j=1:N
                    % Order by j to i
                    Order(i,j) = max(epsilon,Production(k+1,j)/Production(1,j)*IO(i,j) + (Long_ST(i,j)-Stock(i,j))/(Tau_Stock*NbJourStock(i)/NbJourStockU));
                    %Order(i,j) = max(epsilon,Production(k+1,j)/Production(1,j)*IO(i,j) + (Stock_target(i,j)-Stock(i,j))/(Tau_Stock*NbJourStock(i)/NbJourStockU));
                    
                    % if need to be recorded
                    %            mem_Order(k+1,i,j)=Order(i,j);
                    %            mem_Stock(k,i,j)=Stock(i,j);
                    %            mem_Needed_Stock(k,i,j)=S_target(i,j);
                    %            mem_Stock_target(k,i,j)=Stock_target(i,j);
                end
            end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update all of the economic metrics for next time step (k+1)  %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Imports
            Imports_new = Imports_pre_eq.*Production(k+1,:)./Production_pre_eq;
            
            % Value added and cost of intermediate consumption
            [VA_new, Inter_purchases_new,Inter_sales_new] = get_value_added(Production(k+1,:),Imports_new, IO_norm, price(k+1,:));
            actual_Imports(k+1,:) = Imports_new;
            actual_inter_sales(k+1,:) = Inter_sales_new;
            VA(k+1,:) = VA_new;
            Inter_purchases = Inter_purchases_new;
            
            % Employment and labor costs
            actual_Jobs(k+1,:)=Jobs_pre_eq'.*VA_new./VA_pre_eq; % IMPORTANT: hiring always possible
            actual_Labour(k+1,:)=Labor.*Production(k+1,:)./Production_pre_eq; % IMPORTANT: hiring always possible
            Total_Labour(k+1) = sum(actual_Labour(k+1,:));  % total consumed labor
            % Profits
            %   Profits are reduced by reconstruction spending (as a function of
            %   insurance penetration) (warning: unchanged prices)
            Profit(k+1,:)= Production(k+1,:) - (Inter_purchases + actual_Labour(k+1,:)+ actual_Imports(k+1,:))- actual_Reconstr_inv(k,:)*(1-penetrationf);
            Total_profit(k+1) = sum(Profit(k+1,:));
            Prof_rate(k+1,:) = Profit(k+1,:)./T';
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RECONSTRUCTION MODELLING
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % cout = cout + prix(k+1,j)*dt*actual_reconstr_demand_sat_rate(k+1,j);
            [reconstr_demand_matrix_new, reconstr_inv_new] = update_reconstruction_v2(Reconstr_demand_matrix(:,:,k),reconstr_demand_sat, Reconstr_demand,dt,tau_recon);
            actual_Reconstr_inv(k+1,:) = reconstr_inv_new';
            Reconstr_demand_matrix(:,:,k+1) = reconstr_demand_matrix_new;
            Reconstr_needs(k+1,:) =  Reconstr_needs(k,:)- reconstr_inv_new';
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % OVERPRODUCTION MODELLING %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % increase in production capacity
            scarcity_index = (Demand(k+1,:) - Production(k+1,:))./Demand(k+1,:);
            alpha_prod = get_overproduction_capacity(alpha_prod, alpha_prod_max, tau_alpha, scarcity_index,epsilon, dt);
            %    mem_alpha_prod(k+1,:)=alpha_prod;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update loss of productive capital in sector i, as a function of reconstruction needs and total capital amount
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i=1:N
                Destr(i)=sum(Reconstr_demand_matrix(i,:,k+1))/(Assets_pre_eq(i))*frac_loss_prod(sim,i);
            end
            %   mem_Destr(k+1,:) = Destr;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % HOUSEHOLD BUDGET MODELING
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % reduction in consumption by macro_effect is budget is reduced
           % Budget(k+1) = Budget(k)+ ((wage*sum(actual_Labour(k,:))+ alpha * Total_profit(k) + Pi) - sum(actual_Imports_C)- sum(actual_final_cons(k,:)))/12 - (1-penetration)*actual_reconstr(k);
           % macro_effect(k+1) = (DL_ini + 12*1/tauR * Budget(k+1))/DL_ini;
            
            Budget(k+1) = Budget(k)+ (wage*Total_Labour(k+1)-wage*Total_Labour(k)+alpha*(Total_profit(k+1) - Total_profit(k)));
            macro_effect(k+1) = (DL_ini + 1/tauR * Budget(k+1))/DL_ini;
            
            
        end % of loop time
        
        %Store Results
        VA_results(:,:,sim) = VA([1,2:7:end],:);
        Jobs_results(:,:,sim) = actual_Jobs([1,2:7:end],:);
        Employee_comp_results(:,:,sim) = actual_Labour([1,2:7:end],:);
        Profit_results(:,:,sim) = Profit([1,2:7:end],:);
        Reconstr_needs_results(:,:,sim) = Reconstr_needs([1,2:7:end],:);
        Direct_losses(sim) = sum(sum(BayArea_direct_losses(:,:)));
        Indirect_losses(sim) = sum(sum(VA(1,:))-sum(VA,2))*dt;
        Indirect_losses_ind(sim,:) = sum((repmat(VA(1,:),NStep+1,1)-VA)*dt);
        Loss_amplification(sim) = (Direct_losses(sim)+Indirect_losses(sim))/Direct_losses(sim);
        Reconstr_needs = Reconstr_needs([1,2:7:end],:);
        actual_Reconstr_inv_short = zeros(length(2:7:size(VA,1))+1,N);
        actual_Reconstr_inv_short(1,:) = actual_Reconstr_inv(1,:);
        for ii = 2:length(2:7:size(VA,1))+1
            if ii == length(2:7:size(VA,1))+1
                actual_Reconstr_inv_short(ii,:) = sum(actual_Reconstr_inv([t_results(ii):end],:),1);
                
            else
                actual_Reconstr_inv_short(ii,:) = sum(actual_Reconstr_inv([t_results(ii):t_results(ii)+6],:),1);
            end
        end
        
        %% Write a variable for HH reconstruction satisfaction rate
        data_4csv(sim,:) = [sim,Reconstr_needs(1,10),actual_Reconstr_inv_short(:,10)'];
    end
    
    
    VA_delta = (VA_results(:,:,311)-repmat(VA_results(1,:,311),nt,1,1))./repmat(VA_results(1,:,311),nt,1,1)*100;
    fig_25=figure(25)
    hold on
    for sub = 1:15
        subplot(3,5,sub)
        %plot([i-0.5 i-0.5],[0 5],'k')
        plot(t_results/365,VA_delta(:,sub),'k')
        axis([0,8,-20,5])
        % title(ind_code{sub})
    end
    %% save results
    if 1
        results_output_filename =  sprintf('../Output/From ARIO/ARIO_v7_4_welfare_results_%s_sims_sc%i_%i_FEB2018_RIMS%s.mat',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),extra_output_suffix);
        save(results_output_filename,'VA_results','Jobs_results','Employee_comp_results','Profit_results','Reconstr_needs_results',...
            'Direct_losses','Indirect_losses','Loss_amplification','t_results','Indirect_losses_ind')
    end
     if 0
        reconstr_output_filename =  sprintf('../Output/From ARIO/ARIO_v7_4_HH_reconstr_demand_satisfaction_%s_sc%i_%i_FEB2018_RIMS%s.csv',fault,SCENARIOS_descrip(scen,1),SCENARIOS_descrip(scen,2),extra_output_suffix);
        headers = {'sim','HH_recon_needs'};
        count = 2;
        sub_header = {'Day'};
        for x = 1:length(t_results)
            count = count +1;
            headers{count} = sprintf('%s%i',sub_header{1},t_results(x));
        end
        csvwrite_with_headers(  reconstr_output_filename ,data_4csv,headers)
    end
    disp('Results saved in file');
    %end
end
end
