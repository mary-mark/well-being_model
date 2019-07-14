# This script provides data input for the resilience indicator multihazard model for the Philippines, Fiji, Sri Lanka, and (eventually) Malawi. 
# Restructured from the global model and developed by Jinqiang Chen and Brian Walsh 

# Magic
#from IPython import get_ipython
#get_ipython().magic('reset -f')
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')

# Import packages for data analysis
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import isnull
import os, time
import warnings
import sys
import pickle
import copy

from libraries.lib_asset_info import *
from libraries.lib_country_dir import *
from libraries.lib_gather_data import *
from libraries.lib_sea_level_rise import *
from libraries.replace_with_warning import *
from libraries.lib_agents import optimize_reco,get_physical_recon_rate,optimize_reco_w_labour_and_physical_constraint,get_constr_sector_recon_rate

warnings.filterwarnings('always',category=UserWarning)


if len(sys.argv) < 2:
    #print('Need to list country. Currently implemented: MW, PH, FJ, SL')
    print('This is the model for Bay Area')
    myCountry = 'BA'
    suffix = '_baseline'# options ('_baseline',no_code','retrofit','no_UI','UI_2yrs','no_ins','ins50_15')
    nsims = 4
    if not os.path.exists('optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/"):
        os.makedirs('optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/")
    #myCountry = 'SL'
else: myCountry = 'SL'#'BA'#sys.argv[1]

# Set up directories/tell code where to look for inputs & where to save outputs
intermediate = set_directories(myCountry,suffix,nsims)

# Administrative unit (eg region or province or county)
economy = get_economic_unit(myCountry)

# Levels of index at which one event happens
event_level = [economy, 'hazard', 'rp']

#Country dictionaries
# df = state/province names
df = get_places(myCountry,economy)


prov_code,region_code = get_places_dict(myCountry)

# Secondary dataframe, if necessary
# For PH: this is GDP per cap info from FIES2013
#df2 = get_df2(myCountry)

###Define parameters
df['avg_prod_k']             = get_avg_prod(myCountry) # average productivity of capital, value from the global resilience model
df['shareable']              = asset_loss_covered      # target of asset losses to be covered by scale up
df['T_rebuild_K']            = reconstruction_time     # Reconstruction time
df['income_elast']           = inc_elast               # income elasticity
df['max_increased_spending'] = max_support             # 5% of GDP in post-disaster support maximum, if everything is ready
df['pi']                     = reduction_vul           # how much early warning reduces vulnerability

#df['rho']                    = discount_rate           # discount rate
df['rho']                    = 0.3*df['avg_prod_k']    # discount rate
# ^ We have been using a constant discount rate = 0.06
# --> BUT: this breaks the assumption that hh are in steady-state equilibrium before the hazard

# Protected from events with RP < 'protection'
df['protection'] = 1
if myCountry == 'SL': df['protection'] = 5
inc_sf = None
#if myCountry =='FJ': inc_sf = (4.632E9/0.48) # GDP in USD (2016, WDI) -> FJD


# Construct cat_info for the Bay Area separately
print os.getcwd()
if myCountry =='BA':
    raw_data = pd.read_csv('../../Pre-process - Simply analytics data/Output/INPUT_for_resilience_model_v2.csv')
    raw_data_real_estate = pd.read_csv('../../Pre-process - Simply analytics data/Output/INPUT_real_estate_for_resilience_model.csv')
    fault_identifier = 'Hayward_sc28_6'
    
    
    cat_info = pd.DataFrame(raw_data.iloc[:,0:63])
    cat_info = pd.merge(cat_info,raw_data_real_estate,on='tract',how='inner')
    # Drop values where pcwgt ==0 or hhwgt ==0
    cat_info = cat_info.loc[(cat_info.pcwgt!=0) & (cat_info.n_hh!=0) & (cat_info.hh_str_asset_value!=0)]
                                         
#                                         ['county_name','tract','pcwgt','pcinc','pov_frac','pop_dep','hh_share','pcsoc','hh_str_asset_value',
#                 'peinc_AGR','peinc_MIN','peinc_UTI','peinc_CON','peinc_MAN','peinc_WHO','peinc_RET','peinc_TRA',
#                                    'peinc_INF','peinc_FIN','peinc_PRO','peinc_EDU','peinc_ART','peinc_OTH','peinc_GOV',
#                                    'Emp_AGR','Emp_MIN','Emp_UTI','Emp_CON','Emp_MAN','Emp_WHO','Emp_RET','Emp_TRA','Emp_INF','Emp_FIN',
#                                    'Emp_PRO','Emp_EDU','Emp_ART','Emp_OTH','Emp_GOV','Unemp','n_hh','LIL_hh',
#                                    'VLIL_hh','ELIL_hh']])
    cat_info = cat_info.rename(index = str, columns = {'n_hh':'hhwgt','hh_size':'hhsize',
                                                       'hh_str_asset_value':'str_asset_value_tot',
                                                       'county_name':'county',
                                                       'hh_pub_ass_inc':'hh_pub_assist_inc'})
    cat_info['has_ew'] = 0
    ind_names = ['AGR','MIN','UTI','CON','MAN','WHO','RET','TRA','INF','FIN','PRO','EDU','ART','OTH','GOV']

    cat_info = cat_info.set_index(['county','tract'])
    
    
    ## Process the labor income
    cat_info['inc_L_tot'] = 0
    cat_info['pcinc_L'] = 0
    for ind in ind_names:
        # get total industry income
        cat_info['inc_'+ind+'_tot'] = cat_info.loc[:,'peinc_'+ind]*cat_info.loc[:,'Emp_'+ind]
         # add up to the total labour income (all industries)
        cat_info.inc_L_tot += cat_info['inc_'+ind+'_tot']
        # get per capita (pc) industry income
        cat_info['pcinc_'+ind] = cat_info['inc_'+ind+'_tot']/cat_info.pcwgt
        # add up to the total pc labour income (all industries)
        cat_info.pcinc_L += cat_info['pcinc_'+ind]
        
        # round the labour incomes
        cat_info.pcinc_L = cat_info.pcinc_L.round(0)
    
        # drop unnecessary columns
        cat_info = cat_info.drop(columns = 'Emp_'+ind) # drop unnecessary columns
        cat_info = cat_info.drop(columns = 'peinc_'+ind) 
        
    #cat_info = cat_info.drop(columns =['LIL_pc','VLIL_pc','ELIL_pc'])
    ## Variables in previous analyses that are not used anymore
    #cat_info['pcinc_ae'] = None
    #cat_info['hhsize_ae'] = None
    #cat_info['pcwgt_ae'] = None
    #cat_info['quantile'] = None
    #cat_info['c_5'] = None
    #cat_info['gamma_SP'] = None 
    #cat_info['ispoor'] = 0        
    #cat_info['dk'] = None #To add, this is the loss in capital 

    #cat_info['c'] = cat_info.pcinc
    #cat_info['social'] = cat_info.pcsoc/cat_info.c
    
    # Set a floor for pcinc
    cat_info['pcinc_non_adj'] = cat_info['pcinc'].copy()

    # THIS IS FROM PERVIOUS VERSION: Make sure their income is corresponding to the structural value of the houses
    #   cat_info['pcinc'] = pd.concat([cat_info.pcinc,cat_info.str_asset_value_tot*cat_info.hh_share/cat_info.pcwgt*get_avg_prod(myCountry)],axis = 1).max(axis = 1)  

    # Make sure pc income is at least at big as the labour income
    cat_info['pcinc'] =  pd.concat([cat_info.pcinc,cat_info.pcinc_L],axis = 1).max(axis = 1)
    
    # Add all sources of income and calculate the total income
    cat_info['pcinc_oth'] = (cat_info.pcinc-cat_info.pcinc_L)
    cat_info['pcinc_h'] = cat_info.k_pc_h * get_avg_prod(myCountry)
        # Th pcinc_tot total income includes non-monetary income from housing services
    cat_info['pcinc_tot'] =cat_info.pcinc_L + cat_info.pcinc_oth + cat_info.pcinc_h - cat_info.p_rent_pc
    
    
    # Get productive capital values
    cat_info['k_pc_L'] = cat_info.pcinc_L/get_avg_prod(myCountry)
    cat_info['k_pc_oth'] =  cat_info.pcinc_oth/get_avg_prod(myCountry)
    cat_info['k_pc'] =   cat_info.k_pc_L + cat_info.k_pc_h + cat_info.k_pc_oth
# =============================================================================
## =============================================================================
   
    
    # Get total values, aggregated on a census level
    cat_info['k_tot'] =   cat_info.k_pc*cat_info.pcwgt
    cat_info['savings_tot'] = cat_info.savings_per_hh*cat_info.hhwgt
    cat_info['pension_ss_tot'] = cat_info.pension_ss_pc*cat_info.pcwgt


# Drop unnecessary columns
    columns_to_drop =  ['n_hh_inc_less15','n_hh_inc_15_25', 'n_hh_inc_25_35','n_hh_inc_35_50', 'n_hh_inc_50_75',
 'n_hh_inc_75_100', 'n_hh_inc_100_125', 'n_hh_inc_125_150', 'n_hh_inc_150_200', 'n_hh_inc_more200']
    
    cat_info = cat_info.drop(columns = columns_to_drop)

    #Write the data
    cat_info.to_csv(intermediate+'/cat_info.csv',encoding='utf-8', header=True)
    print 'DONE WRITING CAT INFO'

    
    
#%%
## Write the hazard_ratios.csv file    
if myCountry =='BA':
    rp = np.arange(1,nsims+1,1)
    
    #%% Create labour income recovery pickle
    x_max = 10 # 10 years
    x_min, n_steps = 0.,52.*x_max # <-- time step = week
    int_dt,step_dt = np.linspace(x_min,x_max,num=n_steps,endpoint=True,retstep=True)
    L_pcinc_dic = {}
    L_recov_dic = {}
    try:
        loaded_list = pickle.load(open(intermediate+'/labour_recovery_'+fault_identifier+'_'+str(nsims)+'sims.p','rb'))
        L_pcinc_dic = loaded_list[0]
        L_recov_dic = loaded_list[1]
    except:
        for ind in ind_names:
            df_labour = pd.read_csv('../inputs/BA/Employment_ind_'+ fault_identifier +'_DEC2018_RIMS'+suffix+'/Employment_'+ind+'_'+fault_identifier +'.csv')
            L_pcinc_dic[ind] = cat_info['pcinc_'+ind]
            for sim in rp:
                print 'Ind:',ind,' sim:',sim
                L_recov_dic[(ind,sim)] = np.array(df_labour.iloc[sim-1,1:int(n_steps)+1])
        pickle.dump([L_pcinc_dic,L_recov_dic], open(intermediate+'/labour_recovery_'+fault_identifier+'_'+str(nsims)+'sims.p', 'wb' ))
    
    raw_data = pd.read_csv('../../Pre-process - Simply analytics data/Output/INPUT_for_resilience_model_v2.csv')
    hazard = ['EQ']
    hazard_index = pd.MultiIndex.from_product([hazard, rp, cat_info.index.get_level_values(1)],names=['hazard', 'rp','tract'])
    hazard_ratios = pd.DataFrame(index = hazard_index)
    hazard_ratios['avg_prod'] = get_avg_prod(myCountry)
    for i_tract,tract in enumerate(cat_info.index.get_level_values(1)):
    	print 'Distributing data for harazd_ratios, TRACT:',i_tract
        hazard_ratios.loc[(slice(None),slice(None),tract),'k_pc']= cat_info.k_pc[(slice(None),tract)].values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract),'k_pc_str']= cat_info.k_pc_str[(slice(None),tract)].values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract),'k_pc_h']= cat_info.k_pc_h[(slice(None),tract)].values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract),'p_rent_pc']= cat_info.p_rent_pc[(slice(None),tract)].values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract),'mort_pc']= cat_info.mort_pc[(slice(None),tract)].values[0]       
        hazard_ratios.loc[(slice(None),slice(None),tract),'pcinc_tot']= cat_info.pcinc_tot[(slice(None),tract)].values[0]

    
        hazard_ratios.loc[(slice(None),slice(None),tract),'hh_share']= cat_info.hh_share[(slice(None),tract)].values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract),'pcwgt']= cat_info.pcwgt[(slice(None),tract)].values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract),'county'] = raw_data[raw_data['tract']==tract].county_name.values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract), 'str_asset_value_tot'] = cat_info.str_asset_value_tot[(slice(None),tract)].values[0]
        hazard_ratios.loc[(slice(None),slice(None),tract), 'c_pc_min'] = cat_info.pov_lev_pc[(slice(None),tract)].values[0]
    
    
    hazard_ratios = hazard_ratios.reset_index(['hazard','rp'])
    for sim in rp:
            
        print 'Loading losses, sim',sim
        losses = pd.read_csv('../inputs/BA/Loss_HH_'+fault_identifier
            +'_DEC2018'+suffix+'/Loss_HH_'+fault_identifier
            +'_sim'+str(sim)+'_agg.csv').set_index('tract')
        
        losses.reset_index(inplace = True)
        losses = losses.loc[losses.tract.isin(cat_info.index.get_level_values(1))].set_index('tract')
        hazard_ratios.loc[hazard_ratios.rp ==sim,'asset_loss_tot_ins'] = losses.HH_loss_insured
        hazard_ratios.loc[hazard_ratios.rp ==sim,'asset_loss_tot'] = losses.HH_loss_total
        hazard_ratios.loc[hazard_ratios.rp ==sim,'di0_pc_L'] = 0
        for ind in ind_names:
            hazard_ratios.loc[hazard_ratios.rp ==sim,'di0_pc_'+ind] = (L_recov_dic[(ind,sim)][0]*L_pcinc_dic[ind]).round(3).reset_index('county',drop = True)
            hazard_ratios.loc[hazard_ratios.rp ==sim,'di0_pc_L'] +=hazard_ratios.loc[hazard_ratios.rp ==sim,'di0_pc_'+ind]
  
    print 'DONE DISTRIBUTING LOSSES'
    hazard_ratios['asset_loss_pc'] = hazard_ratios.asset_loss_tot/hazard_ratios.pcwgt
    hazard_ratios = hazard_ratios.reset_index().set_index(['county','hazard','rp','tract'])
    hazard_ratios['v'] = hazard_ratios.asset_loss_tot/hazard_ratios.str_asset_value_tot
    #if suffix in ('_ins50_15'):
    hazard_ratios['v_ins'] = hazard_ratios.asset_loss_tot_ins/hazard_ratios.str_asset_value_tot
    hazard_ratios['hh_share'] = hazard_ratios['hh_share'] /hazard_ratios.v*hazard_ratios.v_ins
    #hazard_ratios = hazard_ratios.fillna(0)
    
  #%%    
if myCountry =='BA':
    ### Run the recovery optimization
    print('Running hh_reco_rate optimization')
    hazard_ratios= hazard_ratios.reset_index().set_index(['rp','tract'])
    
    _pi = df['avg_prod_k'].mean()
    _rho = df['rho'].mean()
    df_v = hazard_ratios.loc[:,['v','hh_share','asset_loss_tot']] 
    #hazard_ratios['hh_reco_rate'],rate_opt,rate_constrained = optimize_reco_w_physical_constraint(_pi, _rho, myCountry, df_v)
    d_constr_recon_rate = None#get_constr_sector_recon_rate(df_v, myCountry,fault_identifier)
    hazard_ratios['physical_reco_rate'] = get_physical_recon_rate(_pi, _rho, myCountry,df_v,fault_identifier,suffix=suffix)

    ##
    df_opt = hazard_ratios[['v','v_ins','pcinc_tot','k_pc_h','k_pc_str','hh_share','p_rent_pc','c_pc_min','mort_pc']] 
    hazard_ratios['hh_reco_rate'], rate_opt, rate_physical,rate_constr,recovery_constraint = optimize_reco_w_labour_and_physical_constraint(
            _pi, _rho, myCountry,df_opt,ind_names,L_recov_dic,L_pcinc_dic, d_constr_recon_rate,fault_identifier,suffix=suffix)
    # plot the rates
 

    
    
    hazard_ratios.reset_index(inplace=True)
    hazard_ratios.set_index(['county','hazard','rp','tract'],inplace=True)  
    hazard_ratios.to_csv(intermediate+'/hazard_ratios.csv',encoding='utf-8', header=True)
    
    #np.savetxt(intermediate+'/recovery_rates.csv', (np.transpose(rate_cons),np.transpose(rate_physical),np.transpose(rate_c_min),np.transpose(rate_opt)), delimiter=',')
    
    
   
# =============================================================================
#     try: 
#         v_to_reco_rate = pickle.load(open('optimization_libs/'+myCountry+'_v_to_reco_rate_proto.p','rb'))
#     except:
#         print('Was not able to load v to hh_reco_rate library from optimization_libs/'+myCountry+'_v_to_reco_rate.p')
#         v_to_reco_rate = {}
# 
#     try: hazard_ratios['hh_reco_rate'] = hazard_ratios.apply(lambda x:v_to_reco_rate[round(x.v,2)],axis=1)
#     except:
#         for _n, _i in enumerate(hazard_ratios.index):
#         # Just a progress counter
#         if round(_n/len(hazard_ratios.index)*100,2)%1 == 0:
#             print(round(_n/len(hazard_ratios.index)*100,2),'% of way through')
# 
#         _v = round(hazard_ratios.loc[_i,'v'],2)
#         try: hazard_ratios.loc[_i,'hh_reco_rate'] = v_to_reco_rate[_v]
#         except:
#             _opt = optimize_reco(_pi,_rho,_v)
#             hazard_ratios.loc[_i,'hh_reco_rate'] = _opt
#             v_to_reco_rate[_v] = _opt
# 
#     pickle.dump(v_to_reco_rate, open('optimization_libs/'+myCountry+'_v_to_reco_rate.p', 'wb' ) )
# =============================================================================
#%%  
## Write the macro.csv file    
if myCountry =='BA':
    raw_data = pd.read_csv('../../Pre-process - Simply analytics data/Output/INPUT_for_resilience_model_v2.csv')
    macro = copy.deepcopy(df)
    macro['tau_tax'] = 0
    macro['protection'] = 0
    macro['gdp_pc_BA'] = ((cat_info.pcinc_tot*cat_info.pcwgt).sum())/(cat_info.pcwgt.sum())
    for cid in df.index.values:
        macro.loc[cid,'county'] = raw_data[raw_data.county_id ==cid].county_name.unique()[0]
        #macro.loc[cid,'gdp_pc_county'] = ((raw_data[raw_data.county_id ==cid].pcinc*raw_data[raw_data.county_id ==cid].pcwgt).sum())/(raw_data[raw_data.county_id ==cid].pcwgt.sum())
    macro.set_index('county',inplace = True)
    macro['gdp_pc_county']  = (cat_info['pcinc_tot']*cat_info['pcwgt']).sum(level='county')/macro['population']
    macro.to_csv(intermediate+'/macro.csv',encoding='utf-8', header=True)
    #print raw_data.iloc[:,0:4]

if myCountry =='BA':
    raise Exception('End of Bay Area analysis')

