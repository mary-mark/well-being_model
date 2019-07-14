# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:10:43 2018

@author: Mary
"""
# Import packages for data analysis
import matplotlib.pyplot as plt
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

myCountry = 'BA'
suffix = '_baseline' #'_baseline','_retrofit','_no_code'
output_suffix ='_UI_standard'# '_UI_standard','_UI_2xtime','_UI_standard_retrofit'
nsims = 4
intermediate = set_directories(myCountry,suffix,nsims)
fault_identifier = 'Hayward_sc28_6'


#%% Unemployment parameters
#UI_t = 52 * 2 #extende
UI_t_years = 0.5 #standard = 0.5yr , extended = 2yr
UI_t = 26 #standard

#%% Load the general employment data
raw_data = pd.read_csv('../../Pre-process - Simply analytics data/Output/INPUT_for_resilience_model_v2.csv')
raw_data_real_estate = pd.read_csv('../../Pre-process - Simply analytics data/Output/INPUT_real_estate_for_resilience_model.csv')

    
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
        cat_info['peUI_'+ind+'_weekly'] = np.interp(cat_info['peinc_'+ind],
                [900*4,11674*4],[40,450])
        cat_info['peUI_'+ind+'_weekly'] = cat_info['peUI_'+ind+'_weekly'].clip(upper = 450)
        # add up to the total pc labour income (all industries)
        cat_info.pcinc_L += cat_info['pcinc_'+ind]
        
        # round the labour incomes
        cat_info.pcinc_L = cat_info.pcinc_L.round(0)



#%% Create labour income recovery pickle
    
rp = np.arange(1,nsims+1,1)    
x_max = UI_t_years # as many years as there is insurance
x_min, n_steps = 0.,int(52.*x_max) # <-- time step = week
int_dt,step_dt = np.linspace(x_min,x_max,num=n_steps,endpoint=True,retstep=True)
hazard = ['EQ']
hazard_index = pd.MultiIndex.from_product([hazard, rp, cat_info.index.get_level_values(1)],names=['hazard', 'rp','tract'])
df_UI = pd.DataFrame(index = hazard_index)
for t in np.arange(0,n_steps):
    df_UI['UI_tot_Week'+str(int(t))] = 0
cat_info = cat_info.reset_index('county')
            
loaded_list = pickle.load(open(intermediate+'/labour_recovery_'+fault_identifier+'_'+str(nsims)+'sims.p','rb'))
L_pcinc_dic = loaded_list[0]
L_recov_dic = loaded_list[1]
        
#Define the labour losses data frame
labour_loss_df =  pd.DataFrame.from_dict(L_recov_dic,orient = 'index').round(4).reset_index()
labour_loss_df[['industry','rp']] = labour_loss_df['index'].apply(pd.Series)
labour_loss_df = labour_loss_df.drop(columns=['index']).set_index(['rp','industry'])
#labour_loss_df = broadcast_simple(labour_loss_df,cat_info.index.levels[1])
 
index_labels = [list(x) for x in L_pcinc_dic['AGR'].index.levels]
index_labels.pop(0)
index_labels.append(ind_names)

df_UI = df_UI.reset_index(['hazard','rp'])
for sim in np.arange(1,nsims+1):
    print sim
    for ind in ind_names:
        for t in np.arange(0,UI_t):
            df_UI.loc[df_UI.rp == sim,'UI_tot_Week'+str(int(t))] +=labour_loss_df.loc[(sim,ind),t].clip(0)*cat_info['Emp_'+ind] * cat_info['peUI_'+ind+'_weekly']/cat_info.pcwgt
df_UI_pc = df_UI.reset_index().set_index(hazard_index).drop(columns = ['hazard','rp','tract'])
pickle.dump(df_UI_pc, open('optimization_libs/BA_weekly'+output_suffix+'_'+str(nsims)+'sims.p', 'wb' ) )

'''       

        labour_inc_df = pd.DataFrame(index = pd.MultiIndex.from_product(index_labels,names = ['tract','industry'])).reset_index('industry')
        for ind in ind_names:
            labour_inc_df.loc[labour_inc_df.industry == ind,'pcinc_ind'] = L_pcinc_dic[ind].reset_index('county')['pcinc_'+ind]
            labour_inc_df.loc[labour_inc_df.industry == ind,'county'] = L_pcinc_dic[ind].reset_index('county')['county']
        labour_inc_df = broadcast_simple(labour_inc_df,cats_event.index.levels[2]).reset_index().set_index(['rp','industry','tract'])
        labour_event = labour_loss_df.mul(labour_inc_df.pcinc_ind,axis = 'index').groupby(['rp','tract']).agg(np.sum)
        county_names_df =pd.DataFrame( labour_inc_df.county.groupby(['rp','tract']).first(), columns = ['county'])
        labour_event = pd.concat([labour_event,county_names_df],axis = 1).reset_index()
        labour_event['hazard'] = 'EQ'
        labour_event.set_index(['county','hazard','rp','tract'],inplace = True)
        pickle.dump(labour_event, open(intermediate+'/labour_recovery_df_'+fault_identifier+'_'+str(nsims)+'sims.p', 'wb' ))
'''
