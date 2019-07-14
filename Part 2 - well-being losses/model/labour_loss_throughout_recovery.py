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
nsims = 50
suffix = '_baseline'
intermediate = set_directories(myCountry,suffix,nsims)
fault_identifier = 'Hayward_sc28_6'

labour_event = pickle.load(open(intermediate+'/labour_recovery_df_'+fault_identifier+'_'+str(nsims)+'sims.p','rb'))

#%%
df_di_total =pd.DataFrame(index = labour_event.index)
df_di_total["di_labour_pc"] = 0
dt = 1./52
for t in np.arange(0,520):
    print t
    df_di_total["di_labour_pc"] += labour_event.loc[:,t]*dt

#Save to CSV instead of pickle
df_di_total.to_csv('../post process results/input data/labour_losses_df_'+fault_identifier+suffix+'_'+str(nsims)+'sims.csv')
#pickle.dump(df_di_total, open('../post process results/input data/labour_losses_df_'+fault_identifier+suffix+'_'+str(nsims)+'sims.p', 'wb' ))
        