##################################
#Import packages for data analysis
from libraries.lib_compute_resilience_and_risk import get_weighted_mean
from libraries.lib_poverty_tables_and_maps import run_poverty_duration_plot, run_poverty_tables_and_maps
from libraries.replace_with_warning import *
from libraries.lib_country_dir import *
from libraries.lib_common_plotting_functions import *
from libraries.maps_lib import *

from libraries.lib_average_over_rp import *

from scipy.stats import norm
import matplotlib.mlab as mlab

from pandas import isnull
import pandas as pd
import numpy as np
import os, time
import sys


font = {'family' : 'sans serif',
    'size'   : 20}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.facecolor'] = 'white'

import warnings
warnings.filterwarnings('always',category=UserWarning)

myCountry = 'PH'
if len(sys.argv) >= 2: myCountry = sys.argv[1]
print('Running '+myCountry)

##################################
# Set directories
model  = os.getcwd() #get current directory
output = model+'/../output_country/'+myCountry+'/'
output_plots = model+'/../output_plots/'+myCountry+'/'

economy = get_economic_unit(myCountry)
event_level = [economy, 'hazard', 'rp']
dem = get_demonym(myCountry)

##################################
# Set policy params
base_str = 'no'
pds1_str = 'unif_poor'
pds2_str = 'unif_poor_only'
pds3_str = 'unif_poor_q12'

if myCountry == 'FJ':
    pds1_str  = 'fiji_SPS'
    pds2_str = 'fiji_SPP'

drm_pov_sign = -1 # toggle subtraction or addition of dK to affected people's incomes
all_policies = []#['_exp095','_exr095','_ew100','_vul070','_vul070r','_rec067']

haz_dict = {'SS':'Storm surge','PF':'Precipitation flood','HU':'Hurricane','EQ':'Earthquake'}

##################################
# Load base and PDS files
iah_base = pd.read_csv(output+'iah_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
df_base = pd.read_csv(output+'results_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp'])
public_costs = pd.read_csv(output+'public_costs_tax_'+base_str+'_.csv', index_col=[economy,'hazard','rp']).reset_index()
try:
    iah = pd.read_csv(output+'iah_tax_'+pds1_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
    df = pd.read_csv(output+'results_tax_'+pds1_str+'_.csv', index_col=[economy,'hazard','rp'])
    macro = pd.read_csv(output+'macro_tax_'+pds1_str+'_.csv', index_col=[economy,'hazard','rp'])
except: pass

#try:
#    iah_noPT = pd.read_csv(output+'iah_tax_'+base_str+'__noPT.csv', index_col=[economy,'hazard','rp','hhid'])
#    # ^ Scenario: no public transfers to rebuild infrastructure
#except: pass

iah_SP2, df_SP2 = None,None
iah_SP3, df_SP3 = None,None
try:
    iah_SP2 = pd.read_csv(output+'iah_tax_'+pds2_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
    df_SP2  = pd.read_csv(output+'results_tax_'+pds2_str+'_.csv', index_col=[economy,'hazard','rp'])

    iah_SP3 = pd.read_csv(output+'iah_tax_'+pds3_str+'_.csv', index_col=[economy,'hazard','rp','hhid'])
    df_SP3  = pd.read_csv(output+'results_tax_'+pds3_str+'_.csv', index_col=[economy,'hazard','rp'])
    print('loaded 2 extra files (secondary SP system for '+myCountry+')')
except: pass

for iPol in all_policies:
    iah_pol = pd.read_csv(output+'iah_tax_'+pds1_str+'_'+iPol+'.csv', index_col=[economy,'hazard','rp','hhid'])
    df_pol  = pd.read_csv(output+'results_tax_'+pds1_str+'_'+iPol+'.csv', index_col=[economy,'hazard','rp'])

    iah['dk0'+iPol] = iah_pol[['dk0','pcwgt']].prod(axis=1)
    iah['dw'+iPol] = iah_pol[['dw','pcwgt']].prod(axis=1)/df_pol.wprime.mean()

    print(iPol,'added to iah (these policies are run *with* PDS)')

    del iah_pol
    del df_pol

##################################
# SAVE OUT SOME RESULTS FILES
df_prov = df[['dKtot','dWtot_currency']].copy()
df_prov['gdp'] = df[['pop','gdp_pc_prov']].prod(axis=1).copy()
results_df = macro.reset_index().set_index([economy,'hazard'])
results_df = results_df.loc[results_df.rp==100,'dk_event'].sum(level='hazard')
results_df = results_df.rename(columns={'dk_event':'dk_event_100'})
results_df = pd.concat([results_df,df_prov.reset_index().set_index([economy,'hazard']).sum(level='hazard')['dKtot']],axis=1,join='inner')
results_df.columns = ['dk_event_100','AAL']
results_df.to_csv(output+'results_table_new.csv')
print('Writing '+output+'results_table_new.csv')

##################################
# Manipulate iah 
# --> use AE, in case that's different from per cap
iah['c_initial']    = (iah[['c','pcwgt']].prod(axis=1)/iah['pcwgt_ae']).fillna(0)
# ^ hh consumption, as reported in HIES

iah['di_pre_reco']  = (iah[['di0','pcwgt']].prod(axis=1)/iah['pcwgt_ae']).fillna(0)
iah['dc_pre_reco']  = (iah[['dc0','pcwgt']].prod(axis=1)/iah['pcwgt_ae']).fillna(0)
# ^ hh income loss (di & dc) immediately after disaster

iah['dc_post_reco'] = (iah[['dc_post_reco','pcwgt']].prod(axis=1)/iah['pcwgt_ae']).fillna(0)
# ^ hh consumption loss (dc) after 10 years of reconstruction

iah['pds_nrh']      = ((iah['pc_fee']-iah['help_received'])*iah['pcwgt']/iah['pcwgt_ae']).fillna(0)
# ^ Net post-disaster support

iah['i_pre_reco']   = (iah['c_initial'] + drm_pov_sign*iah['di_pre_reco'])
iah['c_pre_reco']   = (iah['c_initial'] + drm_pov_sign*iah['dc_pre_reco'])
iah['c_post_reco']  = (iah['c_initial'] + drm_pov_sign*iah['dc_post_reco'])
# ^ income & consumption before & after reconstruction

##################################
# Create additional dfs
#
# Clone index of iah with just one entry/hhid
iah_res = pd.DataFrame(index=(iah.sum(level=[economy,'hazard','rp','hhid'])).index)

## Translate from iah by summing over hh categories [(a,na)x(helped,not_helped)]
# These are special--pcwgt has been distributed among [(a,na)x(helped,not_helped)] categories
iah_res['pcwgt']    =    iah['pcwgt'].sum(level=[economy,'hazard','rp','hhid'])
iah_res['pcwgt_ae'] = iah['pcwgt_ae'].sum(level=[economy,'hazard','rp','hhid'])
iah_res['hhwgt']    =    iah['hhwgt'].sum(level=[economy,'hazard','rp','hhid'])

#These are the same across [(a,na)x(helped,not_helped)] categories 
iah_res['k']         = iah['k'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['c']         = iah['c'].mean(level=[economy,'hazard','rp','hhid'])
#iah_res['c_ae']      = iah['pcinc_ae'].mean(level=[economy,'hazard','rp','hhid'])
#iah_res['hhsize_ae'] = iah['hhsize_ae'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['quintile']  = iah['quintile'].mean(level=[economy,'hazard','rp','hhid'])
iah_res['pov_line']  = iah['pov_line'].mean(level=[economy,'hazard','rp','hhid'])

# Get subsistence line
if get_subsistence_line(myCountry) != None: 
    iah['sub_line'] = get_subsistence_line(myCountry)
    iah_res['sub_line'] = get_subsistence_line(myCountry)

# These need to be averaged across [(a,na)x(helped,not_helped)] categories (weighted by pcwgt)
# ^ values still reported per capita
iah_res['dk0']           = iah[[  'dk0','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dc0']           = iah[[  'dc0','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['help_received'] = iah[['help_received','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['pc_fee']        = iah[['pc_fee','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
iah_res['dc_npv_pre']    = iah[['dc_npv_pre','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']

# These are the other policies (scorecard)
# NB: already weighted by pcwgt from their respective files
for iPol in all_policies:
    print('dk0'+iPol)
    iah_res['dk0'+iPol] = iah['dk0'+iPol].sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']
    iah_res['dw'+iPol] = iah['dw'+iPol].sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt']

# Note that we're pulling dw in from iah_base and  here
iah_res['dw']     = (iah_base[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df.wprime.mean()
iah_res['pds_dw'] = (iah[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df.wprime.mean()
try: iah_res['pds2_dw'] = (iah_SP2[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df_SP2.wprime.mean()
except: pass
try: iah_res['pds3_dw'] = (iah_SP3[['dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt'])/df_SP3.wprime.mean()
except: pass

# Huge file
del iah_base

iah_res['c_initial']   = iah[['c_initial'  ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE
iah_res['di_pre_reco'] = iah[['di_pre_reco','pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # di per AE
iah_res['dc_pre_reco'] = iah[['dc_pre_reco','pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # dc per AE
iah_res['pds_nrh']     = iah[['pds_nrh'    ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # nrh per AE
iah_res['i_pre_reco']  = iah[['i_pre_reco' ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # i pre-reco per AE
iah_res['c_pre_reco']  = iah[['c_pre_reco' ,'pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c pre-reco per AE
iah_res['c_post_reco'] = iah[['c_post_reco','pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c post-reco per AE
#iah_res['c_final_pds'] = iah[['c_final_pds','pcwgt_ae']].prod(axis=1).sum(level=[economy,'hazard','rp','hhid'])/iah_res['pcwgt_ae'] # c per AE

# Calc people who fell into poverty on the regional level for each disaster
iah_res['delta_pov_pre_reco']  = iah.loc[(iah.c_initial > iah.pov_line)&(iah.c_pre_reco <= iah.pov_line),'pcwgt'].sum(level=[economy,'hazard','rp'])
iah_res['delta_pov_post_reco'] = iah.loc[(iah.c_initial > iah.pov_line)&(iah.c_post_reco <= iah.pov_line),'pcwgt'].sum(level=[economy,'hazard','rp'])

iah = iah.reset_index()
iah_res  = iah_res.reset_index().set_index([economy,'hazard','rp','hhid'])

# Save out iah by economic unit
iah_out = pd.DataFrame(index=iah_res.sum(level=[economy,'hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out['Asset risk'+iPol] = iah_res[['dk0'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
    iah_out['Well-being risk'+iPol] = iah_res[['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp']) 

print(iah_out.head(10))

# Add public well-being costs to this output
public_costs_prov_sum = (public_costs.loc[(public_costs.contributer != public_costs[economy])]).reset_index().set_index(event_level).sum(level=event_level)
iah_out['Well-being risk'] += public_costs_prov_sum[['dw','dw_soc']].sum(axis=1)

#iah_out['pds_dw']      = iah_res[['pds_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pc_fee']      = iah_res[['pc_fee','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds2_dw'] = iah_res[['pds2_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
#iah_out['pds3_dw'] = iah_res[['pds3_dw','pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])
iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
iah_out.to_csv(output+'geo_sums.csv')

iah_out,_ = average_over_rp(iah_out)

iah_out['SE capacity']  = iah_out['Asset risk']/iah_out['Well-being risk']
iah_out.to_csv(output+'geo_haz_aal_sums.csv')

_ = (iah_out['Asset risk']/1.E6).round(1).unstack().copy()

to_usd = get_currency(myCountry)[2]

if len(_.columns) > 1:
    _['Total'] = _.sum(axis=1)
else:
    _['usd'] = (_*to_usd*1.E3).round(1)

_.loc['Total'] = _.sum()
_.sort_values('PF',ascending=False).to_latex('latex/reg_haz_asset_risk.tex')

iah_out = iah_out.sum(level=economy)

print(iah_out.head())
iah_out[['Asset risk','Well-being risk']]*=to_usd/1.E3 # iah_out is thousands [1E3]

iah_out.loc['Total'] = [float(iah_out['Asset risk'].sum()),
                        float(iah_out['Well-being risk'].sum()),
                        float(iah_out['Asset risk'].sum()/iah_out['Well-being risk'].sum())]
iah_out['SE capacity']  = 100.*iah_out['Asset risk']/iah_out['Well-being risk']

iah_out.to_csv(output+'geo_aal_sums.csv')

iah_out[['Asset risk','SE capacity','Well-being risk']].sort_values(['Well-being risk'],ascending=False).astype(int).to_latex('latex/geo_aal_sums.tex')
print('Wrote latex! Sums:\n',iah_out[['Asset risk','Well-being risk']].sum())

# Save out iah by economic unit, *only for poorest quintile*
iah_out_q1 = pd.DataFrame(index=iah_res.sum(level=[economy,'hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out_q1['Asset risk'+iPol] = iah_res.loc[(iah_res.quintile==1),['dk0'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])*to_usd/1.E3
    iah_out_q1['Well-being risk'+iPol] = iah_res.loc[(iah_res.quintile==1),['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=[economy,'hazard','rp'])*to_usd/1.E3

iah_out_q1.to_csv(output+'geo_sums_q1.csv')
iah_out_q1,_ = average_over_rp(iah_out_q1,'default_rp')
iah_out_q1['SE capacity']  = iah_out_q1['Asset risk']/iah_out_q1['Well-being risk']

iah_out_q1.to_csv(output+'geo_haz_aal_sums_q1.csv')

iah_out_q1 = iah_out_q1.sum(level=economy)
iah_out_q1.loc['Total'] = [float(iah_out_q1['Asset risk'].sum()),
                           float(iah_out_q1['Well-being risk'].sum()),
                           float(iah_out_q1['Asset risk'].sum()/iah_out_q1['Well-being risk'].sum())]
iah_out_q1['SE capacity']  = iah_out_q1['Asset risk']/iah_out_q1['Well-being risk']

iah_out_q1.to_csv(output+'geo_aal_sums_q1.csv')

iah_out_q1['% total RA'] = 100.*(iah_out_q1['Asset risk']/iah_out['Asset risk'])
iah_out_q1['% total RW'] = 100.*(iah_out_q1['Well-being risk']/iah_out['Well-being risk'])
iah_out_q1[['SE capacity']]*=100.
iah_out_q1[['Asset risk','% total RA','SE capacity','Well-being risk','% total RW']].sort_values(['Well-being risk'],ascending=False).astype(int).to_latex('latex/geo_aal_sums_q1.tex',bold_rows=True)
print('Wrote latex! Q1 sums: ',iah_out_q1.sum())

#_grdp = pd.read_csv('../intermediate/'+myCountry+'/gdp.csv')
#iah_out_q1['pop']  = iah_res['pcwgt'].sum(level=event_level).mean(level=event_level[0])/1.E6
#iah_out_q1['grdp'] = iah_res[['pcwgt','c']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])/1.E6

iah_out_q1['pop_q1']  = iah_res.loc[iah_res.quintile==1,'pcwgt'].sum(level=event_level).mean(level=event_level[0])
iah_out_q1['grdp_q1'] = iah_res.loc[iah_res.quintile==1,['pcwgt','c']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])

#iah_out_q1['reg_wprime'] = (iah_res[['pcwgt','c']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])
#                            /iah_res['pcwgt'].sum(level=event_level).mean(level=event_level[0]))**(-1.5)

#iah_out_q1['reg_q1_wprime'] = (iah_res.loc[iah_res.quintile==1,['pcwgt','c']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])
#                               /iah_res.loc[iah_res.quintile==1,'pcwgt'].sum(level=event_level).mean(level=event_level[0]))**(-1.5)

_ = iah_out_q1.drop('Total',axis=0)[['pop_q1','Asset risk','Well-being risk']].copy()
_[['Asset risk','Well-being risk']]/=to_usd
_['Asset risk pc'] = iah_out_q1['Asset risk']*1.E3/(to_usd*iah_out_q1['pop_q1'])
_['Well-being risk pc'] = iah_out_q1['Well-being risk']*1.E3/(to_usd*iah_out_q1['pop_q1'])
_.loc['Total'] = [_['pop_q1'].sum(),
                  _['Asset risk'].sum(),
                  _['Well-being risk'].sum(),
                  _['Asset risk'].sum()*1.E3/_['pop_q1'].sum(),
                  _['Well-being risk'].sum()*1.E3/_['pop_q1'].sum()]
print(_)

_[['pop_q1','Asset risk pc','Well-being risk pc']].round(0).astype(int).sort_values('Well-being risk pc',ascending=False).to_latex('latex/risk_q1.tex')
_.to_csv('tmp/q1_figs.csv')

# Save out iah
iah_out = pd.DataFrame(index=iah_res.sum(level=['hazard','rp']).index)
for iPol in ['']+all_policies:
    iah_out['dk0'+iPol] = iah_res[['dk0'+iPol,'pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
    iah_out['dw'+iPol] = iah_res[['dw'+iPol,'pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
iah_out['pds_dw'] = iah_res[['pds_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
iah_out['pc_fee'] = iah_res[['pc_fee','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
try: iah_out['pds2_dw'] = iah_res[['pds2_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
except: pass
try: iah_out['pds3_dw'] = iah_res[['pds3_dw','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
except: pass

iah_out.to_csv(output+'haz_sums.csv')
print(iah_out.head(10))
iah_out,_ = average_over_rp(iah_out,'default_rp')
iah_out.to_csv(output+'sums.csv')

# Clone index of iah at national level
iah_ntl = pd.DataFrame(index=(iah_res.sum(level=['hazard','rp'])).index)
#
iah_ntl['pop'] = iah_res.pcwgt.sum(level=['hazard','rp'])
iah_ntl['pov_pc_i'] = iah_res.loc[(iah_res.c_initial <= iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_hh_i'] = iah_res.loc[(iah_res.c_initial <= iah_res.pov_line),'hhwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_pc_f'] = iah_res.loc[(iah_res.c_pre_reco <= iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_hh_f'] = iah_res.loc[(iah_res.c_pre_reco <= iah_res.pov_line),'hhwgt'].sum(level=['hazard','rp'])
iah_ntl['pov_pc_D'] = iah_ntl['pov_pc_f'] - iah_ntl['pov_pc_i']
iah_ntl['pov_hh_D'] = iah_ntl['pov_hh_f'] - iah_ntl['pov_hh_i']
#iah_ntl['pov_pc_pds_f'] = iah_res.loc[(iah_res.c_final_pds < iah_res.pov_line),'pcwgt'].sum(level=['hazard','rp'])
#iah_ntl['pov_hh_pds_f'] = iah_res.loc[(iah_res.c_final_pds < iah_res.pov_line),'hhwgt'].sum(level=['hazard','rp'])
#iah_ntl['pov_pc_pds_D'] = iah_ntl['pov_pc_pds_f'] - iah_ntl['pov_pc_i']
#iah_ntl['pov_hh_pds_D'] = iah_ntl['pov_hh_pds_f'] - iah_ntl['pov_hh_i']
print('\n\nInitial poverty incidence:\n',iah_ntl[['pov_pc_i','pov_hh_i']].mean())
print('--> In case of SL: THIS IS NOT RIGHT! Maybe because of the 3 provinces that dropped off?')
#iah_ntl[['pov_pc_i','pov_hh_i']].to_csv('~/Desktop/my_out.csv')

#iah_ntl['eff_pds'] = iah_ntl['pov_pc_pds_D'] - iah_ntl['pov_pc_D']

# Print out plots for iah_res
iah_res = iah_res.reset_index()
iah_ntl = iah_ntl.reset_index()

#########################
# Save out
iah_ntl.to_csv(output+'poverty_ntl_by_haz.csv')
iah_ntl = iah_ntl.reset_index().set_index(['hazard','rp'])
iah_ntl_haz,_ = average_over_rp(iah_ntl,'default_rp')
iah_ntl_haz.sum(level='hazard').to_csv(output+'poverty_haz_sum.csv')

iah_ntl = iah_ntl.reset_index().set_index('rp').sum(level='rp')
iah_ntl.to_csv(output+'poverty_ntl.csv')
iah_sum,_ = average_over_rp(iah_ntl,'default_rp')
iah_sum.sum().to_csv(output+'poverty_sum.csv')
assert(False)
##########################

myHaz = None
if myCountry == 'FJ': myHaz = [['Ba','Lau','Tailevu'],get_all_hazards(myCountry,iah_res),[1,10,100,500,1000]]
elif myCountry == 'PH': myHaz = [['II - Cagayan Valley','NCR','IVA - CALABARZON'],get_all_hazards(myCountry,iah_res),get_all_rps(myCountry,iah_res)]
elif myCountry == 'SL': myHaz = [['Ampara','Colombo','Rathnapura'],get_all_hazards(myCountry,iah_res),get_all_rps(myCountry,iah_res)]

##################################################################
# This code generates the histograms showing income before & after disaster
# ^ this is nationally, so we'll use iah
upper_clip = 1.2E5
scale_fac = 1.0
if myCountry == 'FJ': 
    scale_fac = 2.321208
    upper_clip = 2E4
if myCountry == 'SL': upper_clip = 4.0E5

#run_poverty_duration_plot(myCountry)
#run_poverty_tables_and_maps(myCountry,iah.reset_index().set_index(event_level),event_level)

simple_plot = True
for aReg in myHaz[0]:
    for aDis in get_all_hazards(myCountry,iah):

        try: plt.close('all')
        except: pass
        
        myC_ylim = None
        c_bins = [None,50]
        for anRP in myHaz[2][::-1]:        

            ax=plt.gca()

            plt.xlim(0,upper_clip)
            mny = get_currency(myCountry)
            plt.xlabel(r'Income ['+mny[0].replace('b. ','')+' per person, per year]')
            plt.ylabel('Population'+get_pop_scale_fac(myCountry)[1])
            plt.title(str(anRP)+'-year '+haz_dict[aDis].lower()+' event in region '+aReg)

            # Income dist immediately after disaster
            cf_heights, cf_bins = np.histogram((iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'c_pre_reco']/scale_fac).clip(upper=upper_clip), bins=c_bins[1],
                                               weights=iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt']/get_pop_scale_fac(myCountry)[0])
            if c_bins[0] == None: c_bins = [cf_bins,cf_bins]
            
            # Income dist before disaster
            ci_heights, _bins = np.histogram((iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'c_initial']/scale_fac).clip(upper=upper_clip), bins=c_bins[1],
                                             weights=iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt']/get_pop_scale_fac(myCountry)[0])

            # Income dist after reconstruction
            cf_reco_hgt, _bins = np.histogram((iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'c_post_reco']/scale_fac).clip(upper=upper_clip), bins=c_bins[1],
                                              weights=iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt']/get_pop_scale_fac(myCountry)[0])
            
            ax.step(c_bins[1][1:], ci_heights, label=aReg+' - FIES income', linewidth=1.2,color=greys_pal[8])
            leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)             
            ax.get_figure().savefig(output_plots+'npr_poverty_k_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'_1of3.pdf',format='pdf')

            ax.bar(c_bins[1][:-1], cf_heights, width=(c_bins[1][1]-c_bins[1][0]), label=aReg+' - post-disaster', facecolor=q_colors[0],edgecolor=q_colors[0],alpha=0.45)
            leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            ax.get_figure().savefig(output_plots+'npr_poverty_k_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'_2of3.pdf',format='pdf')

            #ax.bar(c_bins[1][:-1], cf_reco_hgt, width=(c_bins[1][1]-c_bins[1][0]), label=aReg+' - post-reconstruction', facecolor=q_colors[2],edgecolor=q_colors[2],alpha=0.45)
            #ax.step(c_bins[1][1:], ci_heights, label=aReg+' - FIES income', linewidth=1.2,color=greys_pal[8])            

            #if myC_ylim == None: myC_ylim = ax.get_ylim()
            #plt.ylim(myC_ylim[0],2.5*myC_ylim[1])
            # ^ Need this for movie making, but better to let the plot limits float if not

            leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            leg.get_frame().set_color('white')
            leg.get_frame().set_edgecolor(greys_pal[7])
            leg.get_frame().set_linewidth(0.2)

            if not simple_plot:
                ax.annotate('Total asset losses: '+str(round(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),['pcwgt','dk0']].prod(axis=1).sum()/mny[1],1))+mny[0],
                            xy=(0.03,-0.18), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
                ax.annotate('Reg. well-being losses: '+str(round(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),['pcwgt','dw']].prod(axis=1).sum()/(df.wprime.mean()*mny[1]),1))+mny[0],
                            xy=(0.03,-0.50), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
                ax.annotate('Natl. liability: '+str(round(float(public_costs.loc[(public_costs.contributer!=aReg)&(public_costs[economy]==aReg)&(public_costs.hazard==aDis)&(public_costs.rp==anRP),['transfer_pub']].sum()*1.E3/mny[1]),1))+mny[0],
                            xy=(0.03,-0.92), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
                ax.annotate('Natl. well-being losses: '+str(round(float(public_costs.loc[(public_costs.contributer!=aReg)&(public_costs[economy]==aReg)&(public_costs.hazard==aDis)&(public_costs.rp==anRP),['dw','dw_soc']].sum(axis=1).sum()*1.E3/mny[1]),1))+mny[0].replace('b','m'),
                            xy=(0.03,-1.24), xycoords=leg.get_frame(),size=8,va='top',ha='left',annotation_clip=False,zorder=100)
                
            try:
                new_pov_c = int(iah.loc[iah.eval('region==@aReg & hazard==@aDis & rp==@anRP & c_initial>pov_line & c_pre_reco>=sub_line & c_pre_reco<pov_line'),'pcwgt'].sum())
                new_pov_i = int(iah.loc[iah.eval('region==@aReg & hazard==@aDis & rp==@anRP & c_initial>pov_line & c_pre_reco>=sub_line & c_pre_reco<pov_line'),'pcwgt'].sum())
            except:
                new_pov_c = int(iah.loc[iah.eval('district==@aReg & hazard==@aDis & rp==@anRP & c_initial>pov_line & c_pre_reco>=sub_line & c_pre_reco<pov_line'),'pcwgt'].sum())
                new_pov_i = int(iah.loc[iah.eval('district==@aReg & hazard==@aDis & rp==@anRP & c_initial>pov_line & c_pre_reco>=sub_line & c_pre_reco<pov_line'),'pcwgt'].sum())

            new_pov = new_pov_i
            print('c:',new_pov_c,' i:',new_pov_i)

            try: new_pov_pct = round(100.*float(new_pov)/float(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt'].sum()),1)
            except: new_pov_pct = 0

            plt.plot([iah.pov_line.mean()/scale_fac,iah.pov_line.mean()/scale_fac],[0,1.21*cf_heights[:-2].max()],'k-',lw=2.5,color=greys_pal[7],zorder=100,alpha=0.85)
            ax.annotate('Poverty line',xy=(1.1*iah.pov_line.mean()/scale_fac,1.21*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
            ax.annotate(r'$\Delta$N$_p$ = +'+int_w_commas(new_pov)+' ('+str(new_pov_pct)+'% of population)',
                        xy=(1.1*iah.pov_line.mean()/scale_fac,1.14*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False)

            sub_line, new_sub = get_subsistence_line(myCountry), None
            if sub_line is not None:
                new_sub = int(iah.loc[((iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP)
                                           &(iah.c_initial > sub_line)&(iah.c_pre_reco <= sub_line)),'pcwgt'].sum())
                new_sub_pct = round(100.*float(new_sub)/float(iah.loc[(iah[economy]==aReg)&(iah.hazard==aDis)&(iah.rp==anRP),'pcwgt'].sum()),1)
                plt.plot([sub_line,sub_line],[0,1.41*cf_heights[:-2].max()],'k-',lw=2.5,color=greys_pal[7],zorder=100,alpha=0.85)
                ax.annotate('Subsistence line',xy=(1.1*sub_line,1.41*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
                ax.annotate(r'$\Delta$N$_s$ = +'+int_w_commas(new_sub)+' ('+str(new_sub_pct)+'% of population)',
                            xy=(1.1*sub_line,1.34*cf_heights[:-2].max()),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False)

            print(aReg,aDis,anRP,new_pov,'people into poverty &',new_sub,'into subsistence') 

            fig = ax.get_figure()
            fig.savefig(output_plots+'npr_poverty_k_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')
            #fig.savefig(output_plots+'png/npr_poverty_k_'+aReg.replace(' ','').replace('-','')+'_'+aDis+'_'+str(anRP)+'.png',format='png')
            plt.clf()
            plt.close('all')
            print(aReg+'_poverty_k_'+aDis+'_'+str(anRP)+'.pdf')
            

##################################################################
# This code generates the histograms including [k,dk,dc,dw,&pds]
# ^ this is by province, so it will use iah_res
for aProv in myHaz[0]:
    for aDis in myHaz[1]:
        for anRP in myHaz[2]:

            plt.figure(1)
            ax = plt.subplot(111)

            plt.figure(2)
            ax2 = plt.subplot(111)

            for myQ in range(1,6): #nQuintiles
    
                print(aProv,aDis,anRP,'shape:',iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].shape[0])
                
                k = (0.01*iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['k','pcwgt']].prod(axis=1).sum()/
                     iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dk = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dk0','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dc = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dc_npv_pre','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['dw','pcwgt']].prod(axis=1).sum()/
                      iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                
                pds_nrh = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_nrh','pcwgt']].prod(axis=1).sum()/
                           iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

                pds_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds_dw','pcwgt']].prod(axis=1).sum()/
                          iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())

                try:
                    pds2_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds2_dw','pcwgt']].prod(axis=1).sum()/
                                   iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                except: pds2_dw = 0

                try:
                    pds3_dw = (iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),['pds3_dw','pcwgt']].prod(axis=1).sum()/
                                   iah_res.loc[(iah_res[economy]==aProv)&(iah_res.hazard==aDis)&(iah_res.rp==anRP)&(iah_res.quintile==myQ),'pcwgt'].sum())
                except: pds3_dw = 0

                ax.bar([7*ii+myQ for ii in range(1,7)],[dk,dc,dw,pds_nrh,pds_dw,pds2_dw],
                       color=q_colors[myQ-1],alpha=0.7,label=q_labels[myQ-1])

                #np.savetxt('/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/PH/dk_dc_dw_pds_'+aProv+'_'+aDis+'_'+str(anRP)+'_Q'+str(myQ)+'.csv',[dk,dc,dw,pds_nrh,pds_dw],delimiter=',')

                lbl= None
                if myQ==1: 
                    ax2.bar([0],[0],color=[q_colors[0]],alpha=0.7,label='No post-disaster support')
                    ax2.bar([0],[0],color=[q_colors[2]],alpha=0.7,label='80% of avg Q1 losses covered for Q1')
                    ax2.bar([0],[0],color=[q_colors[3]],alpha=0.7,label='80% of avg Q1 losses covered for Q1-Q2')
                    ax2.bar([0],[0],color=[q_colors[1]],alpha=0.7,label='80% of avg Q1 losses covered for Q1-Q5')
                ax2.bar([5*myQ+ii for ii in range(0,4)],[dw,pds2_dw,pds3_dw,pds_dw],color=[q_colors[0],q_colors[1],q_colors[2],q_colors[3]],alpha=0.7)
                
                #np.savetxt('/Users/brian/Desktop/BANK/hh_resilience_model/output_plots/PH/pds_comparison_'+aProv+'_'+aDis+'_'+str(anRP)+'_Q'+str(myQ)+'.csv',[dw,pds_dw,pds2_dw], delimiter=',')
                
            out_str = None
            if myCountry == 'FJ': out_str = ['Asset loss','Consumption\nloss (NPV)','Well-being\nloss','Net cost of\nWinston-like\nsupport','Well-being loss\npost support']
            else: out_str = ['Asset loss','Consumption\nloss','Well-being\nloss','Net cost of\nsupport','Well-being loss\npost support']

            for ni, ii in enumerate(range(1,6)):
                ax.annotate(out_str[ni],xy=(6*ii+1,ax.get_ylim()[0]/4.),xycoords='data',ha='left',va='top',weight='bold',fontsize=9,annotation_clip=False)

            fig = ax.get_figure()    
            leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            leg.get_frame().set_color('white')
            leg.get_frame().set_edgecolor(greys_pal[7])
            leg.get_frame().set_linewidth(0.2)
            
            plt.figure(1)
            plt.plot([xlim for xlim in ax.get_xlim()],[0,0],'k-',lw=0.50,color=greys_pal[7],zorder=100,alpha=0.85)
            ax.xaxis.set_ticks([])
            plt.ylabel('Disaster losses ['+get_currency(myCountry)[0][3:]+' per capita]')

            print('losses_k_'+aDis+'_'+str(anRP)+'.pdf')
            fig.savefig(output_plots+'npr_'+aProv+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')#+'.pdf',format='pdf')
            fig.savefig(output_plots+'png/npr_'+aProv+'_'+aDis+'_'+str(anRP)+'.png',format='png')

            plt.figure(2)
            fig2 = ax2.get_figure()
            leg = ax2.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
            leg.get_frame().set_color('white')
            leg.get_frame().set_edgecolor(greys_pal[7])
            leg.get_frame().set_linewidth(0.2)

            ann_y = -ax2.get_ylim()[1]/30

            n_pds_options = 4
            out_str = ['Q1','Q2','Q3','Q4','Q5']
            for ni, ii in enumerate(range(1,6)): # quintiles
                ax2.annotate(out_str[ni],xy=(5*ii+0.05,ann_y),zorder=100,xycoords='data',
                             ha='left',va='center',weight='bold',fontsize=9,annotation_clip=False)
                plt.plot([5*ii+1.20,5*ii+3.78],[ann_y,ann_y],'k-',lw=1.50,color=greys_pal[7],zorder=100,alpha=0.85)
                plt.plot([5*ii+3.78,5*ii+3.78],[ann_y*0.9,ann_y*1.1],'k-',lw=1.50,color=greys_pal[7],zorder=100,alpha=0.85)

            ax2.xaxis.set_ticks([])
            plt.xlim(3,32)
            plt.plot([i for i in ax2.get_xlim()],[0,0],'k-',lw=1.5,color=greys_pal[7],zorder=100,alpha=0.85)
            plt.ylabel('Well-being losses ['+get_currency(myCountry)[0][3:]+' per capita]')
            fig2.savefig(output_plots+'npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'.pdf',format='pdf')
            fig2.savefig(output_plots+'png/npr_pds_schemes_'+aProv+'_'+aDis+'_'+str(anRP)+'.png',format='png')
        
            plt.clf()
            plt.close('all')
