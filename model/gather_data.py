# This script provides data input for the resilience indicator multihazard model for the Philippines, Fiji, Sri Lanka, and (eventually) Malawi. 
# Restructured from the global model and developed by Jinqiang Chen and Brian Walsh

# Magic
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import packages for data analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import isnull
import os, time
import warnings
import sys
import pickle

from libraries.lib_asset_info import *
from libraries.lib_country_dir import *
from libraries.lib_gather_data import *
from libraries.lib_sea_level_rise import *
from libraries.replace_with_warning import *
from libraries.lib_agents import optimize_reco

warnings.filterwarnings('always',category=UserWarning)

if len(sys.argv) < 2:
    print('Need to list country. Currently implemented: MW, PH, FJ, SL')
    myCountry = 'SL'
else: myCountry = 'SL'#sys.argv[1]

# Set up directories/tell code where to look for inputs & where to save outputs
intermediate = set_directories(myCountry)

# Administrative unit (eg region or province)
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
if myCountry =='FJ': inc_sf = (4.632E9/0.48) # GDP in USD (2016, WDI) -> FJD
cat_info = load_survey_data(myCountry,inc_sf)

print('Survey population:',cat_info.pcwgt.sum())


if myCountry == 'PH':
    get_hhid_FIES(cat_info)
    cat_info = cat_info.rename(columns={'w_prov':'province','w_regn':'region'}).reset_index()
    cat_info['province'].replace(prov_code,inplace=True)     
    cat_info['region'].replace(region_code,inplace=True)
    cat_info = cat_info.reset_index().set_index(economy).drop(['index','level_0'],axis=1)

    # There's no region info in df--put that in...
    df = df.reset_index().set_index('province')
    cat_info = cat_info.reset_index().set_index('province')
    df['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region

    try: df.reset_index()[['province','region']].to_csv('../inputs/PH/prov_to_reg_dict.csv',header=True)
    except: print('Could not update regional-provincial dict')

    df = df.reset_index().set_index(economy)

    df['psa_pop'] = df.sum(level=economy)
    df = df.mean(level=economy)

    # There's no region info in df2--put that in...
    #df2 = df2.reset_index().set_index('province')
    #df2['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region
    #df2 = df2.reset_index().set_index(economy)    

    #df2['gdp_pc_pp'] = df2[['gdp_pc_pp','pop']].prod(axis=1).sum(level=economy)/df2['pop'].sum(level=economy)

    #df2['pop'] = df2['pop'].sum(level=economy)
    #df2['gdp_pp'] = df2['gdp_pp'].sum(level=economy)
    #df2 = df2.mean(level=economy)

    cat_info = cat_info.reset_index().set_index(economy)

if myCountry == 'SL':
    df = df.reset_index()
    df['district'].replace(prov_code,inplace=True)
    df = df.reset_index().set_index(economy).drop(['index'],axis=1)

    cat_info = cat_info.reset_index()
    cat_info['district'].replace(prov_code,inplace=True) #replace district code with its name
    cat_info = cat_info.reset_index().set_index(economy).drop(['index'],axis=1)

# Define per capita income (in local currency)
df['gdp_pc_prov'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info['pcwgt'].sum(level=economy)
df['gdp_pc_nat'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info['pcwgt'].sum()
# ^ this is per capita income

df['pop'] = cat_info.pcwgt.sum(level=economy)

if myCountry == 'PH':
    df['pct_diff'] = 100.*(df['psa_pop']-df['pop'])/df['pop']

#Vulnerability
print('Getting vulnerabilities')
vul_curve = get_vul_curve(myCountry,'wall')
for thecat in vul_curve.desc.unique():

    if myCountry == 'PH': cat_info.ix[cat_info.walls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values
    if myCountry == 'FJ': cat_info.ix[cat_info.Constructionofouterwalls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values
    # Fiji doesn't have info on roofing, but it does have info on the condition of outer walls. Include that as a multiplier?
    if myCountry == 'SL': cat_info.ix[cat_info.walls.values == thecat,'v'] = vul_curve.loc[vul_curve.desc.values == thecat].v.values

# Get roofing data (but Fiji doesn't have this info)
if myCountry != 'FJ':
    print('Getting roof info')
    vul_curve = get_vul_curve(myCountry,'roof')
    for thecat in vul_curve.desc.unique():
        cat_info.ix[cat_info.roof.values == thecat,'v'] += vul_curve.loc[vul_curve.desc.values == thecat].v.values
    cat_info.v = cat_info.v/2

print('Setting c to pcinc') 
cat_info['c'] = cat_info['pcinc']
cat_info['pcsoc'] = cat_info['pcsoc'].clip(upper=0.99*cat_info['pcinc'])
# --> What's the difference between income & consumption/disbursements?
# --> totdis = 'total family disbursements'    
# --> totex = 'total family expenditures'
# --> pcinc_s seems to be what they use to calculate poverty...
# --> can be converted to pcinc_ppp11 by dividing by (365*21.1782)

# Cash receipts, abroad & domestic, other gifts
cat_info['social'] = (cat_info['pcsoc']/cat_info['pcinc']).fillna(0)#.clip(upper=0.99)
# --> All of this is selected & defined in lib_country_dir
# --> Excluding international remittances ('cash_abroad')

print('Getting pov line')
cat_info = cat_info.reset_index().set_index('hhid')
try: 
    cat_info.loc[cat_info.Sector=='Urban','pov_line'] = get_poverty_line(myCountry,'Urban')
    cat_info.loc[cat_info.Sector=='Rural','pov_line'] = get_poverty_line(myCountry,'Rural')
    cat_info['sub_line'] = get_subsistence_line(myCountry)
except: 
    try:
        cat_info['pov_line'] = get_poverty_line(myCountry)
        cat_info['sub_line'] = get_subsistence_line(myCountry)        
    except: 
        cat_info['pov_line'] = get_poverty_line(myCountry)
cat_info = cat_info.reset_index().set_index(event_level[0])

print('Total population:',int(cat_info.pcwgt.sum()))
print('Total n households:',int(cat_info.hhwgt.sum()))

print('\nae',cat_info[['pcinc_ae','pcwgt_ae']].prod(axis=1).sum()/cat_info[['pcwgt_ae']].sum())
print('-',cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info[['pcwgt']].sum())

print('--> Individuals in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc_ae <= cat_info.pov_line),'pcwgt'].sum()/1.E6,3)),'million')
print('-----> Families in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc_ae <= cat_info.pov_line),'hhwgt'].sum()/1.E6,3)),'million')

try:
    print('------> Individuals in poverty (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc_ae<=pov_line & pcinc_ae>sub_line'),'pcwgt'].sum()/1E6,3)),'million')
    print('---------> Families in poverty (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc_ae<=pov_line & pcinc_ae>sub_line'),'hhwgt'].sum()/1E6,3)),'million')
    print('--> Individuals in subsistence (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc_ae<=sub_line'),'pcwgt'].sum()/1E6,3)),'million')
    print('-----> Families in subsistence (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc_ae<=sub_line'),'hhwgt'].sum()/1E6,3)),'million')
except: print('No subsistence info...')

print('\n--> Number in poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1),'pcwgt'].sum()/1E6,3)),'million')
print('--> Poverty rate (flagged poor):',round(100.*cat_info.loc[(cat_info.ispoor==1),'pcwgt'].sum()/cat_info['pcwgt'].sum(),1),'%\n\n\n')
pd.DataFrame({'population':cat_info['pcwgt'].sum(level=economy),
              'nPoor':cat_info.loc[cat_info.ispoor==1,'pcwgt'].sum(level=economy),
              'n_pov':cat_info.loc[cat_info.eval('pcinc_ae<=pov_line & pcinc_ae>sub_line'),'pcwgt'].sum(level=economy),
              'n_sub':cat_info.loc[cat_info.eval('pcinc_ae<=sub_line'),'pcwgt'].sum(level=economy),
              'pctPoor':100.*cat_info.loc[cat_info.ispoor==1,'pcwgt'].sum(level=economy)/cat_info['pcwgt'].sum(level=economy)}).to_csv('../output_country/'+myCountry+'/poverty_rate.csv')
# Could also look at urban/rural if we have that divide

# Change the name: district to code, and create an multi-level index 
cat_info = cat_info.rename(columns={'district':'code','HHID':'hhid'})

# Assing weighted household consumption to quintiles within each province
print('Finding quintiles')
listofquintiles=np.arange(0.20, 1.01, 0.20)
cat_info = cat_info.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),listofquintiles),'quintile'))
# 'c_5_nat' is the upper consumption limit for the lowest 5% throughout country
percentiles_05 = np.arange(0.05, 1.01, 0.05) #create a list of deciles 
my_c5 = match_percentiles(cat_info,perc_with_spline(reshape_data(cat_info.c),reshape_data(cat_info.pcwgt),percentiles_05),'pctle_05_nat')
cat_info['c_5_nat'] = cat_info.ix[cat_info.pctle_05_nat==1,'c'].max()

cat_info = cat_info.reset_index().groupby(economy,sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.c),reshape_data(x.pcwgt),percentiles_05),'pctle_05'))
if 'level_0' in cat_info.columns:
    cat_info = cat_info.drop(['level_0','index'],axis=1)
cat_info_c_5 = cat_info.reset_index().groupby(economy,sort=True).apply(lambda x:x.ix[x.pctle_05==1,'c'].max())
cat_info = cat_info.reset_index().set_index([economy,'hhid']) #change the name: district to code, and create an multi-level index 
cat_info['c_5'] = broadcast_simple(cat_info_c_5,cat_info.index)
cat_info['c_5'] = cat_info.c_5.fillna(cat_info.c_5.mean(level=economy).min())
# ^ this is a line to prevent c_5 from being left empty due to paucity of hh from a given province (for Rotuma, FJ)

cat_info.drop([icol for icol in ['level_0','index','pctle_05','pctle_05_nat'] if icol in cat_info.columns],axis=1,inplace=True)

# Calculate total value of social as fraction of total C
print('Get the tax used for domestic social transfer')
df['tau_tax'] = cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum()/cat_info[['c','pcwgt']].prod(axis=1, skipna=False).sum()

# Fraction of social that goes to each hh
print('Get the share of Social Protection')
cat_info['gamma_SP'] = cat_info[['social','c']].prod(axis=1,skipna=False)*cat_info['pcwgt'].sum()/cat_info[['social','c','pcwgt']].prod(axis=1, skipna=False).sum()

# Calculate K from C
print('Calculating capital from income')
cat_info['k'] = ((cat_info['c']/df['avg_prod_k'])*((1-cat_info['social'])/(1-df['tau_tax']))).clip(lower=0.)

if myCountry == 'FJ':
    #replace division codes with names
    df = df.reset_index()
    df['Division'].replace(prov_code,inplace=True)
    df = df.reset_index().set_index(['Division']).drop(['index'],axis=1)

    cat_info = cat_info.reset_index()
    cat_info['Division'].replace(prov_code,inplace=True) # replace division code with its name
    cat_info = cat_info.reset_index().set_index(['Division','hhid']).drop(['index'],axis=1)

# Shouldn't be losing anything here 
print('Check total population:',cat_info.pcwgt.sum())
cat_info.dropna(inplace=True,how='all')
print('Check total population (after dropna):',cat_info.pcwgt.sum())

# Exposure
cat_info.fillna(0,inplace=True)

# Cleanup dfs for writing out
cat_info_col = [economy,'province','hhid','region','pcwgt','pcwgt_ae','hhwgt','code','np','score','v','c','pcsoc','social','c_5','hhsize',
                'hhsize_ae','gamma_SP','k','quintile','ispoor','pcinc','pcinc_ae','pov_line','SP_FAP','SP_CPP','SP_SPS','nOlds','has_ew',
                'SP_PBS','SP_FNPF','SPP_core','SPP_add','axfin']
cat_info = cat_info.drop([i for i in cat_info.columns if (i in cat_info.columns and i not in cat_info_col)],axis=1)
cat_info_index = cat_info.drop([i for i in cat_info.columns if i not in [economy,'hhid']],axis=1)

#########################
# HAZARD INFO
#
# This is the GAR
#hazard_ratios = pd.read_csv(inputs+'/PHL_frac_value_destroyed_gar_completed_edit.csv').set_index([economy, 'hazard', 'rp'])

# PHILIPPINES:
# This is the AIR dataset:
# df_haz is already in pesos
# --> Need to think about public assets
#df_haz = get_AIR_data(inputs+'/Risk_Profile_Master_With_Population.xlsx','Loss_Results','all','Agg')

df_haz,df_tikina = get_hazard_df(myCountry,economy,agg_or_occ='Agg',rm_overlap=True)
if myCountry == 'FJ': _ = get_SLR_hazard(myCountry,df_tikina)

# Edit & Shuffle provinces
if myCountry == 'PH':
    AIR_prov_rename = {'Shariff Kabunsuan':'Maguindanao',
                       'Davao Oriental':'Davao',
                       'Davao del Norte':'Davao',
                       'Metropolitan Manila':'Manila',
                       'Dinagat Islands':'Surigao del Norte'}
    df_haz['province'].replace(AIR_prov_rename,inplace=True) 

    # Add NCR 2-4 to AIR dataset
    df_NCR = pd.DataFrame(df_haz.loc[(df_haz.province == 'Manila')])
    df_NCR['province'] = 'NCR-2nd Dist.'
    df_haz = df_haz.append(df_NCR)

    df_NCR['province'] = 'NCR-3rd Dist.'
    df_haz = df_haz.append(df_NCR)
    
    df_NCR['province'] = 'NCR-4th Dist.'
    df_haz = df_haz.append(df_NCR)

    # In AIR, we only have 'Metropolitan Manila'
    # Distribute losses among Manila & NCR 2-4 according to assets
    cat_info = cat_info.reset_index()
    k_NCR = cat_info.loc[((cat_info.province == 'Manila') | (cat_info.province == 'NCR-2nd Dist.') 
                          | (cat_info.province == 'NCR-3rd Dist.') | (cat_info.province == 'NCR-4th Dist.')), ['k','pcwgt']].prod(axis=1).sum()

    for k_type in ['value_destroyed_prv','value_destroyed_pub']:
        df_haz.loc[df_haz.province ==        'Manila',k_type] *= cat_info.loc[cat_info.province ==        'Manila', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-2nd Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-2nd Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-3rd Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-3rd Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-4th Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-4th Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        
    # Add region info to df_haz:
    df_haz = df_haz.reset_index().set_index('province')
    cat_info = cat_info.reset_index().set_index('province')
    df_haz['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region

    df_haz = df_haz.reset_index().set_index(economy)
    cat_info = cat_info.reset_index().set_index(economy)

    # Sum over the provinces that we're merging
    # Losses are absolute value, so they are additive
    df_haz = df_haz.reset_index().set_index([economy,'hazard','rp']).sum(level=[economy,'hazard','rp']).drop(['index'],axis=1)

    df_haz['value_destroyed'] = df_haz[['value_destroyed_prv','value_destroyed_pub']].sum(axis=1)
    df_haz['hh_share'] = (df_haz['value_destroyed_prv']/df_haz['value_destroyed']).fillna(1.)
    # Weird things can happen for rp=2000 (negative losses), but they're < 10E-5, so we don't worry much about them
    #df_haz.loc[df_haz.hh_share>1.].to_csv('~/Desktop/hh_share.csv')

elif myCountry == 'SL': df_haz['hh_share'] = 1.
    
elif myCountry == 'FJ':
    df_haz = df_haz.reset_index().set_index([economy,'hazard','rp']).sum(level=[economy,'hazard','rp'])
    # All the magic happens inside get_hazard_df()

# Turn losses into fraction
cat_info = cat_info.reset_index().set_index([economy])

hazard_ratios = cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy).to_frame(name='HIES_capital')
hazard_ratios = hazard_ratios.join(df_haz,how='outer')

hazard_ratios['grdp_to_assets'] = get_subnational_gdp_macro(myCountry,hazard_ratios,float(df['avg_prod_k'].mean()))

if myCountry == 'PH':
    hazard_ratios['frac_destroyed'] = hazard_ratios['value_destroyed']/hazard_ratios['grdp_to_assets']
    hazard_ratios = hazard_ratios.drop(['HIES_capital', 'value_destroyed','value_destroyed_prv','value_destroyed_pub'],axis=1)

elif myCountry == 'FJ':
    pass
    # --> fa is losses/(exposed_value*v)
    #hazard_ratios['frac_destroyed'] = hazard_ratios['fa'] 

elif myCountry == 'SL':
    pass
    # For SL, 'fa' is fa, not frac_destroyed
    #hazard_ratios['frac_destroyed'] = hazard_ratios.pop('fa')


# Have frac destroyed, need fa...
# Frac value destroyed = SUM_i(k*v*fa)

# Merge hazard_ratios with cat_info
hazard_ratios = pd.merge(hazard_ratios.reset_index(),cat_info.reset_index(),on=economy,how='outer')

# Reduce vulnerability by reduction_vul if hh has access to early warning:
hazard_ratios.loc[hazard_ratios.hazard!='EQ','v'] *= (1-reduction_vul*hazard_ratios.loc[hazard_ratios.hazard!='EQ','has_ew'])
hazard_ratios.loc[hazard_ratios['v']<=0.1,'v'] *= np.random.uniform(.8,2,hazard_ratios.loc[hazard_ratios['v']<=0.1].shape[0])
hazard_ratios.loc[hazard_ratios['v'] >0.1,'v'] *= np.random.uniform(.8,1.2,hazard_ratios.loc[hazard_ratios['v'] >0.1].shape[0])

# Calculate frac_destroyed for SL, since we don't have that in this case
if myCountry == 'SL': hazard_ratios['frac_destroyed'] = hazard_ratios[['v','fa']].prod(axis=1)

if 'hh_share' not in hazard_ratios.columns: hazard_ratios['hh_share'] = None
hazard_ratios = hazard_ratios.reset_index().set_index(event_level+['hhid'])[[i for i in ['frac_destroyed','v','k','pcwgt','hh_share','grdp_to_assets','fa'] if i in hazard_ratios.columns]]
hazard_ratios = hazard_ratios.drop([i for i in ['index'] if i in hazard_ratios.columns])

###########################################
# 2 things going on here:
# 1) Pull v out of frac_destroyed
# 2) Transfer fa in excess of 95% to vulnerability
fa_threshold = 0.95

# Calculate avg vulnerability at event level, and use that to find fa
# --> v_mean is weighted by capital & pc_weight 
v_mean = hazard_ratios[['pcwgt','k','v']].prod(axis=1).sum(level=event_level)/hazard_ratios[['pcwgt','k']].prod(axis=1).sum(level=event_level)
v_mean.name = 'v_mean'
hazard_ratios = pd.merge(hazard_ratios.reset_index(),v_mean.reset_index(),on=event_level).reset_index().set_index(event_level+['hhid']).sort_index()

if myCountry != 'SL':
    # Normally, we pull fa out of frac_destroyed.
    # --> for SL, I think we have fa (not frac_destroyed) from HIES
    hazard_ratios['fa'] = (hazard_ratios['frac_destroyed']/hazard_ratios['v_mean']).fillna(1E-8)
    
    hazard_ratios.loc[hazard_ratios.fa>fa_threshold,'v'] = (hazard_ratios.loc[hazard_ratios.fa>fa_threshold,['v','fa']].prod(axis=1)/fa_threshold).clip(upper=0.95)
    hazard_ratios['fa'] = hazard_ratios['fa'].clip(lower=1E-8,upper=fa_threshold)    

hazard_ratios[['fa','v']].mean(level=event_level).to_csv('tmp/fa_v.csv')

# Get optimal reconstruction rate
_pi = df['avg_prod_k'].mean()
_rho = df['rho'].mean()

print('Running hh_reco_rate optimization')
hazard_ratios['hh_reco_rate'] = 0

try: 
    v_to_reco_rate = pickle.load(open('optimization_libs/'+myCountry+'_v_to_reco_rate_proto2.p','rb'))
except:
    print('Was not able to load v to hh_reco_rate library from optimization_libs/'+myCountry+'_v_to_reco_rate.p')
    v_to_reco_rate = {}

try: hazard_ratios['hh_reco_rate'] = hazard_ratios.apply(lambda x:v_to_reco_rate[round(x.v,2)],axis=1)
except:
    for _n, _i in enumerate(hazard_ratios.index):
        if round(_n/len(hazard_ratios.index)*100,2)%1 == 0:
            print(round(_n/len(hazard_ratios.index)*100,2),'% of way through')

        _v = round(hazard_ratios.loc[_i,'v'],2)
        try: hazard_ratios.loc[_i,'hh_reco_rate'] = v_to_reco_rate[_v]
        except:
            _opt = optimize_reco(_pi,_rho,_v)
            hazard_ratios.loc[_i,'hh_reco_rate'] = _opt
            v_to_reco_rate[_v] = _opt

    pickle.dump(v_to_reco_rate, open('optimization_libs/'+myCountry+'_v_to_reco_rate.p', 'wb' ) )

while False:
    _path = '/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/Figures/'
    _ = hazard_ratios.reset_index().copy()
    
    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==10)],['v'],[''],_path+'vulnerabilities_PF10_log.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=True)
    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==10)],['v'],[''],_path+'vulnerabilities_PF10.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=False)

    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==250)],['v'],[''],_path+'vulnerabilities_PF250_log.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=True)
    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==250)],['v'],[''],_path+'vulnerabilities_PF250.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=False)  
  
    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==1000)],['v'],[''],_path+'vulnerabilities_PF1000_log.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=True)
    plot_simple_hist(_.loc[(_.hazard=='PF')&(_.rp==1000)],['v'],[''],_path+'vulnerabilities_PF1000.pdf',uclip=1,nBins=25,xlab='Vulnerability ($v_h$)',logy=False)
    break

cat_info = cat_info.reset_index().set_index([economy,'hhid'])

#cat_info['v'] = hazard_ratios.reset_index().set_index([economy,'hhid'])['v'].mean(level=[economy,'hhid']).clip(upper=0.99)
# ^ I think this is throwing off the losses!! Average vulnerability isn't going to cut it
# --> Use hazard-specific vulnerability for each hh (in hazard_ratios instead of in cats_event)

# This function collects info on the value and vulnerability of public assets
cat_info, hazard_ratios = get_asset_infos(myCountry,cat_info,hazard_ratios,df_haz)

df.to_csv(intermediate+'/macro.csv',encoding='utf-8', header=True,index=True)

cat_info = cat_info.drop([icol for icol in ['level_0','index'] if icol in cat_info.columns],axis=1)
#cat_info = cat_info.drop([i for i in ['province'] if i != economy],axis=1)
cat_info.to_csv(intermediate+'/cat_info.csv',encoding='utf-8', header=True,index=True)

# If we have 2 sets of data on k, gdp, look at them now:
try:
    summary_df = pd.DataFrame({'FIES':df['avg_prod_k'].mean()*cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy)/1E9,
                               'GRDP':df['avg_prod_k'].mean()*hazard_ratios['grdp_to_assets'].mean(level=economy)/1E9})
    summary_df.loc['Total'] = summary_df.sum()
    summary_df['Ratio'] = 100.*summary_df['FIES'].divide(summary_df['GRDP'])

    print(summary_df.round(1))

    summary_df.round(1).to_latex('latex/grdp_table.tex')
    summary_df.to_csv(intermediate+'/gdp.csv')

    totals = summary_df[['FIES','GRDP']].sum().squeeze()
    ratio = totals[0]/totals[1]
    print(totals, ratio)
except: print('Dont have 2 datasets for GDP. Just using hh survey data.')  

hazard_ratios= hazard_ratios.drop(['frac_destroyed','grdp_to_assets'],axis=1).drop(["flood_fluv_def"],level="hazard")
hazard_ratios.to_csv(intermediate+'/hazard_ratios.csv',encoding='utf-8', header=True)

print(hazard_ratios.head())


# Compare assets from survey to assets from AIR-PCRAFI    
if myCountry == 'FJ':

    df_haz = df_haz.reset_index()
    my_df = ((df[['gdp_pc_prov','pop']].prod(axis=1))/df['avg_prod_k']).to_frame(name='HIES')
    my_df['PCRAFI'] = df_haz.ix[(df_haz.rp==1)&(df_haz.hazard=='TC'),['Division','Exp_Value']].set_index('Division')
    
    my_df['HIES']/=1.E9
    my_df['PCRAFI']/=1.E9
    
    ax = my_df.plot.scatter('PCRAFI','HIES')
    fit_line = np.polyfit(my_df['PCRAFI'],my_df['HIES'],1)
    ax.plot()
    
    plt.xlim(0.,8.)
    plt.ylim(0.,5.)
    
    my_linspace_x = np.array(np.linspace(plt.gca().get_xlim()[0],plt.gca().get_xlim()[1],10))
    my_linspace_y = fit_line[0]*my_linspace_x+fit_line[1]
    
    plt.plot(my_linspace_x,my_linspace_y)
    plt.annotate(str(round(100.*my_linspace_x[1]/my_linspace_y[1],1))+'%',[1.,4.])
    
    fig = plt.gcf()
    fig.savefig('HIES_vs_PCRAFI_assets.pdf',format='pdf')
    
