#modified version from lib_compute_resilience_and_risk_financing.py
import matplotlib
matplotlib.use('AGG')

import gc
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import seaborn as sns
import copy


from libraries.pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories, merge_multi
from libraries.lib_gather_data import social_to_tx_and_gsp, get_hh_savings
from libraries.lib_fiji_sps import run_fijian_SPP, run_fijian_SPS
from libraries.lib_country_dir import *
from libraries.lib_agents import *

pd.set_option('display.width', 220)

sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

const_nom_reco_rate, const_pub_reco_rate = None, None
const_rho, const_ie = None, None
const_pds_rate = None

tmp = 'tmp/'

def get_weighted_mean(q1,q2,q3,q4,q5,key,weight_key='pcwgt'):
    
    if q1.shape[0] > 0:
        my_ret = [np.average(q1[key], weights=q1[weight_key])]
    else: my_ret = [0]
    
    if q2.shape[0] > 0:
        my_ret.append(np.average(q2[key], weights=q2[weight_key]))
    else: my_ret.append(0)

    if q3.shape[0] > 0:
        my_ret.append(np.average(q3[key], weights=q3[weight_key]))
    else: my_ret.append(0)

    if q4.shape[0] > 0:
        my_ret.append(np.average(q4[key], weights=q4[weight_key]))
    else: my_ret.append(0)

    if q5.shape[0] > 0:
        my_ret.append(np.average(q5[key], weights=q5[weight_key]))
    else: my_ret.append(0)    

    return my_ret

def get_weighted_median(q1,q2,q3,q4,q5,key):
    
    q1.sort_values(key, inplace=True)
    q2.sort_values(key, inplace=True)
    q3.sort_values(key, inplace=True)
    q4.sort_values(key, inplace=True)
    q5.sort_values(key, inplace=True)

    cumsum = q1.pcwgt.cumsum()
    cutoff = q1.pcwgt.sum() / 2.0
    median_q1 = round(q1[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q2.pcwgt.cumsum()
    cutoff = q2.pcwgt.sum() / 2.0
    median_q2 = round(q2[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q3.pcwgt.cumsum()
    cutoff = q3.pcwgt.sum() / 2.0
    median_q3 = round(q3[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q4.pcwgt.cumsum()
    cutoff = q4.pcwgt.sum() / 2.0
    median_q4 = round(q4[key][cumsum >= cutoff].iloc[0],3)

    cumsum = q5.pcwgt.cumsum()
    cutoff = q5.pcwgt.sum() / 2.0
    median_q5 = round(q5[key][cumsum >= cutoff].iloc[0],3)

    return [median_q1,median_q2,median_q3,median_q4,median_q5]

def compute_with_hazard_ratios(myCountry,pol_str,fname,macro,cat_info,economy,event_level,
                               income_cats,default_rp,rm_overlap,verbose_replace=True,suffix = ''):

    #cat_info = cat_info[cat_info.c>0]
    hazard_ratios = pd.read_csv(fname, index_col=event_level+[income_cats])
    
    cat_info['ew_expansion'] = 0
    print('CURRENT POLICY--> ',pol_str)

    # Introduce different risk mitigation policies
    macro,cat_info,hazard_ratios = apply_policies(pol_str,macro,cat_info,hazard_ratios)

    #compute
    return process_input(myCountry,pol_str,macro,cat_info,hazard_ratios,economy,event_level,
                         income_cats,default_rp,rm_overlap,verbose_replace=True,suffix = suffix)

def apply_policies(pol_str,macro,cat_info,hazard_ratios):
    

    if pol_str in ('_baseline','_no_code','_retrofit','_no_UI','_UI_2yrs','_no_ins','_ins40_15','_ins50_15','_ins_poor_30','_ins_rich_30','_ins_poor_15','_ins_poor_50','_ins_rich_15','_ins_rich_50'):
        pass
    elif pol_str != '':
        print('What is this policy? --> ',pol_str)
        assert(False)

    return macro,cat_info,hazard_ratios

def process_input(myCountry,pol_str,macro,cat_info,hazard_ratios,economy,event_level,
                  income_cats,default_rp,rm_overlap,verbose_replace=True,suffix = ''):
    flag1=False
    flag2=False

    if type(hazard_ratios)==pd.DataFrame:
        
        hazard_ratios = hazard_ratios.reset_index().set_index(economy).dropna()
        
        try: hazard_ratios = hazard_ratios.drop('Unnamed: 0',axis=1)
        except: pass

        #These lines remove countries/regions/provinces in macro not in cat_info
        if myCountry == 'SL': hazard_ratios = hazard_ratios.dropna()
        else: hazard_ratios = hazard_ratios.fillna(0)
        
         
        # Find which counties are in hazard_ratios and cat_info are also in macro
        common_places = [c for c in macro.index if c in cat_info.index and c in hazard_ratios.index]
        print('common places:',common_places)
        
        hazard_ratios = hazard_ratios.reset_index().set_index(event_level+[income_cats])

        # This drops non-common provinces from macro
        macro = macro.ix[common_places]

        # Nothing drops from cat_info
        cat_info = cat_info.ix[common_places]

        # Nothing drops from hazard_ratios
        hazard_ratios = hazard_ratios.ix[common_places]

        if hazard_ratios.empty:
            hazard_ratios=None
	
        
    if hazard_ratios is None:
        raise('Something went wrong with the hazard during clean up')
        hazard_ratios = pd.Series(1,index=pd.MultiIndex.from_product([macro.index,'default_hazard'],names=[economy, 'hazard']))
		
    #if hazard data has no hazard, it is broadcasted to default hazard
    if 'hazard' not in get_list_of_index_names(hazard_ratios):
        print('Should not be here: hazard not in \'hazard_ratios\'')
        hazard_ratios = broadcast_simple(hazard_ratios, pd.Index(['default_hazard'], name='hazard'))     
		
    #if hazard data has no rp, it is broadcasted to default rp
    if 'rp' not in get_list_of_index_names(hazard_ratios):
        print('Should not be here: RP not in \'hazard_ratios\'')
        hazard_ratios_event = broadcast_simple(hazard_ratios, pd.Index([default_rp], name='rp'))

    ## CHANGED RELEVANT FOR BAY AREA seismic analysis
    # Interpolates data to a more granular grid for return periods that includes all protection values that are potentially not the same in hazard_ratios.
    else:
        hazard_ratios_event = copy.deepcopy(hazard_ratios)
        
        
        #hazard_ratios_event =  pd.DataFrame()
        #for haz in hazard_ratios.reset_index().hazard.unique():
        #    hazard_ratios_event = hazard_ratios_event.append(interpolate_rps(hazard_ratios.reset_index().ix[hazard_ratios.reset_index().hazard==haz,:].set_index(hazard_ratios.index.names), macro.protection,option=default_rp))            
        #hazard_ratios_event = same_rps_all_hazards(hazard_ratios_event)


    ## Broadcast financial protection from macro onto the cat_info
    cat_info['protection']=broadcast_simple(macro['protection'],cat_info.index)	
    
    # Define consumption as the pcinc
    cat_info['c_pc'] = cat_info['pcinc_tot'] - cat_info['mort_pc']

      # This value of provincial GDP is derived from hh consumption in HIES
    avg_c = round(np.average(cat_info.c_pc,weights=cat_info.pcwgt)/get_to_USD(myCountry),2)
    print('***Mean consumption: ',avg_c)
    print('***Mean GDP pc ',round(np.average(macro['gdp_pc_county'],weights=macro['population'])/get_to_USD(myCountry),2),' USD.')


    ##add finance to diversification and taxation
    # social is fraction of income from social safety net --> defined as t(which is social protection)/c_i(consumption)
    cat_info['social'] = unpack_social(macro,cat_info)

    ##cat_info['social']+= 0.1* cat_info['axfin']
    # gamma_sp = social*c_pc/sum(social*c_pc*population)
    macro['tau_tax'], cat_info['gamma_SP'] = social_to_tx_and_gsp(economy,cat_info)
            
    #RECompute consumption from k and new gamma_SP and tau_tax
    #cat_info['c_pc'] = macro['avg_prod_k']*(1.-macro['tau_tax'])*cat_info['k_pc']/(1.-cat_info['social'])
    cat_info['c_pc'] = cat_info['c_pc']*(1.-macro['tau_tax'])/(1.-cat_info['social'])
    # ^ this is per capita
    
    #print('all weights ',cat_info['pcwgt'].sum())

    #plt.cla()
    #ax = plt.gca()
    #ci_heights, ci_bins = np.histogram(cat_info.c.clip(upper=50000),bins=50, weights=cat_info.pcwgt)
    #plt.gca().bar(ci_bins[:-1], ci_heights, width=ci_bins[1]-ci_bins[0], facecolor=q_colors[1], label='C2',alpha=0.4)
    #plt.legend()
    #fig = ax.get_figure()
    #fig.savefig('/Users/brian/Desktop/my_plots/'+myCountry+pol_str+'_consumption_init.pdf',format='pdf')

    print('**********Re-recalc mean cons (pc)',round(np.average((cat_info['c_pc']*cat_info['pcwgt']).sum(level=economy)/macro['population'],weights=macro['population']),2),'(local curr)')    


    ####FORMATTING
    #gets the event level index
    event_level_index = hazard_ratios_event.reset_index().set_index(event_level).index #index composed on counties, hazards and rps.

    #Broadcast macro to event level 
    macro_event = broadcast_simple(macro,event_level_index)	
    #rebuilding exponentially to 95% of initial stock in reconst_duration
    
    #global const_nom_reco_rate
    #const_nom_reco_rate = float(np.log(1/0.05) / macro_event['T_rebuild_K'].mean())
    # ^ Won't have nominal reco rate any more

    global const_pub_reco_rate
    const_pub_reco_rate = float(np.log(1/0.05) / macro_event['T_rebuild_K'].mean())
    
    global const_pds_rate
    const_pds_rate = const_pub_reco_rate*2.
    # All hh consume whatever PDS they receive in first 1.5 years (3 years/2) after disaster
    # ^ Will get rid of this next
    
    global const_rho
    const_rho = float(macro_event['rho'].mean())

    global const_ie
    const_ie = float(macro_event['income_elast'].mean())

    #Calculation of macroeconomic resilience
    #macro_event['macro_multiplier'] =(hazard_ratios_event['dy_over_dk'].mean(level=event_level)+const_nom_reco_rate)/(const_rho+const_nom_reco_rate)  #Gamma in the technical paper
    
    #updates columns in macro with columns in hazard_ratios_event
    common_cols = [c for c in macro_event if c in hazard_ratios_event] #columns that are both in macro_event and hazard_ratios_event
    if not common_cols==[]:
        if verbose_replace:
            flag1=True
            print('Replaced in macro: '+', '.join(common_cols))
            macro_event[common_cols] =  hazard_ratios[common_cols]
    
    #Broadcast categories to event level
    cats_event = broadcast_simple(cat_info,  event_level_index)

    #####################
    # Memory management
    _ix = cats_event.index.names
    cats_event = cats_event.reset_index()

    for __ix in _ix:
        cats_event[__ix] = cats_event[__ix].astype('category')
    cats_event = cats_event.reset_index().set_index(_ix)
    #######################

    # Bay Area: Broadcast 'hh_share' to hazard_ratio_event
    # Not Bay Area: Broadcast 'hh_share' to cats_event 
    
    if myCountry == 'BA':
        hazard_ratios_event = hazard_ratios_event.reset_index().set_index(event_level+['tract'])    
        cats_event = cats_event.reset_index().set_index(event_level+['tract'])
        cats_event['hh_share']= hazard_ratios_event['hh_share'] 

    else:      
        hazard_ratios_event = hazard_ratios_event.reset_index().set_index(event_level+['tract'])    
        cats_event = cats_event.reset_index().set_index(event_level+['tract'])
        cats_event['hh_share'] = hazard_ratios_event['hh_share']
        cats_event['hh_share'] = cats_event['hh_share'].fillna(1.).clip(upper=1.)

    # Transfer vulnerability from haz_ratios to cats_event:
    cats_event['v'] = hazard_ratios_event['v']
    cats_event['v_ins'] = hazard_ratios_event['v_ins']

    cats_event['optimal_hh_reco_rate'] = hazard_ratios_event['hh_reco_rate']
    cats_event['physical_reco_rate'] = hazard_ratios_event['physical_reco_rate']
    cats_event['asset_loss_pc'] = hazard_ratios_event['asset_loss_pc']
    cats_event['optimal_hh_reco_rate'] = cats_event['optimal_hh_reco_rate'].fillna(0)
    hazard_ratios_event = hazard_ratios_event.drop(['v','hh_reco_rate'],axis=1)
    
    ## Get rid of the census tracts that have hhwgt = 0
    hazard_ratios_event.drop(cats_event[cats_event.hhwgt==0].index,inplace=True)
    cats_event.drop(cats_event[cats_event.hhwgt==0].index,inplace=True)

    
# Maryia commented out since I do not know what it does =============================================================================
#     ###############
#     # Don't know what this does, except empty the overlapping columns.
#     # --> checked before & after adn saw that no columns actually change
#     #updates columns in cats with columns in hazard_ratios_event	
#     # applies mh ratios to relevant columns
#     #cols_c = [c for c in cats_event if c in hazard_ratios_event] #columns that are both in cats_event and hazard_ratios_event    
#     cols_c = [c for c in ['fa']] #columns that are both in cats_event and hazard_ratios_event 
#     if not cols_c==[]:
#         hrb = broadcast_simple(hazard_ratios_event[cols_c], cat_info.index).reset_index().set_index(get_list_of_index_names(cats_event)) 
#         # ^ explicitly broadcasts hazard ratios to contain income categories
#         cats_event[cols_c] = hrb
# =============================================================================

    return macro_event, cats_event, hazard_ratios_event 

def compute_dK(pol_str,macro_event,cats_event,event_level,affected_cats,myC,
               optionPDS,share_public_assets=True,labourIncome = False, 
               labour_event = None,suffix = '',nsims = None,output_folder = ''):

    #counts affected and non affected
    cats_event_ia=concat_categories(cats_event,cats_event,index= affected_cats)
    

    # Make sure there are no NaN in fa 
    # --> should be ~0 for provinces with no exposure to a particular disaster
    #cats_event['fa'].fillna(value=1.E-8,inplace=True)
    # Maryia:
    cats_event['fa'] = 1
    myCountry = myC
    
    
    
    # From here, [hhwgt, pcwgt, and pcwgt_ae] are merged with fa
    # --> print('From here: weights (pc and hh) = nAffected and nNotAffected hh/ind') 
    for aWGT in ['hhwgt','pcwgt']:
        myNaf = cats_event[aWGT]*cats_event.fa
        myNna = cats_event[aWGT]*(1.-cats_event.fa)
        cats_event_ia[aWGT] = concat_categories(myNaf,myNna, index=affected_cats)

    # de_index so can access cats as columns and index is still event
    cats_event_ia = cats_event_ia.reset_index(['tract', 'affected_cat']).sort_index()
    #cats_event_ia.loc[cats_event_ia.affected_cat== 'a','pcwgt'] = cats_event_ia.loc[cats_event_ia.affected_cat== 'a'].fillna(0.)
    #cats_event_ia.loc[cats_event_ia.affected_cat=='na','pcwgt'] = cats_event_ia.loc[(cats_event_ia.affected_cat=='na'),'pcwgt'].fillna(value=cats_event_ia.loc[cats_event_ia.affected_cat=='na',['hhwgt','hhsize']].prod(axis=1))

    # set vulnerability to zero for non-affected households
    # --> v for [private, public] assets
    cats_event_ia.loc[cats_event_ia.affected_cat=='na',['v']] = 0

    # 'Actual' vulnerability includes migitating effect of early warning systems
    # --> still 0 for non-affected hh
    cats_event_ia['v_with_ew']=cats_event_ia['v']*(1-macro_event['pi']*cats_event_ia['ew_expansion'])


###################### LABOUR INCOME MODEL
    if labourIncome == True:
    	print('Calculating dk based on labour income')
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level+['tract'])
    	cats_event_ia['dk_pc_public']  = 0.
    	cats_event_ia['dk_pc_oth']   =  0. #cats_event_ia[['k_pc_oth','v_with_ew']].prod(axis=1, skipna=False).fillna(0).clip(lower=0.)#.astype('int') 
    	cats_event_ia['dk_pc_labour'] = labour_event[0]/macro_event['avg_prod_k'].mean()
        cats_event_ia['dk_pc_h'] = cats_event_ia[['k_pc_h','v_with_ew']].prod(axis=1, skipna=False).fillna(0).clip(lower=0.)#.astype('int') 
        cats_event_ia['dk0_pc'] = cats_event_ia['dk_pc_public'] + cats_event_ia['dk_pc_oth'] + cats_event_ia['dk_pc_h']+ cats_event_ia['dk_pc_labour'] 
        # Independent of who pays for reconstruction, the total event losses are given by sum of priv, pub, & other:
        macro_event['dk_event_tot'] = ((cats_event_ia['dk0_pc'])*cats_event_ia['pcwgt']).sum(level=event_level)
        # ^ dk_event is WHEN the event happens--doesn't yet include RP/probability
        
        print('** share_public_assets = True --> Sharing infra & public asset costs among all households *nationally*\n')

        # Create a new dataframe for calculation of public fees borne by affected province & all others 
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level+['tract','affected_cat'])
        rebuild_fees = pd.DataFrame(cats_event_ia[['k_pc','dk_pc_labour','dk_pc_public','dk_pc_oth','dk_pc_h','pcwgt']],index=cats_event_ia.index)
       
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level)
        rebuild_fees = rebuild_fees.reset_index().set_index(event_level)

        #######################################################
        # Redefined by MM from the original
        # dk_tot = dk_t_county --> used to be losses per province and not losses per county
        
        # Total value of public & private asset losses, when an event hits a single province              
        rebuild_fees['dk_labour_tot_county'] = rebuild_fees[['pcwgt','dk_pc_labour']].prod(axis=1).sum(level=event_level)
        rebuild_fees['dk_public_tot_county']  = rebuild_fees[['pcwgt', 'dk_pc_public']].prod(axis=1).sum(level=event_level)
        rebuild_fees['dk_other_tot_county']   = rebuild_fees[['pcwgt',  'dk_pc_oth']].prod(axis=1).sum(level=event_level)
        rebuild_fees['dk_housing_tot_county']   = rebuild_fees[['pcwgt',  'dk_pc_h']].prod(axis=1).sum(level=event_level)
        rebuild_fees['dk_tot_county']         = rebuild_fees[['dk_labour_tot_county','dk_public_tot_county','dk_other_tot_county','dk_housing_tot_county']].sum(axis=1)
        
        #######################################################
        # Now we have dk (labour, public, other)
        # Need to calculate each hh's liability for dk_public_tot

        #######################################################
        # Create a temporary dataframe that sums over provinces
        rebuild_fees_tmp = pd.DataFrame(index=cats_event_ia.sum(level=['hazard','rp']).index)
        
        # tot_k_BE is all the assets in the Bay Area BEFORE EVENT *** BE = BEFORE EVENT ***
        rebuild_fees_tmp['k_tot_BA_BE'] = cats_event_ia[['pcwgt','k_pc']].prod(axis=1,skipna=False).sum(level=['hazard','rp'])
        # Can't calculate PE, because the _tmp df is already summed over provinces
        # ^ BE is the same for all events in all provinces; PE is not

        # Merge _tmp into original df
        rebuild_fees = pd.merge(rebuild_fees.reset_index(),rebuild_fees_tmp.reset_index(),on=['hazard','rp']).reset_index().set_index(event_level)

        # tot_k_PE is all the assets in the Bay Area POST EVENT *** PE = POST EVENT ***
        # --> delta(tot_k_BE , tot_k_PE) will include both public and private and other (?) assets
        # --> that's appropriate because we're imagining tax assessors come after the disaster, and the tax is on income
        # MM change --> fix the mistakes
        #rebuild_fees['dk_tot_BA'] = rebuild_fees['dk_tot_county'].sum(level=['hazard','rp'])
        rebuild_fees = rebuild_fees.reset_index().set_index(['hazard','rp'])
        temp_sum = rebuild_fees[['pcwgt','dk_pc_labour']].prod(axis=1)+rebuild_fees[['pcwgt','dk_pc_public']].prod(axis=1)+rebuild_fees[['pcwgt','dk_pc_oth']].prod(axis=1)
        rebuild_fees['dk_tot_BA']= temp_sum.sum(level = ['hazard','rp'])
        rebuild_fees['k_tot_BA_PE'] = rebuild_fees['k_tot_BA_BE'] - rebuild_fees['dk_tot_BA']
        rebuild_fees = rebuild_fees.reset_index().set_index(event_level)
        #######################################################
        
        # Prepare 2 dfs for working together again
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level+['tract','affected_cat'])
        rebuild_fees = rebuild_fees.reset_index().set_index(event_level+['tract','affected_cat'])

        # Determine the fraction of all capital in the Bay Area in each census tract (includes weighting here)
        # NB: note the difference between BE and PE here
        rebuild_fees['frac_k_tot_BE'] = cats_event_ia[['pcwgt','k_pc']].prod(axis=1,skipna=False)/rebuild_fees['k_tot_BA_BE']
        rebuild_fees['frac_k_tot_PE'] = cats_event_ia['pcwgt']*(cats_event_ia['k_pc']-cats_event_ia['dk0_pc'])/rebuild_fees['k_tot_BA_PE']
        print('frac_k_tot_BE and _PE are based on K (tax on capital). Would need to add social income for it to be tax on income.')

        # This is the fraction of damages for which each hh in affected prov will pay
        rebuild_fees['pc_fee_BE'] = (rebuild_fees[['dk_public_tot_county','frac_k_tot_BE']].prod(axis=1)/rebuild_fees['pcwgt']).fillna(0)
        rebuild_fees['pc_fee_PE'] = (rebuild_fees[['dk_public_tot_county','frac_k_tot_PE']].prod(axis=1)/rebuild_fees['pcwgt']).fillna(0)
        # --> this is where it would go sideways, unless we take a different approach...
        # --> dk_public_tot is for a specific province/hazard/rp, and it needs to be distributed among everyone, nationally
        # --> but it only goes to the hh in the province

        # Now calculate the fee paid by each hh
        # --> still assessing both before and after disaster
        rebuild_fees['tract_fee_BE'] = rebuild_fees[['pc_fee_BE','pcwgt']].prod(axis=1)       
        rebuild_fees['tract_fee_PE'] = rebuild_fees[['pc_fee_PE','pcwgt']].prod(axis=1)  
        
        # Transfer per capita fees back to cats_event_ia 
        cats_event_ia[['pc_fee_BE','pc_fee_PE']] = rebuild_fees[['pc_fee_BE','pc_fee_PE']]
        rebuild_fees['dk0_pc'] = cats_event_ia['dk0_pc'].copy()
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level)

        # Sanity Check: we know this works if hh_fee = 'dk_public_hh'*'frac_k'
        #print(rebuild_fees['dk_public_tot'].head(1))
        #print(rebuild_fees[['hh_fee_BE','frac_k_BE']].sum(level=event_level).head(17))
        #print(rebuild_fees[['hh_fee_PE','frac_k_PE']].sum(level=event_level).head(17))
        
        public_costs = distribute_public_costs(macro_event,rebuild_fees,event_level,'dk_pc_public')
        #public_costs.to_csv('../output_country/'+myCountry+'_'+str(nsims)+'sims/'+output_folder+'/public_costs_'+optionPDS+'.csv')

        ############################
        # Choose whether to assess tax on k (BE='before event') or (PE='post event')
        # --> the total k in non-aff provinces doesn't change, either way
        # --> the fraction of assets in the country does, because of destruction in the aff province. 

        # Uncomment these 2 lines for tax assessment before disaster (or *long* after)
        public_costs = public_costs.rename(columns={'transfer_pub_BE':'transfer_pub','pc_fee_BE':'pc_fee'})
        cats_event_ia = cats_event_ia.rename(columns={'pc_fee_PE':'pc_fee'})

        # Uncomment these 2 lines for tax assessment *immediately* after disaster
        #public_costs = public_costs.rename(columns={'transfer_pub_PE':'transfer_pub'})
        #cats_event_ia = cats_event_ia.rename(columns={'pc_fee_PE':'pc_fee'})
        
        # MM: 'scale_fac_soc' --> mean capital loss in county as a fraction of the bay area assets for each simulation
        cats_event_ia['scale_fac_soc'] = (rebuild_fees['dk_tot_county']/rebuild_fees['k_tot_BA_BE']).mean(level=event_level)

        ############################
        # We can already calculate di0, dc0 for hh in province
        #
        # Define di0 for all households in province where disaster occurred
        # NB: cats_event_ia['dk0'] = cats_event_ia['dk_private'] + cats_event_ia['dk_public'] + cats_event_ia['dk_other'] 
        
        # MM: for now assign pcsoc = 0
        cats_event_ia['pcsoc'] = 0.
        cats_event_ia['di0_pc_labour'] = cats_event_ia['dk_pc_labour']*macro_event['avg_prod_k'].mean()*(1-macro_event['tau_tax'].mean())
        cats_event_ia['di0_pc_oth'] = cats_event_ia['dk_pc_oth']*macro_event['avg_prod_k'].mean()*(1-macro_event['tau_tax'].mean())+cats_event_ia[['pcsoc','scale_fac_soc']].prod(axis=1)
        cats_event_ia['di0_pc_h'] = cats_event_ia['dk_pc_h']*macro_event['avg_prod_k'].mean()*(1-macro_event['tau_tax'].mean())
        cats_event_ia['di0_pc_pub'] = cats_event_ia['dk_pc_public']*macro_event['avg_prod_k'].mean()*(1-macro_event['tau_tax'].mean())
        cats_event_ia['di0_pc_rent'] = cats_event_ia['p_rent_pc']*cats_event_ia['v_with_ew']*(1-macro_event['tau_tax'].mean())
        
        cats_event_ia['di0_pc'] = cats_event_ia['di0_pc_labour'] + cats_event_ia['di0_pc_pub'] + cats_event_ia['di0_pc_oth'] + cats_event_ia['di0_pc_h'] - cats_event_ia['di0_pc_rent'] 
        
        # Sanity check: (C-di) does not bankrupt
        _ = cats_event_ia.loc[(cats_event_ia.c_pc-cats_event_ia.di0_pc)<0]
        if _.shape[0] != 0:
           # _.to_csv(tmp+'bankrupt.csv')
            assert(_.shape[0] == 0)

        # Leaving out all terms without time-dependence
        # EG: + cats_event_ia['pc_fee'] + cats_event_ia['pds']

        # Define dc0 for all households in province where disaster occurred
        cats_event_ia['hh_reco_rate'] = cats_event_ia['optimal_hh_reco_rate']
        # MM: The income decresses due to loss of private capital and the public capital
        cats_event_ia['dc0_pc_recon'] = cats_event_ia['optimal_hh_reco_rate']*cats_event_ia[['v_with_ew','k_pc_str','hh_share']].prod(axis = 1)
        cats_event_ia['dc0_pc_pub'] = cats_event_ia['di0_pc_pub']
        cats_event_ia['dc0_pc_oth'] = cats_event_ia['di0_pc_oth']
        cats_event_ia['dc0_pc_labour'] = cats_event_ia['di0_pc_labour']
        cats_event_ia['dc0_pc_h'] = cats_event_ia['di0_pc_h']
        cats_event_ia['dc0_pc_rent'] = cats_event_ia['di0_pc_rent']
        cats_event_ia['dc0_pc'] = (cats_event_ia['dc0_pc_labour']+ cats_event_ia['dc0_pc_pub'] + cats_event_ia['dc0_pc_oth'] 
            + cats_event_ia['dc0_pc_h'] - cats_event_ia['dc0_pc_rent']  + cats_event_ia['dc0_pc_recon']) 

        # Get indexing right, so there are not multiple entries with same index:
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level+['tract','affected_cat'])

        # We will classify these hh responses for studying dw
        cats_event_ia['welf_class'] = 0
        
        nsims = len(cats_event_ia.reset_index().rp.unique())
        #cats_event_ia.to_csv('../output_country/'+myCountry+'_'+str(nsims)+'sims/'+output_folder+'/recovery_rates_pre_substinence_criteria_'+str(nsims)+'sims.csv')
       
        # Get subsistence line # Changed  by MM from sub_line to pov_lev_pc
        if 'pov_lev_pc' in cats_event_ia.columns: cats_event_ia['c_pc_min'] = cats_event_ia.pov_lev_pc
        else: cats_event_ia['c_pc_min'] = get_subsistence_line(myC)
        print('Using tract response: avoid subsistence (poverty mean) '+str(round(float(cats_event_ia['c_pc_min'].mean()),2)))

        # Policy str for understanding results: go back to the 3-yr reconstruction with a cap on hh_dw 
        #if pol_str == '_unif_reco':
        #    return macro_event, cats_event_ia, public_costs

        # Case 1:
        # --> hh was affected (did lose some assets)
        # --> income losses do not push hh into subsistence
        # --> optimum reco costs do not push into subsistence
        # *** HH response: forego reco_frac of post-disaster consumption to fund reconstruction, while also keeping consumption above subsistence
        c1_crit = '(dk_pc_labour+dk_pc_oth +dk_pc_h!=0)&((c_pc-di0_pc)>c_pc_min)&((c_pc-dc0_pc)>c_pc_min)'
        _c1 = cats_event_ia.query(c1_crit)[['c_pc','c_pc_min','dc0_pc_labour','dc0_pc_rent','dc0_pc_h',\
        'dc0_pc_pub','dc0_pc_oth','dc0_pc_recon','dc0_pc','optimal_hh_reco_rate','v_with_ew','k_pc_str','hh_share']].copy()
        _c1['welf_class']   = 1

        # --> Households fund reco at optimal value:
        _c1['dc0_pc_recon']   = _c1.eval('optimal_hh_reco_rate*v_with_ew*k_pc_str*hh_share')
        _c1['dc0_pc']         = _c1.eval('dc0_pc_recon + dc0_pc_labour + dc0_pc_pub + dc0_pc_oth +dc0_pc_h - dc0_pc_rent')
        cats_event_ia.loc[_c1.index.tolist(),['dc0_pc','dc0_pc_recon','welf_class']] = _c1[['dc0_pc','dc0_pc_recon','welf_class']]
        print('C1: '+str(round(float(100*cats_event_ia.loc[cats_event_ia.welf_class==1].shape[0])
                               /float(cats_event_ia.loc[cats_event_ia.dk0_pc!=0].shape[0]),2))+'% of census tracts optimize recovery')

        # Case 2:
        # --> hh was affected (did lose some assets)
        # --> income losses do not push hh into subsistence
        # --> optimum reco costs DO PUSH into subsistence
        # *** HH response: forego all consumption in excess of subsistence line to fund reconstruction
        c2_crit = '(dk_pc_labour+dk_pc_oth+dk_pc_h!=0)&(c_pc-di0_pc>c_pc_min)&((c_pc-dc0_pc)<=c_pc_min)'
        _c2 = cats_event_ia.query(c2_crit)[['c_pc','c_pc_min','di0_pc',\
        'dc0_pc_pub','dc0_pc_oth','dc0_pc_recon','dc0_pc','dc0_pc_h','dc0_pc_labour','dc0_pc_rent',\
        'optimal_hh_reco_rate','v_with_ew','k_pc_str','hh_share']].copy()
        _c2['welf_class']   = 2

        # --> Households can't afford to fund reco at optimal value, so they stop at the subsistence line:
        _c2['hh_reco_rate'] = _c2.eval('(c_pc-di0_pc-c_pc_min)/(v_with_ew*k_pc_str*hh_share)')
        _c2['dc0_pc_recon']      = _c2[['hh_reco_rate','v_with_ew','k_pc_str','hh_share']].prod(axis=1)
        _c2['dc0_pc']          = _c2['dc0_pc_recon'] + _c2['dc0_pc_pub']+ _c2['dc0_pc_oth'] + _c2['dc0_pc_labour'] +_c2['dc0_pc_h'] - _c2['dc0_pc_rent']

        if _c2.shape[0] > 0: 
            assert(_c2['hh_reco_rate'].min()>=0)
            cats_event_ia.loc[_c2.index.tolist(),['dc0_pc','dc0_pc_recon','hh_reco_rate','welf_class']] = _c2[['dc0_pc','dc0_pc_recon','hh_reco_rate','welf_class']]
        print('C2: '+str(round(float(100*cats_event_ia.loc[cats_event_ia.welf_class==2].shape[0])
                               /float(cats_event_ia.loc[cats_event_ia.dk0_pc!=0].shape[0]),2))+'% of census tracts do not optimize to avoid subsistence')
            
        # Case 3: Post-disaster income is below subsistence (or c_5):
        # --> hh are not in any other class
        # --> c OR (c-di0) is below subsistence 
        # HH response: do not reconstruct
        _c3 = cats_event_ia.query('(dk_pc_labour+dk_pc_oth+k_pc_h!=0)&(c_pc-di0_pc<c_pc_min)')[['dc0_pc','dc0_pc_labour','dc0_pc_recon',\
        'dc0_pc_pub','dc0_pc_oth','dc0_pc_h','dc0_pc_rent']].copy()
        _c3['welf_class']   = 3
        _c3['hh_reco_rate'] = 1e-10             # No Reconstruction
        _c3['dc0_recon']      = 0#_c3['di0_pc_prv'] # No Reconstruction
        _c3['dc0']          = _c3['dc0_pc_recon'] +_c3['dc0_pc_labour'] + _c3['dc0_pc_pub']+ _c3['dc0_pc_oth'] + _c3['dc0_pc_h'] + _c3['dc0_pc_rent']

        if _c3.shape[0]!=0: cats_event_ia.loc[_c3.index.tolist(),['dc0_pc','dc0_pc_recon','hh_reco_rate','welf_class']] = _c3[['dc0_pc','dc0_pc_recon','hh_reco_rate','welf_class']]
        print('C3: '+str(round(float(100*cats_event_ia.loc[(cats_event_ia.welf_class==3)].shape[0])
                               /float(cats_event_ia.loc[cats_event_ia.dk0_pc!=0].shape[0]),2))+'% of census tracts do not reconstruct')
        # See if anyone's consumption goes negative
        if cats_event_ia.loc[(cats_event_ia.dc0_pc > cats_event_ia.c_pc)].shape[0] != 0:
            hh_extinction = str(round(float(cats_event_ia.loc[(cats_event_ia.dc0_pc > cats_event_ia.c_pc)].shape[0]/cats_event_ia.shape[0])*100.,2))
            print('***ISSUE: '+hh_extinction+'% of (hh x event) combos face dc0 > i0. Could mean extinction!\n--> SOLUTION: capping dw at 20xGDPpc \n\n')
            assert(False)
      
        # REMOVE THE NON-AFFECTED
        cats_event_ia.drop('na', level=4,axis=0,inplace=True)
        #cats_event_ia.to_csv('../output_country/'+myCountry+'_'+str(nsims)+'sims/'+output_folder+'/recovery_rates_post_substinence_criteria_'+str(nsims)+'sims.csv')
       
        # make plot here: (income vs. length of reco)
        #plt.figure
        #plt.scatter(cats_event_ia.pcinc_tot,np.log(1/0.05)/cats_event_ia.hh_reco_rate)
        #plt.ylim(0,10)
        print 'Correlation between income and reconstruction rate: ',np.corrcoef(cats_event_ia.pcinc_tot,np.log(1/0.05)/cats_event_ia.hh_reco_rate)
        
        # NOTES
        # --> when affected by a disaster, hh lose their private assets...
        # --> *AND* the fraction of the public assets for which they're responsible
        # So far, the losses don't include pc_fee (public reco) or PDS
        # --> pc_fee is only for people (aff & non-aff) in the affected province
        ############################
        cats_event_ia = cats_event_ia.reset_index().set_index(event_level)
        assert(cats_event_ia.hh_reco_rate.min()>=0)
        
        # NPV consumption losses accounting for reconstruction and productivity of capital (pre-response)
        cats_event_ia['macro_multiplier'] = (macro_event['avg_prod_k'].mean(level=event_level)+cats_event_ia['hh_reco_rate'])/(const_rho+cats_event_ia['hh_reco_rate'])  
        # ^ Gamma in the technical paper
        cats_event_ia['dc_npv_pre'] = cats_event_ia[['dc0_pc','macro_multiplier']].prod(axis=1)
    else:
        raise ('MODEL WITH NO LABOUR INCOME IS NOT IMPLEMENTED')
    return macro_event, cats_event_ia, public_costs

def distribute_public_costs(macro_event,rebuild_fees,event_level,transfer_type):
    
    ############################        
    # Make another output file --> public_costs.csv
    # --> this contains the cost to each province/region of each disaster (hazard x rp) in another province
    public_costs = pd.DataFrame(index=macro_event.index)

    # Total capital in each province
    public_costs['tot_k_recipient_BE'] = rebuild_fees[['pcwgt','k_pc']].prod(axis=1).sum(level=event_level) 
    public_costs['tot_k_recipient_PE'] = (rebuild_fees['pcwgt']*(rebuild_fees['k_pc']-rebuild_fees['dk0_pc'])).sum(level=event_level)
        
    # Total public losses from each event        
    public_costs['dk_public_recipient']    = rebuild_fees[['pcwgt',transfer_type]].prod(axis=1).sum(level=event_level).fillna(0)
    
    # Cost to province where disaster occurred of rebuilding public assets
    rebuild_fees['dk_tot_public'] = rebuild_fees[['pcwgt',transfer_type]].prod(axis=1)
    # ^ gives total public losses suffered by all people represented by each hh
    public_costs['int_cost_BE'] = (rebuild_fees[['dk_tot_public','frac_k_tot_BE']].sum(level=event_level)).prod(axis=1)
        
    # Total cost to ALL provinces where disaster did not occur rebuilding public assets
    public_costs['ext_cost_BE'] = public_costs['dk_public_recipient'] - public_costs['int_cost_BE']
    public_costs['tmp'] = 1

    # Grab a list of names of all regions/provinces
    prov_k = pd.DataFrame(index=rebuild_fees.sum(level=event_level[0]).index)
    prov_k.index.names = ['contributer']
    prov_k['frac_k_BE'] = rebuild_fees['frac_k_tot_BE'].sum(level=event_level).mean(level=event_level[0])/rebuild_fees['frac_k_tot_BE'].sum(level=event_level).mean(level=event_level[0],skipna=True).sum()
    prov_k['tot_k_contributer'] = rebuild_fees[['pcwgt','k_pc']].prod(axis=1).sum(level=event_level).mean(level=event_level[0])
    # Can't define frac_k_PE here: _tmp does not operate at event level
    prov_k['tmp'] = 1
    prov_k = prov_k.reset_index()
        
    public_costs = pd.merge(public_costs.reset_index(),prov_k.reset_index(),on=['tmp']).reset_index().set_index(event_level+['contributer']).sort_index()
    public_costs = public_costs.drop(['index','level_0','tmp'],axis=1)
    # ^ broadcast prov index to public_costs (2nd provincial index)
    
    public_costs = public_costs.reset_index()
    public_costs['prov_assets_PE'] = 0
    public_costs.loc[(public_costs.contributer==public_costs[event_level[0]]),'prov_assets_PE'] = public_costs.loc[(public_costs.contributer==public_costs[event_level[0]]),'tot_k_recipient_PE']
    public_costs.loc[(public_costs.contributer!=public_costs[event_level[0]]),'prov_assets_PE'] = public_costs.loc[(public_costs.contributer!=public_costs[event_level[0]]),'tot_k_contributer']
    
    public_costs = public_costs.reset_index().set_index(event_level).drop('index',axis=1)
    
    public_costs['frac_k_PE'] = public_costs['prov_assets_PE']/public_costs['prov_assets_PE'].sum(level=event_level)
    
    public_costs['transfer_pub_BE'] = public_costs[['dk_public_recipient','frac_k_BE']].prod(axis=1)
    public_costs['transfer_pub_PE'] = public_costs[['dk_public_recipient','frac_k_PE']].prod(axis=1)
    public_costs['PE_to_BE'] = public_costs['transfer_pub_PE']/public_costs['transfer_pub_BE']
    
    #public_costs = public_costs.drop([i for i in public_costs.columns if i not in ['contributer','dk_public_recipient','frac_k_BE','frac_k_PE',
    #                                                                               'transfer_pub_BE','transfer_pub_PE','PE_to_BE']],axis=1)
    
    public_costs['dw'] = None
    
    return public_costs

def calc_dw_outside_affected_province(macro_event, cat_info, public_costs_pub, public_costs_pds, event_level, is_contemporaneous=False, is_local_welfare=False, is_revised_dw=True):

    #public_costs = pd.DataFrame(index=public_costs_pub.index)
    public_costs = public_costs_pub[['contributer','transfer_pub','tot_k_recipient_BE','tot_k_recipient_PE','tot_k_contributer']].copy()
    public_costs['transfer_pds'] = public_costs_pds['transfer_pds']

    ############################
    # So we have cost of each disaster in each province to every other province
    # - need to calc welfare impact of these transfers
    public_costs = public_costs.reset_index()
    cat_info = cat_info.reset_index().set_index(['hhid'])

    for iP in public_costs.contributer.unique():
        print('Running revised, non-contemp dw for hh outside '+iP) 
        
        tmp_df = cat_info.loc[(cat_info[event_level[0]]==iP)].copy()

        tmp_df['hh_frac_k'] = tmp_df[['pcwgt','k']].prod(axis=1)/(tmp_df[['pcwgt','k']].prod(axis=1)).sum()
        tmp_df['pc_frac_k'] = tmp_df['hh_frac_k']/tmp_df['pcwgt']
        # ^ this grabs a single instance of each hh in each contributing (non-aff) province
        # --> 'k' and 'c' are not distributed between {a,na} (use mean), but pcwgt is (use sum)
        # --> 'pc_frac_k' used to determine what they'll pay when a disaster happens elsewhere
        # Only using capital--not income...
        tmp_t_reco     = float(macro_event['T_rebuild_K'].mean())
        c_mean         = float(cat_info[['pcwgt','c']].prod(axis=1).sum()/cat_info['pcwgt'].sum())
        h = 1.E-4

        wprime = c_mean**(-const_ie)
        # ^ these *could* vary by province/event, but don't (for now), so I'll use them outside the pandas dfs.
            
        for iRecip in public_costs[event_level[0]].unique():
            for iHaz in public_costs.hazard.unique():
                for iRP in public_costs.rp.unique():

                    # Calculate wprime
                    tmp_wp = None
                    if is_local_welfare:
                        tmp_gdp = macro_event.loc[(macro_event[event_level[0]]==iP),'gdp_pc_county'].mean()
                        tmp_wp =(welf(tmp_gdp/const_rho+h,const_ie)-welf(tmp_gdp/const_rho-h,const_ie))/(2*h)
                    else: 
                        tmp_wp =(welf(macro_event['gdp_pc_nat'].mean()/const_rho+h,const_ie)-welf(macro_event['gdp_pc_nat'].mean()/const_rho-h,const_ie))/(2*h)

                    tmp_cost_pub = float(public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer == iP)
                                                           &(public_costs.hazard == iHaz)&(public_costs.rp==iRP)),'transfer_pub'])
                    tmp_cost_pds = float(public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer == iP)
                                                           &(public_costs.hazard == iHaz)&(public_costs.rp==iRP)),'transfer_pds'])
                    # ^ this identifies the amount that a province (iP, above) will contribute to another province when a disaster occurs

                    if is_contemporaneous or not is_revised_dw: 
                        tmp_df['dc_per_cap'] = (tmp_cost_pub+tmp_cost_pds)*tmp_df['pc_frac_k']

                        if not is_revised_dw:
                            tmp_df['dw'] = tmp_df['pcwgt']*(welf1(tmp_df['c']/const_rho, const_ie, tmp_df['c_5']/const_rho)
                                                            - welf1((tmp_df['c']-tmp_df['dc_per_cap'])/const_rho, const_ie,tmp_df['c_5']/const_rho))/tmp_wp
                            # ^ returns NPV

                        else:
                            tmp_df['dw'] += tmp_df['pcwgt']*(welf1(tmp_df['c'], const_ie, tmp_df['c_5']) - welf1((tmp_df['c']-tmp_df['dc_per_cap']), const_ie,tmp_df['c_5']))/wprime
                       
                    else:                        
                        # Here, we calculate the impact of transfers for public assets & PDS
                        tmp_df['dw']     = 0.
                        for iT in [tmp_cost_pub,tmp_cost_pds]:
                            tmp_df['dw'] += tmp_df['pcwgt']*(welf1(tmp_df['c'], const_ie, tmp_df['c_5']) - welf1((tmp_df['c']-iT*tmp_df['pc_frac_k']), const_ie,tmp_df['c_5']))
                            
                        frac_dk_natl = float((public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)
                                                                &(public_costs.contributer == iP)),'tot_k_recipient_BE']-
                                              public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)
                                                                &(public_costs.contributer == iP)),'tot_k_recipient_PE'])/
                                             (public_costs.loc[(public_costs[event_level[0]]==iRecip)&(public_costs.hazard == iHaz)&(public_costs.rp==iRP),'tot_k_contributer']).sum())
                        
                        # Also need to add impacts of social transfer decreases
                        # Set-up to be able to calculate integral
                        tmp_df['const'] = -1.*tmp_df['c']**(1.-const_ie)/(1.-const_ie)
                        tmp_df['integ'] = 0.

                        x_min, x_max, n_steps = 0.5,tmp_t_reco+0.5,12.
                        int_dt,step_dt = np.linspace(x_min,x_max,num=n_steps,endpoint=True,retstep=True)
                        # ^ make sure that, if T_recon changes, so does x_max!

                        # Calculate integral
                        for _t in int_dt:
                            # need total asset losses --> tax base reduction
                            tmp_df['integ'] += step_dt*((1.-frac_dk_natl*tmp_df['social']*math.e**(-_t*const_nom_reco_rate))**(1-const_ie)-1)*math.e**(-_t*const_rho)
                            # ^ const_nom_reco_rate doesn't get replaced by hh-dep values here, because this is aggregate reco
                            
                        # put it all together, including w_prime:
                        tmp_df['dw_soc'] = tmp_df[['pcwgt','const','integ']].prod(axis=1)

                    public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer==iP)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)),'dw'] = \
                        abs(tmp_df['dw'].sum()/wprime)
                    public_costs.loc[((public_costs[event_level[0]]==iRecip)&(public_costs.contributer==iP)&(public_costs.hazard==iHaz)&(public_costs.rp==iRP)),'dw_soc'] = \
                        abs(tmp_df['dw_soc'].sum()/wprime)

    public_costs = public_costs.reset_index().set_index(event_level).drop('index',axis=1)
    
    return public_costs


# We already have the time-dependent parts of dk, di, dc
# 'calculate_response' is a separate function because it bifurcates each hh again between {helped X not helped}
def calculate_response(myCountry,pol_str,macro_event,cats_event_ia,public_costs,event_level,helped_cats,default_rp,option_CB,
                       optionFee='tax',optionT='data', optionPDS='unif_poor', optionB='data',loss_measure='dk_private',fraction_inside=1, 
                       share_insured=.25,labourIncome = False, labour_event = None, suffix = '',nsims = None,output_folder =''):

    
    cats_event_iah = broadcast_simple(cats_event_ia, helped_cats).reset_index().set_index(event_level)

    # Baseline case (no insurance):
    cats_event_iah['help_received'] = 0
    cats_event_iah['help_fee'] =0

#    macro_event, cats_event_iah, public_costs = compute_response(myCountry, pol_str,macro_event, cats_event_iah, public_costs, event_level,default_rp,option_CB,optionT=optionT, 
#                                                                 optionPDS=optionPDS, optionB=optionB, optionFee=optionFee, fraction_inside=fraction_inside, loss_measure = loss_measure, 
#                                                                 labourIncome = labourIncome, labour_event = labour_event, suffix = suffix, nsims = nsims,output_folder =output_folder )
#    
#    cats_event_iah.drop(['protection'],axis=1, inplace=True)
#    return macro_event, cats_event_iah, public_costs
	
#def compute_response(myCountry, pol_str, macro_event, cats_event_iah,public_costs, event_level, default_rp, option_CB,optionT='data', 
#                     optionPDS='unif_poor', optionB='data', optionFee='tax', fraction_inside=1, loss_measure='dk_total',
#                     labourIncome = False, labour_event = None, suffix = '', nsims = None, output_folder = ''):    

#    print('NB: when summing over cats_event_iah, be aware that each hh appears 4X in the file: {a,na}x{helped,not_helped}')
    print('MM: when summing over cats_event_iah, be aware that each census tract appears 2X in the file: {helped,not_helped}')
    # This function computes aid received, aid fee, and other stuff, 
    # --> including losses and PDS options on targeting, financing, and dimensioning of the help.
    # --> Returns copies of macro_event and cats_event_iah updated with stuff

    #macro_event    = macro_event.copy()
    #cats_event_iah = cats_event_iah.copy()

    macro_event['fa'] = (cats_event_iah.loc[(cats_event_iah.affected_cat=='a'),'pcwgt'].sum(level=event_level)/(cats_event_iah['pcwgt'].sum(level=event_level))).fillna(1E-8)
    # No factor of 2 in denominator affected households are counted twice in both the num & denom
    # --> at this point, each appears twice (helped & not_helped)
    # --> the weights haven't been adjusted to include targetng errors
    
    ####targeting errors
    if optionPDS == 'fiji_SPS' or optionPDS == 'fiji_SPP':
        macro_event['error_incl'] = 1.0
        macro_event['error_excl'] = 0.0
    elif optionT=='perfect':
        macro_event['error_incl'] = 0
        macro_event['error_excl'] = 0    
    elif optionT=='prop_nonpoor_lms':
        macro_event['error_incl'] = 0
        macro_event['error_excl'] = 1-25/80  #25% of pop chosen among top 80 DO receive the aid
    elif optionT=='data':
        macro_event['error_incl']=(1)/2*macro_event['fa']/(1-macro_event['fa'])
        macro_event['error_excl']=(1)/2
    elif optionT=='x33':
        macro_event['error_incl']= .33*macro_event['fa']/(1-macro_event['fa'])
        macro_event['error_excl']= .33
    elif optionT=='incl':
        macro_event['error_incl']= .33*macro_event['fa']/(1-macro_event['fa'])
        macro_event['error_excl']= 0
    elif optionT=='excl':
        macro_event['error_incl']= 0
        macro_event['error_excl']= 0.33

    else:
        print('unrecognized targeting error option '+optionT)
        return None
            
    #counting (mind self multiplication of n)
    df_index = cats_event_iah.index.names    

    cats_event_iah = pd.merge(cats_event_iah.reset_index(),macro_event.reset_index()[[i for i in macro_event.index.names]+['error_excl','error_incl']],on=[i for i in macro_event.index.names])

    ## MM: EXCLUDE ALL THE NON-AFFECTED AND HELPED OPTIONS for now
    #for aWGT in ['hhwgt','pcwgt']:
     #   cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='a') ,aWGT]*=(1-cats_event_iah['error_excl'])
      #  cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='a') ,aWGT]*=(  cats_event_iah['error_excl'])
       # cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')    & (cats_event_iah.affected_cat=='na'),aWGT]*=(  cats_event_iah['error_incl'])  
        #cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped')& (cats_event_iah.affected_cat=='na'),aWGT]*=(1-cats_event_iah['error_incl'])

    cats_event_iah = cats_event_iah.reset_index().set_index(df_index).drop([icol for icol in ['index','error_excl','error_incl'] if icol in cats_event_iah.columns],axis=1)
    #cats_event_iah = cats_event_iah.drop(['index'],axis=1)
    
    # MAXIMUM NATIONAL SPENDING ON SCALE UP
    macro_event['max_increased_spending'] = 0.05

    # max_aid is per cap, and it is constant for all disasters & provinces
    # --> If this much were distributed to everyone in the country, it would be 5% of GDP
    # MM: Aid is the same throughout the Bay Area
    macro_event['max_aid'] = macro_event['max_increased_spending'].mean()*macro_event[['gdp_pc_county','population']].prod(axis=1).sum(level=['hazard','rp']).mean()/(macro_event['population'].sum(level=['hazard','rp']).mean())

    ############################
    # Calculate SP payout
    cats_event_iah['help_received'] = 0

    if optionPDS=='no':
        macro_event['aid'] = 0
        optionB='no'

    elif optionPDS == 'fiji_SPP': cats_event_iah = run_fijian_SPP(macro_event,cats_event_iah)        
    elif optionPDS=='fiji_SPS': cats_event_iah = run_fijian_SPS(macro_event,cats_event_iah)   

    elif optionPDS=='unif_poor':
        # For this policy: help_received by all helped hh = shareable (0.8) * average losses (dk) of lowest quintile
        cats_event_iah.loc[cats_event_iah.eval('helped_cat=="helped"'),'help_received'] = macro_event['shareable'] * cats_event_iah.loc[cats_event_iah.eval('(affected_cat=="a") & (quintile==1)'),[loss_measure,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah.loc[cats_event_iah.eval('(affected_cat=="a") & (quintile==1)'),'pcwgt'].sum(level=event_level)

    elif optionPDS=='unif_poor_only':
        # For this policy: help_received by all helped hh in 1st quintile = shareable (0.8) * average losses (dk) of lowest quintile
        cats_event_iah.loc[cats_event_iah.eval('(helped_cat=="helped")&(quintile==1)'),'help_received']=macro_event['shareable']*cats_event_iah.loc[cats_event_iah.eval('(affected_cat=="a") & (quintile==1)'),[loss_measure,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah.loc[cats_event_iah.eval('(affected_cat=="a") & (quintile==1)'),'pcwgt'].sum(level=event_level)

    elif optionPDS=='unif_poor_q12':
        # For this policy: help_received by all helped hh in 1st & 2nd quintiles = shareable (0.8) * average losses (dk) of lowest quintile
        cats_event_iah.loc[cats_event_iah.eval('(helped_cat=="helped")&(quintile<=2)'),'help_received']=macro_event['shareable']*cats_event_iah.loc[cats_event_iah.eval('(affected_cat=="a") & (quintile==1)'),[loss_measure,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah.loc[cats_event_iah.eval('(affected_cat=="a") & (quintile==1)'),'pcwgt'].sum(level=event_level)

    elif 'prop' in optionPDS:
        # Assuming there is no inclusion error for prop, since it assumes demonstrated losses ... there can still be excl error, though
        cats_event_iah = cats_event_iah.reset_index().set_index(event_level+['hhid','affected_cat'])        

        if optionPDS=='prop':
            cats_event_iah.loc[(cats_event_iah.helped_cat=='helped'),'help_received']= 0.5*cats_event_iah.loc[(cats_event_iah.helped_cat=='helped'),loss_measure]

        #if not 'has_received_help_from_PDS_cat' in cats_event_iah.columns: print('pds = prod with additional criteria not implemented')
        #else: print('pds = prod with additional criteria not implemented')
    
        if optionPDS=='prop_q1':
            cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.quintile==1),'help_received']= 0.5*cats_event_iah.loc[(cats_event_iah.helped_cat=='helped'),loss_measure]

        if optionPDS=='prop_q12':
            cats_event_iah.loc[(cats_event_iah.helped_cat=='helped')&(cats_event_iah.quintile<=2),'help_received']= 0.5*cats_event_iah.loc[(cats_event_iah.helped_cat=='helped'),loss_measure]

        cats_event_iah = cats_event_iah.reset_index(['tract','affected_cat'],drop=False)

    #######################################
    # Save out info on the costs of these policies
    my_sp_costs = pd.DataFrame({'event_cost':cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=event_level)},index=cats_event_iah.sum(level=event_level).index)
    my_sp_costs=my_sp_costs.reset_index('rp')
    my_sp_costs['avg_admin_cost'],probability = average_over_rp(my_sp_costs.reset_index().set_index(event_level)['event_cost'],myCountry = myCountry  )

    print('SP: avg admin costs:',my_sp_costs['avg_admin_cost'].mean(level=event_level[:1]).sum())
    
    my_sp_costs['avg_natl_cost'] = my_sp_costs['avg_admin_cost'].mean(level=event_level[:1]).sum()
    #my_sp_costs.to_csv('../output_country/'+myCountry+'_'+str(nsims)+'sims/'+output_folder+'/sp_costs_'+optionPDS+'.csv')

    #actual aid reduced by capacity
    # ^ Not implemented for now
		
    #######################################
    # Above code calculated the benefits
    # now need to determine the costs

    # Already did all the work for this in determining dk_public
    public_costs = public_costs.drop([i for i in public_costs.columns if i not in ['contributer','frac_k_BE','frac_k_PE']], axis=1)
    public_costs['pds_cost'] = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=event_level)
    public_costs['pds_cost'] = public_costs['pds_cost'].fillna(0)

    public_costs['transfer_pds_BE'] = public_costs[['pds_cost','frac_k_BE']].prod(axis=1)
    public_costs['transfer_pds_PE'] = public_costs[['pds_cost','frac_k_PE']].prod(axis=1)

    # Uncomment this line for tax assessment before disaster
    public_costs = public_costs.rename(columns={'transfer_pds_BE':'transfer_pds','pc_fee_BE':'pc_fee'})
    # Uncomment this line for tax assessment after disaster
    #public_costs = public_costs.rename(columns={'transfer_pds_PE':'transfer_pds'})

    # this will do for hh in the affected province
    if optionFee=='tax':

        # Original code:
        #cats_event_iah['help_fee'] = fraction_inside*macro_event['aid']*cats_event_iah['k']/agg_to_event_level(cats_event_iah,'k',event_level)
        # ^ this only manages transfers within each province 

        cats_event_iah['help_fee'] = 0
        if optionPDS == 'fiji_SPS' or optionPDS == 'fiji_SPP':
            
            cats_event_iah = pd.merge(cats_event_iah.reset_index(),(cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])).reset_index(),on=['hazard','rp'])
            cats_event_iah = cats_event_iah.reset_index().set_index(event_level)
            cats_event_iah = cats_event_iah.rename(columns={0:'totex'}).drop(['index','level_0'],axis=1)
            ## ^ total expenditure

            cats_event_iah = pd.merge(cats_event_iah.reset_index(),(cats_event_iah[['k','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])).reset_index(),on=['hazard','rp'])
            cats_event_iah = cats_event_iah.reset_index().set_index(event_level)
            cats_event_iah = cats_event_iah.rename(columns={0:'totk'})          
            
            cats_event_iah['help_fee'] = (cats_event_iah[['totex','k','pcwgt']].prod(axis=1)/cats_event_iah['totk'])/cats_event_iah['pcwgt']
            # ^ could eliminate the two instances of 'pcwgt', but I'm leaving them in to make clear to future me how this was constructed

            cats_event_iah = cats_event_iah.reset_index().set_index(event_level)
            cats_event_iah = cats_event_iah.drop(['index','totex','totk','nOlds','SP_CPP','SP_FAP','SP_FNPF','SP_SPS','SP_PBS'],axis=1)

            # Something like this should evaluate to true
            #assert((cats_event_iah[['pcwgt','help_received']].prod(axis=1).sum(level=['hazard','rp'])).ix['TC'] == 
            #       (cats_event_iah[['pcwgt','help_fee']].prod(axis=1).sum(level=['hazard','rp'])).ix['TC'])

        else:
            cats_event_iah.loc[cats_event_iah.pcwgt != 0,'help_fee'] = (cats_event_iah.loc[cats_event_iah.pcwgt != 0,['help_received','pcwgt']].prod(axis=1).sum(level=event_level) * 
                                                                        # ^ total expenditure
                                                                        (cats_event_iah.loc[cats_event_iah.pcwgt != 0,['k_pc','pcwgt']].prod(axis=1) /
                                                                         cats_event_iah.loc[cats_event_iah.pcwgt != 0,['k_pc','pcwgt']].prod(axis=1).sum(level=event_level)) /
                                                                        # ^ weighted average of capital
                                                                        cats_event_iah.loc[cats_event_iah.pcwgt != 0,'pcwgt']) 
                                                                        # ^ help_fee is per individual!

    elif optionFee=='insurance_premium':
        print('Have not implemented optionFee==insurance_premium')
        assert(False)


    cats_event_iah.drop(['protection'],axis=1, inplace=True)
    return macro_event, cats_event_iah, public_costs


def calc_dw_inside_affected_province(myCountry,pol_str,optionPDS,macro_event,cats_event_iah,event_level,option_CB,return_stats=True,return_iah=True,is_revised_dw=True,
                                     labourIncome = False, labour_event = None,nsims = 0,suffix = '',output_folder = ''):

    cats_event_iah = cats_event_iah.reset_index().set_index(event_level+['tract','affected_cat','helped_cat'])

    # These terms contribute to dc in the affected province:
    # 1) dk0 -> di0 -> dc0
    # 2) dc_reco (private only)
    # 3) PDS receipts
    # 4) New tax burden
    # 5)*Soc. transfer reductions
    # 6)*Public asset reco fees
    # 7)*PDS fees

    # These terms contribute to dc outside the affected province: 
    # 5)*Soc. transfer reductions
    # 6)*Public asset reco fees
    # 7)*PDS fees

    # *These are calculated in calc_dw_outside_affected_province()

    ###################
    # Memory management
    _ix = cats_event_iah.index.names
    cats_event_iah = cats_event_iah.reset_index()

    for __ix in _ix:
        print(__ix)
        cats_event_iah[__ix] = cats_event_iah[__ix].astype('category')
    cats_event_iah = cats_event_iah.reset_index().set_index(_ix)
    ###################

    #cats_event_iah['dc_npv_post'] = cats_event_iah['dc_npv_pre']-cats_event_iah['help_received']+cats_event_iah['help_fee']*option_CB
    if is_revised_dw:
        print('changing dc to include help_received and help_fee, since instantaneous loss is used instead of npv for dw')
        print('how does timing affect the appropriateness of this?')
        #cats_event_iah['dc_post_pds'] = cats_event_iah['dc0']-cats_event_iah['help_received']+cats_event_iah['help_fee']*option_CB
    
    cats_event_iah['dc_post_reco'], cats_event_iah['dw'] = [0,0]
    cats_event_iah.loc[cats_event_iah.pcwgt!=0,'dc_post_reco'], cats_event_iah.loc[cats_event_iah.pcwgt!=0,'dw'] = calc_delta_welfare(myCountry,cats_event_iah,macro_event,
                                                                                                                                      pol_str,optionPDS,is_revised_dw,
                                                                                                                                      labourIncome = labourIncome, labour_event = labour_event,
                                                                                                                                      nsims = nsims,suffix = suffix,output_folder=output_folder)
    assert(cats_event_iah['dc_post_reco'].shape[0] == cats_event_iah['dc_post_reco'].dropna().shape[0])
    assert(cats_event_iah['dw'].shape[0] == cats_event_iah['dw'].dropna().shape[0])

    cats_event_iah = cats_event_iah.reset_index().set_index(event_level)

    ###########
    #OUTPUT
    df_out = pd.DataFrame(index=macro_event.index)
    
    #aggregates dK and delta_W at df level
    # --> dK, dW are averages per individual
    df_out['dK']        = cats_event_iah[['dk0_pc'    ,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)
    #df_out['dK_public'] = cats_event_iah[['dk_public','pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)
    df_out['delta_W']   = cats_event_iah[['dw'     ,'pcwgt']].prod(axis=1).sum(level=event_level)/cats_event_iah['pcwgt'].sum(level=event_level)
    
    # dktot is already summed with RP -- just add them normally to get losses
    df_out['dKtot']       =      df_out['dK']*cats_event_iah['pcwgt'].sum(level=event_level)#macro_event['pop']
    df_out['delta_W_tot'] = df_out['delta_W']*cats_event_iah['pcwgt'].sum(level=event_level)#macro_event['pop'] 
    # ^ dK and dK_tot include both public and private losses

    if return_stats:
        stats = np.setdiff1d(cats_event_iah.columns,event_level+[i for i in ['helped_cat','affected_cat','hhid','has_received_help_from_PDS_cat'] if i in cats_event_iah.columns])
		
        print('stats are '+','.join(stats))
        df_stats = agg_to_event_level(cats_event_iah,stats,event_level)
        df_out[df_stats.columns]=df_stats 
		    
    if return_iah:
        return df_out,cats_event_iah
    else: 
        return df_out
    
	
def process_output(myCountry,pol_str,out,macro_event,economy,default_rp,return_iah=True,is_local_welfare=False,is_revised_dw=True):

    #unpacks if needed
    if return_iah:
        dkdw_event,cats_event_iah  = out

    else:
        dkdw_event = out

    ##AGGREGATES LOSSES
    #Averages over return periods to get dk_{hazard} and dW_{hazard}
    dkdw_h = average_over_rp1(dkdw_event,default_rp,myCountry,macro_event['protection'])
    print dkdw_h
    dkdw_h.set_index(macro_event.index,inplace = True)
    macro_event[dkdw_h.columns]=dkdw_h

    #computes socio economic capacity and risk at economy level
    macro = calc_risk_and_resilience_from_k_w(macro_event,cats_event_iah,economy,is_local_welfare,is_revised_dw)

    ###OUTPUTS
    if return_iah:
        return macro, cats_event_iah
    else:
        return macro
	
def unpack_social(m,cat):
    """Compute social from gamma_SP, taux tax and k and avg_prod_k"""
    c  = cat.c_pc # this is consumption
    
    # gs = each one's social protection/(total social protection). 
    gs = 0 #cat.gamma_SP FIX THIS   ---> is total social protection per tract??
    
    #Total per capita social assistance for each of the tracts
    assistance_pc = gs*m.gdp_pc_BA*m.tau_tax
    
    # social is fraction of income from social safety net -->defined as t(which is social protection)/c_i(consumption)
    social = assistance_pc/(c+1.0e-10) 
    # gdp*tax should give the total social protection (per capita). 
    
    return social
    
def same_rps_all_hazards(fa_ratios):
    ''' inspired by interpolate_rps but made to make sure all hazards have the same return periods (not that the protection rps are included by hazard)'''
    flag_stack= False
    if 'rp' in get_list_of_index_names(fa_ratios):
        fa_ratios = fa_ratios.unstack('rp')
        flag_stack = True
        
    #in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(fa_ratios.columns)==pd.MultiIndex:
        keys = fa_ratios.columns.get_level_values(0).unique()
        return pd.concat({col:same_rps_all_hazards(fa_ratios[col]) for col in  keys}, axis=1).stack('rp')

    ### ACTUAL FUNCTION    
    #figures out all the return periods to be included
    all_rps = fa_ratios.columns.tolist()
    
    fa_ratios_rps = fa_ratios.copy()
    
    fa_ratios_rps = fa_ratios_rps.reindex_axis(sorted(fa_ratios_rps.columns), axis=1)
    # fa_ratios_rps = fa_ratios_rps.interpolate(axis=1,limit_direction="both",downcast="infer")
    fa_ratios_rps = fa_ratios_rps.interpolate(axis=1,limit_direction="both")
    if flag_stack:
        fa_ratios_rps = fa_ratios_rps.stack('rp')
    
    return fa_ratios_rps    


	
def interpolate_rps(fa_ratios,protection_list,option):
    ###INPUT CHECKING
    default_rp=option
    if fa_ratios is None:
        return None
    
    if default_rp in fa_ratios.index:
        return fa_ratios
            
    flag_stack= False
    if 'rp' in get_list_of_index_names(fa_ratios):
        fa_ratios = fa_ratios.unstack('rp')
        flag_stack = True
 
    if type(protection_list) in [pd.Series, pd.DataFrame]:
        protection_list=protection_list.squeeze().unique().tolist()
        
    #in case of a Multicolumn dataframe, perform this function on each one of the higher level columns
    if type(fa_ratios.columns)==pd.MultiIndex:
        keys = fa_ratios.columns.get_level_values(0).unique()
        return pd.concat({col:interpolate_rps(fa_ratios[col],protection_list,option) for col in  keys}, axis=1).stack('rp')


    ### ACTUAL FUNCTION    
    #figures out all the return periods to be included
    all_rps = list(set(protection_list+fa_ratios.columns.tolist()))
    
    fa_ratios_rps = fa_ratios.copy()
    
    #extrapolates linear towards the 0 return period exposure  (this creates negative exposure that is tackled after interp) (mind the 0 rp when computing probas)
    if len(fa_ratios_rps.columns)==1:
        fa_ratios_rps[0] = fa_ratios_rps.squeeze()
    else:
        fa_ratios_rps[0]=fa_ratios_rps.iloc[:,0]- fa_ratios_rps.columns[0]*(
        fa_ratios_rps.iloc[:,1]-fa_ratios_rps.iloc[:,0])/(
        fa_ratios_rps.columns[1]-fa_ratios_rps.columns[0])
        
    
    #add new, interpolated values for fa_ratios, assuming constant exposure on the right
    x = fa_ratios_rps.columns.values
    y = fa_ratios_rps.values
    fa_ratios_rps= pd.concat(
        [pd.DataFrame(interp1d(x,y,bounds_error=False)(all_rps),index=fa_ratios_rps.index, columns=all_rps)]
        ,axis=1).sort_index(axis=1).clip(lower=0).fillna(method='pad',axis=1)
    fa_ratios_rps.columns.name='rp'

    if flag_stack:
        fa_ratios_rps = fa_ratios_rps.stack('rp')
    
    return fa_ratios_rps    

def agg_to_economy_level (df, seriesname,economy):
    """ aggregates seriesname in df (string of list of string) to economy (country) level using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T*df['pcwgt']).T.sum()#level=economy)
	
def agg_to_event_level (df, seriesname,event_level):
    """ aggregates seriesname in df (string of list of string) to event level (country, hazard, rp) across income_cat and affected_cat using n in df as weight
    does NOT normalize weights to 1."""
    return (df[seriesname].T*df['pcwgt']).T.sum(level=event_level)

def calc_delta_welfare(myC, temp, macro, pol_str,optionPDS,is_revised_dw=True,study=False,
                       labourIncome = False, labour_event = None,nsims = 0,suffix = '',output_folder = ''):
    # welfare cost from consumption before (c) and after (dc_npv_post) event. Line by line
    
    gc.collect()
    mac_ix = macro.index.names
    mic_ix = temp.index.names
    myCountry = myC
    #####################################
    # Collect info from temp before I rewrite it...
    # temp is cats_event_iah
    # Upper limit for per cap dw
    c_pc_mean = None
    try: c_pc_mean = temp[['pcwgt','c_pc']].prod(axis=1).sum()/temp['pcwgt'].sum()
    except: print('could not calculate c_mean! HALT!'); assert(False)

    my_dw_limit = abs(20.*c_pc_mean)
    my_natl_wprime = c_pc_mean**(-const_ie)
    print('c_mean = ',int(c_pc_mean/1E3),'(000s) K \ndw_lim = ',int(my_dw_limit/1E3),'K\nwprime_nat = ',my_natl_wprime,'\n')

    
    macro['k_tot'] = (temp[['pcwgt','k_pc']].prod(axis=1).sum(level=mac_ix)).groupby(level=['hazard','rp']).transform('sum')
    k_tot = macro['k_tot'].mean()
    

    # ^ We're going to recalculate scale_fac_soc at every step
    # --> This couples the success of rich & poor households
    # --> ^ It's a small effect, but the poor do better if the rich recover more quickly

    # Going to separate into a & na now, for speed
    temp = temp.reset_index('affected_cat')

    temp_na = temp.loc[(temp.pcwgt!=0)&(temp.affected_cat=='na')&(temp.help_received==0)&(temp.dc0_pc==0),['affected_cat','pcwgt','dk0_pc','c_pc','dc0_pc','pc_fee']].reset_index().copy()
    # ^ ALL HH that are: not affected AND didn't receive help AND don't have any social income

    temp = temp.loc[(temp.pcwgt!=0)&((temp.affected_cat=='a')|(temp.help_received!=0)|(temp.dc0_pc!=0))].reset_index().copy()
    # ^ ALL HH that are: affected OR received help OR receive social
        
    print('--> length of temp:',temp.shape[0])
    print('--> length of temp_na:',temp_na.shape[0],'\n')

    #############################
    # First debit savings for temp_na
    temp_na['dw'] = temp_na['pc_fee']*(temp_na['c_pc']**(-const_ie))*const_rho
    # ^ assuming every hh has this much savings...

    temp_na['dc_t'] = 0.
    # wprime, dk0

    #temp_na.head(10).to_csv(tmp+'temp_na.csv')
    # Will re-merge temp_na with temp at the bottom...trying to optimize here!

    #############################
    # Drop cols from temp, because it's huge...
    temp = temp.drop([i for i in ['pcinc','hhsize','help_fee','pc_fee_PE','pc_fee_BE',
                                  'index','level_0','axfin','has_ew','macro_multiplier','dc_npv_pre','di0_pc','dc_post_reco',
                                  'gamma_SP','ew_expansion','fa','social','quintile','c_5',
                                  u'peinc_AGR', u'peinc_MIN', u'peinc_UTI', u'peinc_CON', u'peinc_MAN', u'peinc_WHO', u'peinc_RET',
                                  u'peinc_TRA', u'peinc_INF', u'peinc_FIN', u'peinc_PRO', u'peinc_EDU', u'peinc_ART', u'peinc_OTH', 
                                  u'peinc_GOV',u'hh_pub_ass_inc', u'n_hh_inc_less15', u'n_hh_inc_15_25', u'n_hh_inc_25_35', u'n_hh_inc_35_50', 
                                  u'n_hh_inc_50_75', u'n_hh_inc_75_100', u'n_hh_inc_100_125', u'n_hh_inc_125_150', u'n_hh_inc_150_200',
                                  u'n_hh_inc_more200', u'LIL_hh', u'LIL_pc', u'VLIL_hh', u'VLIL_pc', u'ELIL_hh',u'inc_AGR_tot', u'inc_MIN_tot',
                                  u'inc_UTI_tot', u'inc_CON_tot', u'inc_MAN_tot', u'inc_WHO_tot', u'inc_RET_tot', u'inc_TRA_tot', u'inc_INF_tot',
                                  u'inc_FIN_tot', u'inc_PRO_tot',
                                  u'inc_EDU_tot', u'inc_ART_tot', u'inc_OTH_tot', u'inc_GOV_tot'] if i in temp.columns],axis=1)

    ########################################
    # Calculates the revised ('rev') definition of dw
    print('using revised calculation of dw')
    
    # Set-up to be able to calculate integral
    temp['const'] = -1.*(temp['c_pc']**(1.-const_ie))/(1.-const_ie)
    temp['integ'] = 0.0

    my_out_x, my_out_yA, my_out_yNA, my_out_yzero = [], [], [], []
    x_max = 10 # 10 years

    x_min, n_steps = 0.,52.*x_max # <-- time step = week
    if study == True: x_min, x_max, n_steps = 0.,1.,2 # <-- time step = 1 year
    max_treco = math.log(1/0.05)/x_max

    print('('+optionPDS+') Integrating well-being losses over '+str(x_max)+' years after disaster ('+pol_str+')') 
    # ^ make sure that, if T_recon changes, so does x_max!
    
    temp['t_start_prv_reco'] = -1
    temp['t_pov_inc'],temp['t_pov_cons'] = [0., 0.]
    # ^ set these "timers" to collect info about when the hh starts reco, and when it exits poverty

    my_avg_prod_k = macro.avg_prod_k.mean()
    my_tau_tax = macro['tau_tax'].mean()

    if labourIncome:
        temp['dk_pc_labour_t'] = temp['dk_pc_labour'].copy()
        temp['dk_pc_oth_t'] = temp['dk_pc_oth'].copy()
        temp['dk_pc_h_t'] = temp['dk_pc_h'].copy()
        temp['di_pc_labour_t'] = temp['di0_pc_labour'].copy()
        temp['di_pc_rent_t'] = temp['di0_pc_rent'].copy()

        # use this to count down as hh rebuilds
    
        # First, assign savings
        # --> sav_f = intial savings at initialization. will decrement as hh spend savings.
        sav_dir = '../inputs/'+myC+'/'
        #mac_ix[0] is county
        #temp['sav_f'] = get_hh_savings(temp[['pcwgt','c_pc',mac_ix[0],'ispoor']],myC,mac_ix[0],pol_str,sav_dir).round(2)
        temp['sav_pc_f'] = get_hh_savings(temp[['pcwgt','c_pc',mac_ix[0],'savings_per_hh','hhwgt']],myC,mac_ix[0],pol_str,sav_dir).round(2)
        temp = temp.drop([i for i in ['index','province','ispoor'] if i in temp.columns],axis=1)
    
        ################################
        # SAVINGS
        temp['sav_pc_i'] = temp.eval('sav_pc_f+pc_fee').round(2)
        # ^ add pc_fee to sav_f to get sav_i
        # --> this is assuming the government covers the costs for now, and assesses taxes much, much later
    
        # Add help received to sav_f so that it's treated like any savings the hh already had
        temp['sav_pc_f'] += temp['help_received'].round(2)
    
        #print(temp.loc[temp.sav_f<0].shape[0],' hh borrow to pay their fees...')
        # --> let this go...assume that hh will be able to borrow in order to pay that fee
        
        temp = temp.drop(['help_received','pc_fee'],axis=1)
    
    
         # Define parameters of welfare integration
        int_dt,step_dt = np.linspace(x_min,x_max,num=n_steps,endpoint=True,retstep=True)
        print('using time step = ',step_dt)
    
        ################################
        # Use savings (until they run out) to offset dc to optimum level
        temp['sav_offset_to'], temp['t_exhaust_sav'] = [0,0]
        temp= temp.set_index(mic_ix).reset_index(['affected_cat','helped_cat'])
        temp = pd.concat([temp,labour_event],axis = 1).reset_index()
        nsims = len(temp.rp.unique())
     
        
        ########### Apply unemployment insurance, if any
        if pol_str == '_UI_2yrs':
            UI_weeks = np.arange(0,52*1.5,1)/52.
            df_UI =  pickle.load(open('optimization_libs/BA_weekly_UI_2xtime_'+str(nsims)+'sims.p','rb'))
            temp = temp.set_index(['hazard','rp','tract'])
            for i_t,_t in enumerate(UI_weeks):
                temp.loc[:,i_t] = (temp.loc[:,i_t] - df_UI.loc[:,'UI_tot_Week'+str(i_t)]*52).round(2)
            temp = temp.reset_index()
        elif pol_str in ['_baseline','_no_code','_retrofit','_no_ins','_ins40_15','_ins50_15','_ins_poor_15','_ins_poor_50','_ins_rich_15','_ins_rich_50','_ins_rich_30','_ins_poor_30']:
            UI_weeks = np.arange(0,52*0.5,1)/52.
            if pol_str in ['_no_code','_retrofit']:
                df_UI =  pickle.load(open('optimization_libs/BA_weekly_UI_standard'+pol_str+'_'+str(nsims)+'sims.p','rb'))  
            else:
                df_UI =  pickle.load(open('optimization_libs/BA_weekly_UI_standard_'+str(nsims)+'sims.p','rb'))
            temp = temp.set_index(['hazard','rp','tract'])
            for i_t,_t in enumerate(UI_weeks):
                temp.loc[:,i_t] = (temp.loc[:,i_t] - df_UI.loc[:,'UI_tot_Week'+str(i_t)]*52).round(2)
            temp = temp.reset_index()
        elif pol_str == '_no_UI':
            pass
        else:
            raise('INCORRECT POLICY.... STOPPING ANALYSIS')
        #####################################################


        try: 
            print('TRY: load savings optima from file')
            opt_lib = pickle.load(open('optimization_libs/'+myCountry+pol_str+'_'+str(nsims)+'sims/BA_optimal_savings_rate_labour_'+str(nsims)+'sims.p','rb')).to_dict()
            temp['sav_offset_to'] = temp.apply(lambda x:opt_lib['sav_offset_to'][(int(x.c_pc), int(x.dk0_pc), round(x.hh_reco_rate,3), round(float(macro.avg_prod_k.mean()),3), int(x.sav_pc_f))],axis=1)
            temp['t_exhaust_sav'] = temp.apply(lambda x:opt_lib['t_exhaust_sav'][(int(x.c_pc), int(x.dk0_pc), round(x.hh_reco_rate,3), round(float(macro.avg_prod_k.mean()),3), int(x.sav_pc_f))],axis=1)
            temp['uniform_savings'] = temp.apply(lambda x:opt_lib['uniform_savings'][(int(x.c_pc), int(x.dk0_pc), round(x.hh_reco_rate,3), round(float(macro.avg_prod_k.mean()),3), int(x.sav_pc_f))],axis=1)
            
            print('SUCCESS!')
            #del opt_lib
        except: 
            print('FAIL: finding optimal savings numerically')
            temp['i'] = temp.index.values
            temp[['sav_offset_to','t_exhaust_sav','uniform_savings']] = temp.apply(lambda x:pd.Series(smart_savers(x.c_pc,x.k_pc, x.dk0_pc, x.hh_reco_rate,macro.avg_prod_k.mean(),\
                x.sav_pc_f,x.i,x.loc[np.arange(0,520)],x.k_pc_oth,x.v_with_ew,x.hh_share,x.k_pc_str,x.k_pc_h,x.p_rent_pc,labourIncome)),axis=1)
            temp = temp.drop(columns = ['i'])
            opt_in = temp[['c_pc','dk0_pc','hh_reco_rate','sav_pc_f','sav_offset_to','t_exhaust_sav','uniform_savings']].copy()
            opt_in['avg_prod_k'] = macro.avg_prod_k.mean()
        
            opt_in[['c_pc','dk0_pc','sav_pc_f']] = opt_in[['c_pc','dk0_pc','sav_pc_f']].astype('int')
            opt_in[['hh_reco_rate','avg_prod_k']] = opt_in[['hh_reco_rate','avg_prod_k']].round(3)
    
            opt_in = opt_in.reset_index().set_index(['c_pc','dk0_pc','hh_reco_rate','avg_prod_k','sav_pc_f']).drop('index',axis=1)
            pickle.dump(opt_in.loc[opt_in.index.unique()], open('optimization_libs/'+myCountry+pol_str+'_'+str(nsims)+'sims/BA_optimal_savings_rate_labour_'+str(nsims)+'sims.p', 'wb' ))    
           # del opt_in; del opt_lib
                 
       
        temp['intermed_sf_num'], temp['di_pc_t'], temp['di_pc_oth_t'], temp['di_pc_h_t'], \
        temp['di_pc_pub_t'], temp['dc_pc_t'], temp['dc_pc_net'] = [0,0,0,0,0,0,0] 
        
        # Set-up consumption vector
        temp_cons = pd.DataFrame(index=temp.index)
        temp_cons['c_pc'] = temp['c_pc']
        temp_cons['pcwgt'] = temp['pcwgt']
        temp_cons['tract'] = temp['tract']
        temp_cons['rp'] = temp['rp']

        counter = 0
        # Calculate integral
        for i_t,_t in enumerate(int_dt):
            print '..'+str(round(10*_t,0))+'%'

            # Recalculate scale_fac_soc based on reconstruction of all assets
            # NB: scale_fac_soc was initialized above
            temp['intermed_sf_num'].update(temp.eval('pcwgt*(dk_pc_labour_t+dk_pc_oth_t+dk_pc_h_t+dk_pc_public*@math.e**(-@_t*@const_pub_reco_rate))'))
    
            temp = temp.reset_index().set_index(mac_ix)
            temp['scale_fac_soc'].update(temp.groupby(level=mac_ix)['intermed_sf_num'].transform('sum')/k_tot)
            temp = temp.reset_index().drop([i for i in ['index','level_0'] if i in temp.columns],axis=1)
    
            # BELOW: this is value of dc at time _t (duration = step_dt), assuming no savings
            temp['di_pc_labour_t'].update(temp.eval('dk_pc_labour_t*@my_avg_prod_k*(1-@my_tau_tax) + pcsoc*scale_fac_soc').round(2))
            temp['di_pc_pub_t'].update(temp.eval('di0_pc_pub*@math.e**(-@_t*@const_pub_reco_rate)').round(2))
            temp['di_pc_oth_t'].update(temp.eval('dk_pc_oth_t*@my_avg_prod_k*(1-@my_tau_tax)').round(2)) ## CHECK THIS
            temp['di_pc_h_t'].update(temp.eval('dk_pc_h_t*@my_avg_prod_k*(1-@my_tau_tax)').round(2)) ## CHECK THIS
            
            
            temp['di_pc_t'].update(temp.eval('di_pc_labour_t + di_pc_pub_t + di_pc_oth_t + di_pc_h_t - di_pc_rent_t').round(2))
            #temp['di_t'] = temp.eval('di_prv_t + di_pub_t - help_received*@const_pds_rate*@math.e**(-@_t*@const_pds_rate)')
            ####################################        
            # Calculate di(t) & dc(t) 
            # di(t) won't change again within this time step, but dc(c) could
            #temp['di_t'] = temp.eval('di_prv_t+di_pub_t-help_received*@const_pds_rate*@math.e**(-@_t*@const_pds_rate)')      
            temp['dc_pc_t'].update(temp.eval('di_pc_t+hh_reco_rate*v*k_pc_str*hh_share*@math.e**(-@_t*hh_reco_rate)').round(2))
            #assert(temp.loc[temp.di_pc_t<0].shape[0] == 0)

            
            ####################################
            # Let the hh optimize (pause, (re-)start, or adjust) its reconstruction
            # NB: this doesn't change income--just consumption
            
            # welf_class == 1: let them run
            recalc_crit_1 = '(welf_class==1) & (hh_reco_rate!=0) & (dk_pc_labour_t+dk_pc_oth_t + dk_pc_h_t != 0) & (dc_pc_t>c_pc)'
            if temp.loc[temp.eval(recalc_crit_1)].shape[0] != 0: 
                assert(False)
                
            neg_bug_crit = '(dc_pc_t<0)'
            if temp.loc[temp.eval(neg_bug_crit)].shape[0] != 0: 
                print 'Consumption increases in ', temp.loc[temp.eval(neg_bug_crit)].shape[0],' census tracts'
                #temp.loc[temp.eval(neg_bug_crit)].drop(columns = np.arange(0,520)).to_csv(tmp+'increased_consumption'+str(counter)+'.csv')
                
    
            # welf_class == 2: keep consumption at subsistence until they can afford macro_reco_frac without going into subsistence
            recalc_crit_2 = '(welf_class==2 | welf_class==3) & (hh_reco_rate!=0) & (dk_pc_labour_t+dk_pc_oth_t  +dk_pc_h_t !=0) & (hh_reco_rate < optimal_hh_reco_rate | dc_pc_t > c_pc )'
            recalc_hhrr_2 = '(c_pc-di_pc_t-c_pc_min)/(v_with_ew*k_pc_str*hh_share)' # how much the hh can afford to reconstruct this year
    
            # welf_class == 3: if they recover income above subsistence, apply welf_class == 2 rules
            start_criteria  = '(welf_class==3) & (hh_reco_rate==0) & (dk_pc_labour_t+dk_pc_oth_t + dk_pc_h_t != 0) & (c_pc-di_pc_t > c_pc_min)' # Subsistence escape criteria
            stop_criteria   = '(welf_class==3) & (hh_reco_rate!=0) & (c_pc-di_pc_t < c_pc_min)' # Subsistence criteria
    
            print('('+optionPDS+' - t = '+str(round(_t*52,1))+' weeks after disaster; '
                  +str(round(100*_t/x_max,1))+'% through reco): '
                  +str(temp.loc[temp.eval(start_criteria)].shape[0])+' hh escape subs & '
                  +str(temp.loc[temp.eval(recalc_crit_2)].shape[0])+' recalc in wc=2 & '
                  +str(temp.loc[temp.eval(stop_criteria)].shape[0])+' stop reco\n')
    
            temp.loc[temp.eval(recalc_crit_2),'hh_reco_rate'] = temp.loc[temp.eval(recalc_crit_2)].eval(recalc_hhrr_2).round(3)
            temp.loc[temp.eval(start_criteria),'hh_reco_rate'] = temp.loc[temp.eval(start_criteria)].eval(recalc_hhrr_2).round(3)
            temp.loc[temp.eval(stop_criteria),'hh_reco_rate'] = 0.
                       
            ####################################        
            # Calculate dc(t) 
            temp['dc_pc_t'].update(temp.eval('di_pc_t+hh_reco_rate*v_with_ew*k_pc_str*hh_share*@math.e**(-@_t*hh_reco_rate)').round(2))
            
            
            if temp.loc[temp.hh_reco_rate<0].shape[0] != 0:
                print('Found',temp.loc[temp.hh_reco_rate<0].shape[0],'hh with negative reco_rate! FATAL!')
                #temp.loc[temp.hh_reco_rate<0].to_csv('tmp/fatal_neg_hh_reco_rate.csv')
                assert(False)
    
            if temp.loc[temp.eval('dc_pc_t>c_pc')].shape[0] != 0:
                print('Finding hh with dc_t > c !! Fatal error!!')
                #temp.loc[temp.eval('dc_pc_t>c_pc')].to_csv('tmp/fatal_neg_c.csv')
                assert(False)
    
            ########################
            # Now apply savings (if any left) #### CHANGED BY MM
            #
            if False:
                # If there are still any savings after the hh was supposed to run out, use up the rest of savings
                temp.loc[temp.t_exhaust_sav<=_t,'sav_offset_to'] = 0.
            
                # Find dC net of savings (min = sav_offset_to if dc_t > 0  -OR-  min = dc_t if dc_t < 0 ie: na & received PDS)
                temp['dc_pc_net'].update(temp['dc_pc_t'])
                
                sav_criteria = '(sav_pc_f>0.1)&(dc_pc_t>sav_offset_to)' # there are savings and the consumption
                sav_criteria_2a = sav_criteria+'&(dc_pc_net!=dc_pc_t)&(hh_reco_rate!=0)'
                sav_criteria_2b = sav_criteria+'&(dc_pc_net!=dc_pc_t)&(hh_reco_rate==0)' # no housing recovery
        
                # This is how much dc is offset by savings
                temp.loc[temp.eval(sav_criteria),'dc_pc_net'] = temp.loc[temp.eval(sav_criteria)].eval('dc_pc_t-sav_pc_f/@step_dt').round(2)
                temp.loc[temp.eval(sav_criteria),'dc_pc_net'] = temp.loc[temp.eval(sav_criteria),'dc_pc_net'].clip(lower=temp.loc[temp.eval(sav_criteria),['sav_offset_to','dc_pc_t']].min(axis=1).squeeze())
        
                temp['sav_delta'] = 0
                _dsav_a = '(dc_pc_t/hh_reco_rate)*(1-@math.e**(-hh_reco_rate*@step_dt))-dc_pc_net*@step_dt'
                _dsav_b = '@step_dt*(dc_pc_t-dc_pc_net)'
                temp.loc[temp.eval(sav_criteria_2a),'sav_delta'] = temp.loc[temp.eval(sav_criteria_2a)].eval(_dsav_a).round(2)
                temp.loc[temp.eval(sav_criteria_2b),'sav_delta'] = temp.loc[temp.eval(sav_criteria_2b)].eval(_dsav_b).round(2)
                
                # Adjust savings after spending    
                temp['sav_pc_f'] -= temp['sav_delta']
                
                # Sanity check: savings should not go negative
                savings_check = '(sav_pc_i>0.)&(sav_pc_f+0.1<0.)'
                if temp.loc[temp.eval(savings_check)].shape[0] != 0:
                    print('Some hh overdraft their savings!')
                    #temp.loc[temp.eval(savings_check)].to_csv(tmp+'bug_negative_savings_'+str(counter)+'.csv')
                    #assert(False)
            else:
                 # If there are still any savings after the hh was supposed to run out, use up the rest of savings
                temp.loc[temp.t_exhaust_sav<=_t,'sav_offset_to'] = temp['dc_pc_t']
                
               ##### NEW MARYIA SAVINGS CODE
                temp['dc_pc_net'].update(temp['dc_pc_t'])
                temp['sav_delta'] = 0
                
                sav_criteria = '(sav_pc_f>0.1)&(dc_pc_t>sav_offset_to)&(uniform_savings ==0)'
                _dsav = '@step_dt*(dc_pc_t-sav_offset_to)'
                temp.loc[temp.eval(sav_criteria),'sav_delta'] = temp.loc[temp.eval(sav_criteria)].eval(_dsav).round(2)
                temp.loc[temp.eval(sav_criteria),'dc_pc_net'] = temp.loc[temp.eval(sav_criteria),'dc_pc_net'].clip(lower=temp.loc[temp.eval(sav_criteria),['sav_offset_to','dc_pc_t']].min(axis=1).squeeze())
                
                sav_crit_uniform = '(uniform_savings !=0)'
                _dc_uniform = 'dc_pc_t-sav_delta/@step_dt'
                temp.loc[temp.eval(sav_crit_uniform),'sav_delta'] = temp.loc[temp.eval(sav_crit_uniform),'uniform_savings'].round(2)*step_dt
                temp.loc[temp.eval(sav_crit_uniform),'dc_pc_net'] = temp.loc[temp.eval(sav_crit_uniform)].eval(_dc_uniform).clip(lower=0)
                temp.loc[temp.eval(sav_crit_uniform),'sav_offset_to'] = temp.loc[temp.eval(sav_crit_uniform),'dc_pc_net']
                
                _fix_overdraft = 'sav_pc_f<sav_delta'
                _dc_overdraft = 'sav_delta/@step_dt+sav_offset_to'
                temp.loc[temp.eval(_fix_overdraft),'sav_delta'] = temp.loc[temp.eval(_fix_overdraft),'sav_pc_f']
                temp.loc[temp.eval(_fix_overdraft),'dc_pc_net'] = temp.loc[temp.eval(_fix_overdraft)].eval(_dc_overdraft).round(2) 
                temp.loc[temp.eval(_fix_overdraft),'t_exhaust_sav'] = _t
                temp.loc[temp.eval(_fix_overdraft),'uniform_savings'] = 0
              #########################
                
              ##### OLD BRIAN SAVINGS CODE
              #   # Find dC net of savings (min = sav_offset_to if dc_t > 0  -OR-  min = dc_t if dc_t < 0 ie: na & received PDS)
              #  temp['dc_pc_net'].update(temp['dc_pc_t'])
              #  
              #  sav_criteria = '(sav_pc_f>0.1)&(dc_pc_t>sav_offset_to)' # there are savings and the consumption
              #  sav_criteria_2a = sav_criteria+'&(dc_pc_net!=dc_pc_t)&(hh_reco_rate!=0)'
              #  sav_criteria_2b = sav_criteria+'&(dc_pc_net!=dc_pc_t)&(hh_reco_rate==0)' # no housing recovery
        
              #   # This is how much dc is offset by savings
              #  #temp.loc[temp.eval(sav_criteria),'dc_pc_net'] = temp.loc[temp.eval(sav_criteria)].eval('dc_pc_t-sav_pc_f/@step_dt').round(2)
              #  temp.loc[temp.eval(sav_criteria),'dc_pc_net'] = temp.loc[temp.eval(sav_criteria),'dc_pc_net'].clip(lower=temp.loc[temp.eval(sav_criteria),['sav_offset_to','dc_pc_t']].min(axis=1).squeeze())
              #  temp['sav_delta'] = 0
                
                if i_t<519: temp['labour_inc_next_t'] = temp.loc[:,i_t+1]
                else: temp['labour_inc_next_t'] = temp.loc[:,519] 

              #  _dsav_a = ('(labour_inc_next_t+di_pc_labour_t)*@step_dt/2'+ 
              #  '+ @my_avg_prod_k *v_with_ew*k_pc_h*(@math.e**(-hh_reco_rate*@_t)-@math.e**(-hh_reco_rate*(@_t+@step_dt)))/hh_reco_rate'+
              #  '-p_rent_pc*v_with_ew*(@math.e**(-hh_reco_rate*@_t)-@math.e**(-hh_reco_rate*(@_t+@step_dt)))/hh_reco_rate'+
              #  '+v_with_ew*k_pc_str*hh_share*(@math.e**(-hh_reco_rate*@_t)-@math.e**(-hh_reco_rate*(@_t+@step_dt)))'+
              #  '-dc_pc_net*@step_dt')
              #  _dsav_b = '@step_dt*(dc_pc_t-dc_pc_net)'
              #  temp.loc[temp.eval(sav_criteria_2a),'sav_delta'] = temp.loc[temp.eval(sav_criteria_2a)].eval(_dsav_a).round(2)
              #  temp.loc[temp.eval(sav_criteria_2b),'sav_delta'] = temp.loc[temp.eval(sav_criteria_2b)].eval(_dsav_b).round(2)
              #######################################
               
                # Adjust savings after spending    
                temp['sav_pc_f'] -= temp['sav_delta']
                
                # Sanity check: savings should not go negative
                savings_check = '(sav_pc_i>0.)&(sav_pc_f+1<0.)'
                if temp.loc[temp.eval(savings_check)].shape[0] != 0:
                    print('Some hh overdraft their savings!')
                    print temp.loc[temp.eval(savings_check),'sav_pc_f']
                    #temp.loc[temp.eval(savings_check)].to_csv(tmp+'bug_negative_savings_'+str(counter)+'.csv')
                    assert(False)
                
            ########################
            # Increment time in poverty
            temp.loc[temp.eval('c_pc-di_pc_t<=pov_lev_pc'),'t_pov_inc'] += step_dt
            temp.loc[temp.eval('c_pc-dc_pc_net<=pov_lev_pc'),'t_pov_cons'] += step_dt
        
            ########################  
            # Finally, calculate welfare losses
            temp['integ'] += temp.eval('@step_dt*((1.-dc_pc_net/c_pc)**(1.-@const_ie)-1.)*@math.e**(-@_t*@const_rho)')
    
           # if temp.loc[temp.integ<0].shape[0] != 0:
           #     temp.loc[temp.integ<0].drop(columns = np.arange(0,520)).to_csv('tmp/negative_integ_iter'+str(i_t)+'.csv')
                #assert(False)
                                          
            if temp.shape[0] != temp.dropna(subset=['integ']).shape[0]:
                temp['integ'] = temp['integ'].fillna(-1E9)
                #temp.loc[temp.integ==-1E9].to_csv('tmp/fatal_integ_null.csv')
                assert(False)
            # Write the consumption
            temp_cons['cons_t'+str(i_t)] = temp['dc_pc_net']
            # NB: no expansion here because dc_net already gives the instantateous value at t = _t
            # --> Could take the average within each time step (step_dt)
    
            # Decrement dk(t)
            _dk_pc_oth_exp = 'dk_pc_oth_t - hh_reco_rate*dk_pc_oth*@math.e**(-hh_reco_rate*@_t)*@step_dt'
            _dk_pc_h_exp = 'dk_pc_h_t - hh_reco_rate*dk_pc_h*@math.e**(-hh_reco_rate*@_t)*@step_dt'
            _di_pc_rent_exp = 'di_pc_rent_t - hh_reco_rate*di0_pc_rent*@math.e**(-hh_reco_rate*@_t)*@step_dt'
            temp.loc[temp.eval('dk_pc_oth_t>0'),'dk_pc_oth_t'] = temp.loc[temp.eval('dk_pc_oth_t>0')].eval(_dk_pc_oth_exp).round(2).clip(0) ## CHECK THIS
            temp.loc[temp.eval('dk_pc_h_t>0'),'dk_pc_h_t'] = temp.loc[temp.eval('dk_pc_h_t>0')].eval(_dk_pc_h_exp).round(2).clip(0) ## CHECK THIS
            temp.loc[temp.eval('di_pc_rent_t>0'),'di_pc_rent_t'] = temp.loc[temp.eval('di_pc_rent_t>0')].eval(_di_pc_rent_exp).round(2).clip(0) ## CHECK THIS
            

            if i_t < len(int_dt)-1:
                temp['dk_pc_labour_t'] = temp.loc[:,i_t+1]/my_avg_prod_k
            else:
                temp['dk_pc_labour_t'] = temp.loc[:,len(int_dt)-1]/my_avg_prod_k
            
                
    
            # Sanity check: dk_prv_t should not be higher than dk_private (initial value)
            if temp.loc[(temp.dk_pc_oth_t+temp.dk_pc_labour_t+temp.dk_pc_h_t>temp.dk_pc_oth+temp.dk_pc_labour+temp.dk_pc_h+0.01)].shape[0] > 0:
                print('Some hh lose more at the end than in the disaster!')
                #temp.loc[(temp.dk_pc_oth_t+temp.dk_pc_labour_t+temp.dk_pc_h_t>temp.dk_pc_oth+temp.dk_pc_labour+temp.dk_pc_h+0.01)].to_csv(tmp+'bug_ghost_losses'+optionPDS+'_'+str(counter)+'.csv')
                #assert(False)  
    
            # Save out the files for debugging
            #if ((counter<=10) or (counter%50 == 0)): temp.head(10).to_csv(tmp+'temp_'+optionPDS+pol_str+'_'+str(counter)+'.csv')
    
            counter+=1
            gc.collect()
    
            # NB: dc0 has both public and private capital in it...could rewrite to allow for the possibility of independent reconstruction times
            # NB: if consumption goes negative, the integral can't be calculated...death!
            # NB: if consumption increases, dw will come out negative. that's just making money off the disaster (eg from targeting error)
    
        ################################
        # Write out the poverty duration info
        #temp[[mac_ix[0],'hazard', 'rp', 'pcwgt', 'c_pc', 'dk0_pc', 't_pov_inc', 't_pov_cons', 
        #      't_start_prv_reco', 'hh_reco_rate', 'optimal_hh_reco_rate']].to_csv('../output_country/'+myCountry+'_'+str(nsims)+'sims/'+output_folder+'/poverty_duration_'+optionPDS+'.csv')
    
        ################################
        # 'revised' calculation of dw
        temp['wprime_hh']       = temp.eval('c_pc**(-@const_ie)')
        temp['dw_curr_no_clip'] = temp.eval('const*integ/@my_natl_wprime')
        
        ####### DEBUG INTEGRAL RESULTS
        #temp['dw_test_intergal'] = temp.eval('const*integ')
        #temp['dw_test_savings'] = temp.eval('(sav_pc_i-sav_pc_f)*wprime_hh')
        #temp['dsavings'] = temp.eval('sav_pc_i-sav_pc_f')
        #temp.loc[:,['dw_test_intergal','dw_test_savings','dsavings']].to_csv('tmp/test_integral_components.csv')
        #temp = temp.drop(['dw_test_intergal','dw_test_savings','dsavings'],axis=1)


        temp['dw'] = temp.eval('const*integ+(sav_pc_i-sav_pc_f)*wprime_hh')
        #                       ^ dw(dc)   ^ dw(spent savings)
        temp['dw'] = temp['dw'].clip(upper=my_dw_limit*my_natl_wprime)
        # apply upper limit to DW
         
        # Re-merge temp and temp_na
        temp = pd.concat([temp,temp_na]).reset_index().set_index([i for i in mic_ix]).sort_index()
        temp['dc_pc_net'] = temp['dc_pc_net'].fillna(temp['dc_pc_t'])
    
        if temp['dw'].shape[0] != temp.dropna(subset=['dw']).shape[0]:
            temp['dw'] = temp['dw'].fillna(-1E9)
            #temp.loc[temp.dw==-1E9].to_csv('tmp/fatal_dw_null.csv')
            print(temp.loc[temp.dw==-1E9].shape[0],' hh have NaN dw')
            assert(False)
    
        print('dw:',temp['dw'].shape[0],'dw.dropna:',temp.dropna(subset=['dw']).shape[0])
        assert(temp['dc_pc_t'].shape[0] == temp['dc_pc_t'].dropna().shape[0])
        assert(temp['dc_pc_net'].shape[0] == temp['dc_pc_net'].dropna().shape[0])
        assert(temp['dw'].shape[0] == temp.dropna(subset=['dw']).shape[0])    
    
        # Divide by dw'
        temp['dw_curr'] = temp['dw']/my_natl_wprime
    
        print('Applying upper limit for dw = ',round(my_dw_limit,0))
        temp['dw_tot'] = temp[['dw_curr','pcwgt']].prod(axis=1)
        temp = temp.drop([int(x) for x in np.arange(1,520)],axis = 1)
        #temp = temp.drop(columns = ["%i" % x for x in np.arange(1,520)])
        temp.to_csv('../output_country/'+myCountry+'_'+str(nsims)+'sims/'+output_folder+'/my_summary_'+optionPDS+'_all_tracts'+pol_str+'_'+str(nsims)+'sims.csv')
        temp_cons.to_csv('../output_country/'+myCountry+'_'+str(nsims)+'sims/'+output_folder+'/consumption_summary_'+optionPDS+'_all_tracts'+pol_str+'_'+str(nsims)+'sims.csv')
        #temp.loc[(temp.pcwgt!=0)&(temp.dw_curr==my_dw_limit)].to_csv(tmp+'my_late_excess'+optionPDS+'.csv')
    
        print('\n ('+optionPDS+' Total well-being losses ('+pol_str+'):',temp[['dw_curr_no_clip','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6)
            
        ##################################
        # Write summary stats
        temp = temp.reset_index().set_index([i for i in mic_ix]).reset_index(level='affected_cat')
        tmp_out = pd.DataFrame(index=temp.sum(level=[i for i in mac_ix]).index)
    
        tmp_out['dk_tot'] = temp[['dk0_pc','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
        tmp_out['dw_tot'] = temp[['dw_curr','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
        tmp_out['res_tot'] = tmp_out['dk_tot']/tmp_out['dw_tot']
    
        tmp_out['dk_lim'] = temp.loc[(temp.dw_curr==my_dw_limit),['dk0_pc','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6    
        tmp_out['dw_lim'] = temp.loc[(temp.dw_curr==my_dw_limit),['dw_curr','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
        tmp_out['res_lim'] = tmp_out['dk_lim']/tmp_out['dw_lim']
    
        tmp_out['dk_sub'] = temp.loc[(temp.dw_curr!=my_dw_limit),['dk0_pc','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
        tmp_out['dw_sub'] = temp.loc[(temp.dw_curr!=my_dw_limit),['dw_curr','pcwgt']].prod(axis=1).sum(level=[i for i in mac_ix])/1.E6
        tmp_out['res_sub'] = tmp_out['dk_sub']/tmp_out['dw_sub']
    
        tmp_out['ratio_dw_lim_tot']  = tmp_out['dw_lim']/tmp_out['dw_tot']
    
        tmp_out['avg_reco_t']         = (np.log(1/0.05)/temp.loc[(temp.affected_cat=='a')&(temp.hh_reco_rate!=0),'hh_reco_rate']).mean(skipna=True,level=[i for i in mac_ix])
        tmp_out['sub_avg_reco_t']     = (np.log(1/0.05)/temp.loc[(temp.affected_cat=='a')&(temp.welf_class==3)&(temp.hh_reco_rate!=0),'hh_reco_rate']).mean(skipna=True,level=[i for i in mac_ix])
        tmp_out['non_sub_avg_reco_t'] = (np.log(1/0.05)/temp.loc[(temp.affected_cat=='a')&(temp.welf_class!=3)&(temp.hh_reco_rate!=0),'hh_reco_rate']).mean(skipna=True,level=[i for i in mac_ix])
        tmp_out['pct_subs'] = temp.loc[(temp.affected_cat=='a')&(temp.hh_reco_rate==0),'pcwgt'].sum(level=[i for i in mac_ix])/temp.loc[(temp.affected_cat=='a'),'pcwgt'].sum(level=[i for i in mac_ix])
    
        #tmp_out.to_csv('../output_country/'+myC+'/my_summary_'+optionPDS+pol_str+'.csv')
        tmp_out_aal,_ = average_over_rp(tmp_out[['dk_tot','dw_tot']])
        tmp_out_aal['resilience'] = tmp_out_aal['dk_tot']/tmp_out_aal['dw_tot']
        #tmp_out_aal.to_csv('../output_country/'+myC+'/my_summary_aal_'+optionPDS+pol_str+'.csv')
        print('Wrote out summary stats for dw ('+optionPDS+'/'+pol_str+')')
        ##################################
    
        temp = temp.reset_index().set_index([i for i in mic_ix])
        return temp['dc_pc_net'], temp['dw']


def welf1(c,elast,comp):
    """"Welfare function"""
    y=(c**(1-elast)-1)/(1-elast)
    row1 = c<comp
    row2 = c<=0
    y[row1]=(comp**(1-elast)-1)/(1-elast) + comp**(-elast)*(c-comp)
#    y[row2]=(comp**(1-elast)-1)/(1-elast) + comp**(-elast)*(0-comp)
    return y
	
def welf(c,elast):
    y=(c**(1-elast)-1)/(1-elast)
    return y
	
def calc_risk_and_resilience_from_k_w(df, cats_event_iah,economy,is_local_welfare,is_revised_dw): 
    """Computes risk and resilience from dk, dw and protection. Line by line: multiple return periods or hazard is transparent to this function"""
    df=df.copy()    
    ############################
    #Expressing welfare losses in currency 
    #discount rate
    h=1e-4

    if is_revised_dw:
        #if is_local_welfare or not is_local_welfare:
        # ^ no dependence on this flag, for now
        c_mean = cats_event_iah[['pcwgt','c_pc']].prod(axis=1).sum()/cats_event_iah['pcwgt'].sum()
        wprime = c_mean**(-const_ie)

    if not is_revised_dw:
        print('Getting wprime (legacy)')
        if is_local_welfare:
            wprime =(welf(df['gdp_pc_county']/const_rho+h,df['income_elast'])-welf(df['gdp_pc_county']/const_rho-h,df['income_elast']))/(2*h*(1-const_ie))
            print('Adding (1-eta) to denominator of legacy wprime...')
        else:
            wprime =(welf(df['gdp_pc_BA']/const_rho+h,df['income_elast'])-welf(df['gdp_pc_BA']/const_rho-h,df['income_elast']))/(2*h*(1-const_ie))
            print('Adding (1-eta) to denominator of legacy wprime...') 
    
    dWref = wprime*df['dK']
    #dWref = wprime*(df['dK']-df['dK_public'])
    # ^ doesn't add in dW from transfers from other provinces...

    #expected welfare loss (per family and total)
    df['wprime'] = wprime
    df['dWref'] = dWref
    df['dWpc_currency'] = df['delta_W']/wprime 
    df['dWtot_currency']=df['dWpc_currency']*cats_event_iah['pcwgt'].sum(level=[economy,'hazard','rp'])#*df['pop']
    
    #Risk to welfare as percentage of local GDP
    df['risk']= df['dWpc_currency']/(df['gdp_pc_county'])
    
    ############
    #SOCIO-ECONOMIC CAPACITY)
    df['resilience'] =dWref/(df['delta_W'] )

    ############
    #RISK TO ASSETS
    df['risk_to_assets']  =df.resilience*df.risk
    
    return df


def get_labour_df(cats_event,intermediate,nsims,x_max,fault_identifier): 
    try:
        labour_event = pickle.load(open(intermediate+'/labour_recovery_df_'+fault_identifier+'_'+str(nsims)+'sims.p','rb'))
        return labour_event
    except:
        x_min, n_steps = 0.,52.*x_max # <-- time step = week
        int_dt,step_dt = np.linspace(x_min,x_max,num=n_steps,endpoint=True,retstep=True)
        loaded_list = pickle.load(open(intermediate+'/labour_recovery_'+fault_identifier+'_'+str(nsims)+'sims.p','rb'))
        L_pcinc_dic = loaded_list[0]
        L_recov_dic = loaded_list[1]
        ind_names = ['AGR','MIN','UTI','CON','MAN','WHO','RET','TRA','INF','FIN','PRO','EDU','ART','OTH','GOV']
        
        ### MEMORY EFFICIENT APPROACH
        index_labels = [list(x) for x in L_pcinc_dic['AGR'].index.levels]
        index_labels.pop(0)
        index_labels.append(ind_names)
        labour_inc_df = pd.DataFrame(index = pd.MultiIndex.from_product(index_labels,names = ['tract','industry'])).reset_index('industry')
        for ind in ind_names:
            labour_inc_df.loc[labour_inc_df.industry == ind,'pcinc_ind'] = L_pcinc_dic[ind].reset_index('county')['pcinc_'+ind]
            labour_inc_df.loc[labour_inc_df.industry == ind,'county'] = L_pcinc_dic[ind].reset_index('county')['county']
        labour_inc_df = labour_inc_df.reset_index().set_index(['tract'])
        
        labour_loss_df =  pd.DataFrame.from_dict(L_recov_dic,orient = 'index').round(4).reset_index()
        labour_loss_df[['industry','rp']] = labour_loss_df['index'].apply(pd.Series)
        labour_loss_df = labour_loss_df.drop(columns=['index']).set_index(['rp','industry'])
      
        
        
        index_labels_2 = []
        index_labels_2.append([x for x in np.arange(1,nsims+1,1)])
        index_labels_2.append(index_labels[0])

        labour_event = pd.DataFrame(0,index = pd.MultiIndex.from_product(index_labels_2,names = ['rp','tract']),columns = np.arange(0,int(n_steps),1))
        labour_event.reset_index('rp',inplace = True)
        temp_labour_df = pd.DataFrame(0,index = pd.Index(index_labels_2[1],name = 'tract'),columns = np.arange(0,int(n_steps),1))
        for sim in index_labels_2[0]:
            labour_event.loc[labour_event.rp == sim,'county'] = labour_inc_df.loc[labour_inc_df.industry == ind,'county']
            print sim
            for ind in ind_names:
                temp_labour_df.loc[:,:] = labour_loss_df.loc[(sim,ind),:].values
                labour_event.loc[labour_event.rp == sim,np.arange(0,int(n_steps))] +=temp_labour_df.loc[:,:].multiply( labour_inc_df.loc[labour_inc_df.industry == ind,'pcinc_ind'] ,axis= "index")
                
        labour_event = labour_event.reset_index()
        labour_event['hazard'] = 'EQ'
        labour_event.set_index(['county','hazard','rp','tract'],inplace = True)
        
        ### MEMORY INEFFICIENT APPTOACH
        #Define the labour losses data frame
        '''
        labour_loss_df =  pd.DataFrame.from_dict(L_recov_dic,orient = 'index').round(4).reset_index()
        labour_loss_df[['industry','rp']] = labour_loss_df['index'].apply(pd.Series)
        labour_loss_df = labour_loss_df.drop(columns=['index']).set_index(['rp','industry'])
        labour_loss_df = broadcast_simple(labour_loss_df,cats_event.index.levels[3])
        
        index_labels = [list(x) for x in L_pcinc_dic['AGR'].index.levels]
        index_labels.pop(0)
        index_labels.append(ind_names)
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
        '''
        ###################

        pickle.dump(labour_event, open(intermediate+'/labour_recovery_df_'+fault_identifier+'_'+str(nsims)+'sims.p', 'wb' ))
        return labour_event
