#from IPython import get_ipython
#get_ipython().magic('reset -f')
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')
import matplotlib
#matplotlib.use('AGG')

import gc
import sys
import os, time
import warnings
import numpy as np
import pandas as pd

from libraries.lib_country_dir import *
from libraries.lib_compute_resilience_and_risk import *
from libraries.replace_with_warning import *

from multiprocessing import Pool
from itertools import repeat
from itertools import product

def launch_compute_resilience_and_risk_thread(myCountry,pol_str='',optionPDS='no'):

    #dw = calc_delta_welfare(None,None,is_revised=True,study=True)
    # Use this to study change in definition of dw
    # ^ should be deleted when we're happy with the change
    
    warnings.filterwarnings('always',category=UserWarning)

    # Setup directories
    output       = model+'/../output_country/'+myCountry+'/'
    intermediate = model+'/../intermediate/'+myCountry+'/'
    if not os.path.exists(output):
        os.makedirs(output)

    ########################################################
    ########################################################

    # How is it paid for? 
    # --> 'tax' = based on income 
    # --> 'insurance_premium' = based on vulnerability
    optionFee = 'tax'

    if optionFee=='insurance_premium':
        optionB='unlimited'
        optionT='perfect'
    else:
        optionB='data'
        optionT='perfect'#'data'    

    # Cap on benefits (bool)
    option_CB = 1 #0 is for calculation of benefits only; 1 by default

    if option_CB==0: option_CB_name = 'benefits_only'
    else: option_CB_name = ''

    # Below: this is just so that social protection plus (SPP) can be passed in the starmap as a policy alternative.
    if pol_str == 'fiji_SPP' or pol_str == 'fiji_SPS':
        if optionPDS == 'no':
            optionPDS = pol_str
            pol_str   = ''
        else: return False # Just don't want to run fiji_SPP multiple times

    if 'nosavings' in optionPDS: # '_nosavings' and '_nosavingsdata'
        pol_str = optionPDS[:]
        optionPDS = 'no'

    # Show what we're running
    print('--> pol_str:',pol_str)
    print('optionFee =',optionFee, '\noptionPDS =', optionPDS, '\noptionB =', optionB, '\noptionT =', optionT, '\noption_CB =', option_CB_name,'\n')

    #Options and parameters
    nat_economy   = 'national'
    global economy
    economy       = get_economic_unit(myCountry)
    event_level   = [economy, 'hazard', 'rp']                            #levels of index at which one event happens
    default_rp    = 'default_rp'                                         #return period to use when no rp is provided (mind that this works with protection)
    income_cats   = 'hhid'                                               #categories of households
    affected_cats = pd.Index(['a', 'na'], name='affected_cat')           #categories for social protection
    helped_cats   = pd.Index(['helped','not_helped'], name='helped_cat')
    
    is_local_welfare = False
    is_rev_dw = True

    share_public_assets = True
    if pol_str == 'noPT': share_public_assets = False

    #read data
    macro = pd.read_csv(intermediate+'macro.csv', index_col=economy)
    cat_info = pd.read_csv(intermediate+'cat_info.csv',  index_col=[economy, income_cats])

    #calc_delta_welfare(None,None,'','no',is_revised_dw=True,study=True)
    #assert(False)

    # First function: compute_with_hazard_ratios 
    # --> This is a shell that loads hazard_ratios, then calls process_input
    #  Inputs:
    #  - macro has province-level info
    #  - cat_info has household-level info
    #  - hazard_ratios has fa for each household (which varies not by hh, but by province, hazard, & RP) 
    macro_event, cats_event, hazard_ratios_event = compute_with_hazard_ratios(myCountry,pol_str,intermediate+'hazard_ratios.csv',macro,cat_info,economy,event_level,income_cats,default_rp,rm_overlap=False,verbose_replace=True)
    gc.collect()
    print('A')
    #verbose_replace=True by default, replace common columns in macro_event and cats_event with those in hazard_ratios_event
    
    # compute_dK does the following:
    # -- adds dk_event column to macro_event
    # -- adds affected/na categories to cats_event
    macro_event, cats_event_ia, pub_costs_inf = compute_dK(pol_str,macro_event,cats_event,event_level,affected_cats,myCountry,optionPDS,share_public_assets) 
    # ^ calculate the actual vulnerability, the potential damange to capital, income, and consumption
    print('B\n\n')
    
    pub_costs_pds = pub_costs_inf.copy()
    macro_event, cats_event_iah, pub_costs_pds = calculate_response(myCountry,pol_str,macro_event,cats_event_ia,pub_costs_pds,event_level,helped_cats,default_rp,
                                                                    option_CB,optionFee=optionFee,optionT=optionT, optionPDS=optionPDS, optionB=optionB,
                                                                    loss_measure='dk_private',fraction_inside=1, share_insured=.25)
    print('C\n\n')
    
    pub_costs_inf.to_csv(output+'pub_costs_inf_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)
    pub_costs_pds.to_csv(output+'pub_costs_pds_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)

    if False:
        is_contemporaneous = False 
        # For people outside affected province, do the collections for public asset reco & PDS happen at the same time?
        public_costs = calc_dw_outside_affected_province(macro_event, cat_info, pub_costs_inf, pub_costs_pds,event_level,is_contemporaneous,is_local_welfare,is_rev_dw)
        public_costs.to_csv(output+'public_costs_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)
        
    #optionFee: tax or insurance_premium  optionFee='insurance_premium',optionT='perfect', optionPDS='prop', 
    #optionB='unlimited',optionFee='tax',optionT='data', optionPDS='unif_poor', optionB='data',
    #optionT(targeting errors):perfect, prop_nonpoor_lms, data, x33, incl, excl.
    #optionB:one_per_affected, one_per_helped, one, unlimited, data, unif_poor, max01, max05
    #optionPDS: unif_poor, no, 'prop', 'prop_nonpoor'

    macro_event.to_csv(output+'macro_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)
    print('Step D: Wrote '+output+'macro_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv')

    #cats_event_iah.to_csv(output+'cats_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)
    print('Step E:  NOT writing out '+output+'cats_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv')

    #out = compute_dW(myCountry,pol_str,macro_event,cats_event_iah,event_level,option_CB,return_stats=True,return_iah=True,is_revised_dw=is_rev_dw)    
    out = calc_dw_inside_affected_province(myCountry,pol_str,optionPDS,macro_event,cats_event_iah,event_level,option_CB,return_stats=False,return_iah=True,is_revised_dw=is_rev_dw)
    print('F')

    # Flag: running local welfare
    print('running national welfare')
    results,iah = process_output(pol_str,out,macro_event,economy,default_rp,True,is_local_welfare,is_rev_dw)
    print('G')

    results.to_csv(output+'results_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)
    print('H')

    iah.head(10000).to_csv(output+'iah_full_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)
    iah = iah.drop([icol for icol in ['index','social','pcsoc','v','v_shew','gamma_SP','c_5','n','pcinc','shew','fa',
                                      'hhsize','hhsize_ae','hh_share','public_loss_v','v_shew',
                                      'dk_other','dk_private', 'dk_public',
                                      'di0_prv','di0_pub','dc0_prv','dc0_pub',
                                      'pc_fee_BE','scale_fac_soc',
                                      'c_min','macro_multiplier','help_fee',
                                      'SP_CPP','SP_FAP','SP_FNPF','SP_SPS','SP_PBS','SPP_core','SPP_add','nOlds','dc_0'] if icol in iah.columns],axis=1)
    iah.loc[iah.pcwgt!=0].to_csv(output+'iah_'+optionFee+'_'+optionPDS+'_'+option_CB_name+pol_str+'.csv',encoding='utf-8', header=True)
    print('\n******************\nStep I: wrote iah (excluding all hh with pcwgt = 0) ... still a huge file. See anything to drop?\n',iah.columns)
    #return True

    # result1=pd.read_csv('output-old/results.csv', index_col=economy)
    # iah1=pd.read_csv('output-old/iah.csv', index_col=event_level+['income_cat','affected_cat','helped_cat'])
    # print(((result1-results)/results).max())
    # print(((iah1-iah.reset_index().set_index(event_level+['income_cat','affected_cat','helped_cat']))/iah1).max())

if __name__ == '__main__':

    myCountry = 'SL'
    debug = False

    if len(sys.argv) > 1: myCountry = sys.argv[1]    
    if len(sys.argv) > 2 and (sys.argv[2] == 'true' or sys.argv[2] == 'True'): debug = True
    
    if myCountry == 'PH':
        pds_str = ['no','unif_poor']
        pol_str = ['']

        if debug: launch_compute_resilience_and_risk_thread(myCountry,'','no')
        else:
            for _pds in pds_str:
                launch_compute_resilience_and_risk_thread(myCountry,'',_pds)
        
    else: 

        if myCountry == 'FJ':
            pds_str = ['no']
            pol_str = ['',
                       '_noPT',
                       'fiji_SPS',     # Fijian social protection PLUS <-- Gets transferred to pds_str(optionPDS)!
                       'fiji_SPP']
                       # '_exp095',      # reduce exposure of poor by 5% (of total exposure!)
                       # '_exr095',      # reduce exposure of rich by 5% (of total exposure!)
                       # '_pcinc_p_110', # increase per capita income of poor people by 10%
                       # '_soc133',      # increase social transfers to poor by 33%
                       # '_rec067',      # decrease reconstruction time by 33%
                       # '_ew100',       # universal access to early warnings 
                       # '_vul070',      # decrease vulnerability of poor by 30%
                       # '_vul070r']     # decrease vulnerability of rich by 30%
        
            # Other policies:
            # --> develop market insurance for rich
            # --> universal access to finance
            # --> 

        if myCountry == 'SL':
            pds_str = ['no','unif_poor','unif_poor_only','unif_poor_q12','prop_q1','prop_q12']
            pol_str = ['']
            
        if True:# debug:
            print('Running in debug mode!')
            launch_compute_resilience_and_risk_thread(myCountry,'','no')
        else:
            with Pool() as pool: # this is for parallel
                print('LAUNCHING',len(list(product([myCountry],pol_str,pds_str))),'THREADS:\n',list(product([myCountry],pol_str,pds_str)))
                pool.starmap(launch_compute_resilience_and_risk_thread, list(product([myCountry],pol_str,pds_str)))
            
