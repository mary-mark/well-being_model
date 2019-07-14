import pandas as pd

def run_fijian_SPP(macro_event,cats_event_iah):

    sp_payout = pd.DataFrame(index=macro_event.sum(level='rp').index)
    sp_payout = sp_payout.reset_index()
    
    sp_payout['monthly_allow'] = 177#FJD
        
    sp_payout['frac_core'] = 0.0
    #sp_payout.loc[sp_payout.rp >= 20,'frac_core'] = 1.0
    sp_payout.loc[sp_payout.rp >= 10,'frac_core'] = 1.0        
    
    sp_payout['frac_add'] = 0.0
    #sp_payout.loc[(sp_payout.rp >=  5)&(sp_payout.rp < 10),'frac_add'] = 0.25
    sp_payout.loc[(sp_payout.rp >= 10)&(sp_payout.rp < 20),'frac_add'] = 0.50
    sp_payout.loc[(sp_payout.rp >= 20)&(sp_payout.rp < 40),'frac_add'] = 0.75
    sp_payout.loc[(sp_payout.rp >= 40),'frac_add'] = 1.00
    
    sp_payout['multiplier'] = 1.0
    sp_payout.loc[(sp_payout.rp >=  40)&(sp_payout.rp <  50),'multiplier'] = 2.0
    sp_payout.loc[(sp_payout.rp >=  50)&(sp_payout.rp < 100),'multiplier'] = 3.0
    sp_payout.loc[(sp_payout.rp >= 100),'multiplier'] = 4.0

    #sp_payout['payout_core'] = sp_payout[['monthly_allow','frac_core','multiplier']].prod(axis=1)
    #sp_payout['payout_add']  = sp_payout[['monthly_allow', 'frac_add','multiplier']].prod(axis=1)
    # ^ comment out these lines when uncommenting below
    sp_payout['payout'] = sp_payout[['monthly_allow','multiplier']].prod(axis=1)
    # ^ this line is for when we randomly choose people in each group

    sp_payout.to_csv('../output_country/FJ/SPP_details.csv')
 
    cats_event_iah = pd.merge(cats_event_iah.reset_index().copy(),sp_payout[['rp','payout','frac_core','frac_add']].reset_index(),on=['rp'])
    cats_event_iah = cats_event_iah.reset_index().set_index(['Division','hazard','hhid','rp','affected_cat','helped_cat'])

    cats_event_iah = cats_event_iah.drop([i for i in ['level_0'] if i in cats_event_iah.columns],axis=1)

    # Generate random numbers to determine payouts
    cats_event_iah['SP_lottery'] = np.random.uniform(0.0,1.0,cats_event_iah.shape[0])

    # Calculate payouts for core: 100% payout for RP >= 20
    cats_event_iah.loc[(cats_event_iah.SPP_core == True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'help_received'] = cats_event_iah.loc[(cats_event_iah.SPP_core == True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'payout']/cats_event_iah.loc[(cats_event_iah.SPP_core == True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'hhsize']
    
    # Calculate payouts for additional: variable payout based on lottery
    cats_event_iah.loc[(cats_event_iah.SPP_add==True)&(cats_event_iah.SP_lottery<cats_event_iah.frac_add),'SP_lottery_win'] = True
    cats_event_iah['SP_lottery_win'] = cats_event_iah['SP_lottery_win'].fillna(False)
    
    cats_event_iah.loc[(cats_event_iah.SP_lottery_win==True),'help_received'] = cats_event_iah.loc[(cats_event_iah.SP_lottery_win==True),'payout']/cats_event_iah.loc[(cats_event_iah.SP_lottery<cats_event_iah.frac_core),'hhsize']

    cats_event_iah = cats_event_iah.reset_index().set_index(['Division','hazard','rp','hhid']).sortlevel()
    # ^ Take helped_cat and affected_cat out of index. Need to slice on helped_cat, and the rest of the code doesn't want hhtypes in index
  
    cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped'),'help_received'] = 0
    cats_event_iah = cats_event_iah.drop([i for i in ['level_0','SPP_core','SPP_add','payout','frac_core','frac_add','SP_lottery','SP_lottery_win'] if i in cats_event_iah.columns],axis=1)

    my_out = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
    my_out.to_csv('../output_country/FJ/SPplus_expenditure.csv')
    my_out,_ = average_over_rp(my_out.sum(level=['rp']),default_rp)
    my_out.sum().to_csv('../output_country/FJ/SPplus_expenditure_annual.csv')

    return cats_event_iah

def run_fijian_SPS(nacro_event,cats_event_iah):

    sp_payout = macro_event['dk_event'].copy()
    sp_payout = sp_payout.sum(level=['hazard','rp'])
    sp_payout = sp_payout.reset_index().set_index(['hazard'])

    sp_200yr = sp_payout.loc[sp_payout.rp==200,'dk_event']
        
    sp_payout = pd.concat([sp_payout,sp_200yr],axis=1,join='inner')
    sp_payout.columns = ['rp','dk_event','benchmark_losses']

    sp_payout['f_benchmark'] = (sp_payout['dk_event']/sp_payout['benchmark_losses']).clip(lower=0.0,upper=1.0)
    sp_payout.loc[(sp_payout.rp < 10),'f_benchmark'] = 0.0
    sp_payout = sp_payout.drop(['dk_event','benchmark_losses'],axis=1)
    sp_payout = sp_payout.reset_index().set_index(['hazard','rp'])
    sp_payout.to_csv('../output_country/FJ/SPS_details.csv')

    cats_event_iah = pd.merge(cats_event_iah.reset_index().copy(),sp_payout.reset_index(),on=['hazard','rp'])
    cats_event_iah = cats_event_iah.reset_index().set_index(['Division','hazard','rp','hhid']).sortlevel()

    # paying out per cap
    cats_event_iah['help_received'] = 0

    print('\nTotal N Households:',cats_event_iah['hhwgt'].sum(level=['hazard','rp']).mean())
    print('\nSPS enrollment:',cats_event_iah.loc[(cats_event_iah.SP_SPS==True),['nOlds','hhwgt']].prod(axis=1).sum(level=['hazard','rp']).mean())
    print('\nCPP enrollment:',cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'hhwgt'].sum(level=['hazard','rp']).mean())
    print('\nFAP enrollment:',cats_event_iah.loc[(cats_event_iah.SP_FAP==True),'hhwgt'].sum(level=['hazard','rp']).mean())
    print('\nPBS enrollment:',cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'hhwgt'].sum(level=['hazard','rp']).mean())
    
    print('\nMax SPS expenditure =',(cats_event_iah.loc[(cats_event_iah.SP_SPS==True),['hhwgt','nOlds']].prod(axis=1).sum()*300+
                                     cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'hhwgt'].sum()*300+
                                     cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'hhwgt'].sum()*600)/(17*4),'\n')
        
    cats_event_iah.loc[(cats_event_iah.SP_SPS==True),'help_received']+=300*(cats_event_iah.loc[(cats_event_iah.SP_SPS==True),['hhwgt','nOlds','f_benchmark']].prod(axis=1)/
                                                                            cats_event_iah.loc[(cats_event_iah.SP_SPS==True),'pcwgt']).fillna(0)
    cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'help_received']+=300*(cats_event_iah.loc[(cats_event_iah.SP_CPP==True),['hhwgt','f_benchmark']].prod(axis=1)/
                                                                            cats_event_iah.loc[(cats_event_iah.SP_CPP==True),'pcwgt']).fillna(0)
    cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'help_received']+=600*(cats_event_iah.loc[(cats_event_iah.SP_PBS==True),['hhwgt','f_benchmark']].prod(axis=1)/
                                                                            cats_event_iah.loc[(cats_event_iah.SP_PBS==True),'pcwgt']).fillna(0)
    cats_event_iah.loc[(cats_event_iah.helped_cat=='not_helped'),'help_received'] = 0
    my_out = cats_event_iah[['help_received','pcwgt']].prod(axis=1).sum(level=['hazard','rp'])
    my_out.to_csv('../output_country/FJ/SPS_expenditure.csv')
    my_out,_ = average_over_rp(my_out.sum(level=['rp']),default_rp)
    my_out.sum().to_csv('../output_country/FJ/SPS_expenditure_annual.csv')

    cats_event_iah = cats_event_iah.drop(['f_benchmark'],axis=1)

    return cats_event_iah
