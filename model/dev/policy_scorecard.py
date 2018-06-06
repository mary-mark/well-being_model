import os
import pandas as pd

myCountry = 'FJ'
economy = 'Division'
model  = os.getcwd() #get current directory
output = model+'/../output_country/'+myCountry+'/'

allDis = ['TC','EQTS']
allRPs = [1,10,22,50,72,100,224,475,975,2475]

upper_clip = 20000
drm_pov_sign = -1

my_out_file = open('/Users/brian/Desktop/BANK/hh_resilience_model/output_country/FJ/policy_scorecard.csv', 'w')
my_out_file.write('policy,asset losses (fjd),asset losses as pct of GDP,welf losses (fjd),welf losses as pct of GDP\n')

for pol_str in ['',
                '_exp095',      # reduce exposure of poor by 5% (of total exposure!)
                '_exr095',      # reduce exposure of rich by 5% (of total exposure!)
                '_pcinc_p_110', # increase per capita income of poor people by 10%
                '_soc133',      # increase social transfers to poor by 33%
                '_rec067',      # decrease reconstruction time by 33%
                '_ew100',       # universal access to early warnings 
                '_vul070']:      # decrease vulnerability of poor by 30%

    df = pd.read_csv(output+'results_tax_no_'+pol_str+'.csv', index_col=[economy,'hazard','rp'])
    iah = pd.read_csv(output+'iah_tax_no_'+pol_str+'.csv', index_col=[economy,'hhid'])
    macro = pd.read_csv(output+'macro_tax_no_'+pol_str+'.csv')
    npv_sf = macro['avg_prod_k'].mean()+1/macro['T_rebuild_K'].mean()

    df_prov = df[['dKtot','dWtot_currency']]
    df_prov['gdp'] = df[['pop','gdp_pc_pp_prov']].prod(axis=1)

    df_prov['R_asst'] = round(100.*df_prov['dKtot']/df_prov['gdp'],2)
    df_prov['R_welf'] = round(100.*df_prov['dWtot_currency']/df_prov['gdp'],2)
    df_prov = df_prov.sum(level='Division')
    df_prov['gdp'] = df[['pop','gdp_pc_pp_prov']].prod(axis=1).mean(level='Division')
    df_prov.to_csv('~/Desktop/my_file.csv')
    
    #print(df_prov)
    #print(df_prov[['dKtot','dWtot_currency','gdp']].sum())
    print('--> pol_str = ',pol_str)
    print('R_asset:',df_prov['dKtot'].sum(),'(',100.*df_prov['dKtot'].sum()/df_prov['gdp'].sum(),'%)')
    print('R_welf:',df_prov['dWtot_currency'].sum(),'(',100.*df_prov['dWtot_currency'].sum()/df_prov['gdp'].sum(),'%)')
    
    my_out_file.write((pol_str+','+str(df_prov['dKtot'].sum())+','+str(100.*df_prov['dKtot'].sum()/df_prov['gdp'].sum())+','
                       +str(df_prov['dWtot_currency'].sum())+','+str(100.*df_prov['dWtot_currency'].sum()/df_prov['gdp'].sum())+'\n'))
    
    continue
    for myDis in allDis:
        for myRP in allRPs:
            
            macro = macro.loc[(macro.hazard == myDis)&(macro.rp == myRP)]

            cut_rps = iah.loc[(iah.hazard == myDis)&(iah.rp == myRP)].fillna(0)

            cut_rps['c_initial'] = 0
            cut_rps.loc[cut_rps.pcwgt_ae != 0.,'c_initial'] = cut_rps.loc[cut_rps.pcwgt_ae != 0.,['c','hhsize']].prod(axis=1)/cut_rps.loc[(cut_rps.pcwgt_ae != 0.), 'hhsize_ae']

            cut_rps['pov_line'] *= cut_rps['c_initial']/cut_rps['pcinc_ae']

            cut_rps['delta_c'] = 0
            cut_rps.loc[cut_rps.pcwgt_ae != 0.,'delta_c'] = npv_sf*cut_rps.loc[(cut_rps.pcwgt_ae!=0.),['dk','pcwgt']].prod(axis=1)/cut_rps.loc[(cut_rps.pcwgt_ae != 0.),'pcwgt_ae']
    
            cut_rps['c_final']   = (cut_rps['c_initial'] + drm_pov_sign*cut_rps['delta_c']).clip(upper=upper_clip)
            
            print(myDis,myRP,(cut_rps.loc[(cut_rps.c_initial <= cut_rps.pov_line),['hhwgt','hhsize']].prod(axis=1).sum()),' + ',
                  cut_rps.loc[(cut_rps.c_final<= cut_rps.pov_line)&(cut_rps.c_initial > cut_rps.pov_line),['hhwgt','hhsize']].prod(axis=1).sum())

            

    print('\n')

my_out_file.close()
                               
                      
    #iah['nPov'] = 0
    #iah.loc[(),'nPov'] = 
