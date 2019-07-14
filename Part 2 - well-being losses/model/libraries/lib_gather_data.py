import pandas as pd
from libraries.pandas_helper import get_list_of_index_names, broadcast_simple, concat_categories
import numpy as np
from scipy.interpolate import UnivariateSpline,interp1d
from libraries.lib_average_over_rp import *

def mystriper(string):
    '''strip blanks and converts everythng to lower case''' 
    if type(string)==str:
        return str.strip(string).lower()
    else:
        return string

def get_hhid_FIES(df):
    df['hhid'] =  df['w_regn'].astype('str')
    df['hhid'] += df['w_prov'].astype('str')    
    df['hhid'] += df['w_mun'].astype('str')
    df['hhid'] += df['w_bgy'].astype('str')
    df['hhid'] += df['w_ea'].astype('str') 
    df['hhid'] += df['w_shsn'].astype('str')
    df['hhid'] += df['w_hcn'].astype('str')   

#weighted average		
def wavg(data,weights): 
    df_matched =pd.DataFrame({'data':data,'weights':weights}).dropna()
    return (df_matched.data*df_matched.weights).sum()/df_matched.weights.sum()

#gets share per agg category from the data in one of the sheets in PAGER_XL	
def get_share_from_sheet(PAGER_XL,pager_code_to_aggcat,iso3_to_wb,sheetname='Rural_Non_Res'):
    data = pd.read_excel(PAGER_XL,sheetname=sheetname).set_index('ISO-3digit') #data as provided in PAGER
    #rename column to aggregate category
    data_agg =    data[pager_code_to_aggcat.index].rename(columns = pager_code_to_aggcat) 
    #only pick up the columns that are the indices in paper_code_to_aggcat, and change each name to median, fragile etc. based on pager_code_to_aggcat 
    #group by category and sum
    data_agg= data_agg.sum(level=0,axis=1) #sum each category up and shows only three columns with fragile, median and robust.

    data_agg = data_agg.set_index(data_agg.reset_index()['ISO-3digit'].replace(iso3_to_wb));
    
    data_agg.index.name='country'
    return data_agg[data_agg.index.isin(iso3_to_wb)] #keeps only countries
	
def social_to_tx_and_gsp(economy,cat_info):       
        '''(tau_tax, gamma_SP) from cat_info[['social','c','weight']] '''
        
        # total per capital social tax collected/dispurced
        tau_tax = cat_info.loc[['social','c_pc','pcwgt']].prod(axis=1, skipna=False).sum() / cat_info.loc[:,['c_pc','pcwgt']].prod(axis=1, skipna=False).sum()
        #income from social protection PER PERSON as fraction of PER CAPITA social protection
        gsp= cat_info.loc[:,['social','c_pc']].prod(axis=1,skipna=False) / cat_info.loc[:,['social','c_pc','pcwgt']].prod(axis=1, skipna=False).sum()
        gsp.fillna(0,inplace=True)
        return tau_tax, gsp
		
		
def perc_with_spline(data, wt, percentiles):
	assert np.greater_equal(percentiles, 0.0).all(), 'Percentiles less than zero' 
	assert np.less_equal(percentiles, 1.0).all(), 'Percentiles greater than one' 
	data = np.asarray(data) 
	assert len(data.shape) == 1 
	if wt is None: 
		wt = np.ones(data.shape, np.float) 
	else: 
		wt = np.asarray(wt, np.float) 
		assert wt.shape == data.shape 
		assert np.greater_equal(wt, 0.0).all(), 'Not all weights are non-negative.' 
	assert len(wt.shape) == 1 
	i = np.argsort(data) 
	sd = np.take(data, i, axis=0)
	sw = np.take(wt, i, axis=0) 
	aw = np.add.accumulate(sw) 
	if not aw[-1] > 0: 
	 raise ValueError('Nonpositive weight sum' )
	w = (aw)/aw[-1] 
	# f = UnivariateSpline(w,sd,k=1)
	f = interp1d(np.append([0],w),np.append([0],sd))
	return f(percentiles)	 
	
def match_percentiles(hhdataframe,quintiles,col_label):
    hhdataframe.loc[hhdataframe['c']<=quintiles[0],col_label]=1

    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['c']<=quintiles[j])&(hhdataframe['c']>quintiles[j-1]),col_label]=j+1
        
    return hhdataframe
	
def match_quintiles_score(hhdataframe,quintiles):
    hhdataframe.loc[hhdataframe['score']<=quintiles[0],'quintile_score']=1
    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['score']<=quintiles[j])&(hhdataframe['score']>quintiles[j-1]),'quintile_score']=j+1
    return hhdataframe
	
	
def reshape_data(income):
	data = np.reshape(income.values,(len(income.values))) 
	return data

def get_AIR_data(fname,sname,keep_sec,keep_per):
    # AIR dataset province code to province name
    AIR_prov_lookup = pd.read_excel(fname,sheetname='Lookup_Tables',usecols=['province_code','province'],index_col='province_code')
    AIR_prov_lookup = AIR_prov_lookup['province'].to_dict()
    # NOTE: the coding in AIR differs from the latest PSA coding for Zamboanga Peninsula 
    # --> AIR: 80: 'Zamboanga del Norte', 81: 'Zamboanga del Sur', 82: 'Zamboanga Sibugay'
    # --> PSA: 

    AIR_prov_corrections = {'Tawi-Tawi':'Tawi-tawi',
                            #'Metropolitan Manila':'Manila',
                            #'Davao del Norte':'Davao',
                            #'Batanes':'Batanes_off',
                            'North Cotabato':'Cotabato'}
    
    # AIR dataset peril code to peril name
    AIR_peril_lookup_1 = pd.read_excel(fname,sheetname='Lookup_Tables',usecols=['perilsetcode','peril'],index_col='perilsetcode')
    AIR_peril_lookup_1 = AIR_peril_lookup_1['peril'].dropna().to_dict()
    #AIR_peril_lookup_2 = {'EQ':'EQ', 'HUSSPF':'TC', 'HU':'wind', 'SS':'surge', 'PF':'flood'}

    AIR_value_destroyed = pd.read_excel(fname,sheetname='Loss_Results',
                                        usecols=['perilsetcode','province','Perspective','Sector','AAL','EP1','EP10','EP25','EP30','EP50','EP100','EP200','EP250','EP500','EP1000']).squeeze()
    AIR_value_destroyed.columns=['hazard','province','Perspective','Sector','AAL',1,10,25,30,50,100,200,250,500,1000]

    # Change province code to province name
    #AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard','Perspective','Sector'])
    AIR_value_destroyed['province'].replace(AIR_prov_lookup,inplace=True)
    AIR_value_destroyed['province'].replace(AIR_prov_corrections,inplace=True) 
    #AIR_prov_corrections

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('province')
    AIR_value_destroyed = AIR_value_destroyed.drop('All Provinces')

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard','province','Perspective','Sector'])
    AIR_value_destroyed = AIR_value_destroyed.drop(['index'],axis=1, errors='ignore')

    AIR_aal = AIR_value_destroyed['AAL']

    # Stack return periods column
    AIR_value_destroyed = AIR_value_destroyed.drop('AAL',axis=1)
    AIR_value_destroyed.columns.name='rp'
    AIR_value_destroyed = AIR_value_destroyed.stack()

    # Name values
    #AIR_value_destroyed.name='v'

    # Choose only Sector = 0 (Private Assets) 
    # --> Alternative: 15 = All Assets (Private + Govt (16) + Emergency (17))
    sector_dict = {'Private':0, 'private':0,
                   'Public':16, 'public':16,
                   'Government':16, 'government':16,
                   'Emergency':17, 'emergency':17,
                   'All':15, 'all':15}
    
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['Sector'])
    AIR_value_destroyed = AIR_value_destroyed.drop([iSec for iSec in range(0,30) if iSec != sector_dict[keep_sec]])

    AIR_aal = AIR_aal.reset_index().set_index(['Sector'])
    AIR_aal = AIR_aal.drop([iSec for iSec in range(0,30) if iSec != sector_dict[keep_sec]])
    
    # Choose only Perspective = Occurrence ('Occ') OR Aggregate ('Agg')
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['Perspective'])
    AIR_value_destroyed = AIR_value_destroyed.drop([iPer for iPer in ['Occ', 'Agg'] if iPer != keep_per])

    AIR_aal = AIR_aal.reset_index().set_index(['Perspective'])
    AIR_aal = AIR_aal.drop([iPer for iPer in ['Occ', 'Agg'] if iPer != keep_per])
    
    # Map perilsetcode to perils to hazard
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['hazard'])
    AIR_value_destroyed = AIR_value_destroyed.drop(-1)

    AIR_aal = AIR_aal.reset_index().set_index(['hazard'])
    AIR_aal = AIR_aal.drop(-1)

    # Drop Sector and Perspective columns
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['province','hazard','rp'])
    AIR_value_destroyed = AIR_value_destroyed.drop(['Sector','Perspective'],axis=1, errors='ignore')

    AIR_aal = AIR_aal.reset_index().set_index(['province','hazard'])
    AIR_aal = AIR_aal.drop(['Sector','Perspective'],axis=1, errors='ignore')
    
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('province')
    AIR_value_destroyed['hazard'].replace(AIR_peril_lookup_1,inplace=True)

    AIR_aal = AIR_aal.reset_index().set_index('province')
    AIR_aal['hazard'].replace(AIR_peril_lookup_1,inplace=True)

    # Keep earthquake (EQ), wind (HU), storm surge (SS), and precipitation flood (PF)
    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index('hazard')   
    AIR_value_destroyed = AIR_value_destroyed.drop(['HUSSPF'],axis=0)

    AIR_aal = AIR_aal.reset_index().set_index('hazard')   
    AIR_aal = AIR_aal.drop(['HUSSPF'],axis=0)

    AIR_value_destroyed = AIR_value_destroyed.reset_index().set_index(['province','hazard','rp'])
    AIR_value_destroyed = AIR_value_destroyed.sort_index().squeeze()

    AIR_aal = AIR_aal.reset_index().set_index(['province','hazard'])
    AIR_aal = AIR_aal.sort_index().squeeze()

    AIR_value_destroyed = AIR_extreme_events(AIR_value_destroyed,AIR_aal,sec=keep_sec,per=keep_per)

    return AIR_value_destroyed

def AIR_extreme_events(df_air,df_aal,sec='',per=''):

    # add frequent events
    last_rp = 20
    new_rp = 1

    added_proba = 1/new_rp - 1/last_rp

    # places where new values are higher than values for 10-yr RP
    test = df_air.unstack().replace(0,np.nan).dropna().assign(test=lambda x:x[new_rp]/x[10]).test

    max_relative_exp = .8

    overflow_frequent_countries = test[test>max_relative_exp].index
    print("overflow in {n} (region, event)".format(n=len(overflow_frequent_countries)))
    #print(test[overflow_frequent_countries].sort_values(ascending=False))

    # for this places, add infrequent events
    hop=df_air.unstack()


    hop[1]=hop[1].clip(upper=max_relative_exp*hop[10])
    df_air = hop.stack()
    #^ changed from: frac_value_destroyed_gar = hop.stack()
    #print(frac_value_destroyed_gar_completed)

    print(df_air.head(10))

    new_rp = 2000
    added_proba = 1/2000

    df_air_avg, _ = average_over_rp(df_air)

    new_frac_destroyed = (df_aal - df_air_avg.squeeze())/(added_proba)

    #REMOVES 'tsunamis' and 'earthquakes' from this thing
    # new_frac_destroyed = pd.DataFrame(new_frac_destroyed).query("hazard in ['tsunami', 'earthquake']").squeeze()

    hop = df_air.unstack()
    hop[new_rp]=   new_frac_destroyed
    hop= hop.sort_index(axis=1)

    df_air = hop.stack()
    #frac_value_destroyed_gar_completed.head(10)

    test = df_air.unstack().replace(0,np.nan).dropna().assign(test=lambda x:x[new_rp]/x[1000]).test
    #print(frac_value_destroyed_gar_completed["United States"])

    df_air_averages, _ = average_over_rp(df_air)

    pd.DataFrame((df_air_averages.squeeze()/df_aal).replace(0,np.nan).dropna().sort_values())

    print('GAR preprocessing script: writing out intermediate/frac_value_destroyed_gar_completed.csv')
    df_air.to_csv('../inputs/PH/Risk_Profile_Master_With_Population_with_EP1_and_EP2000'+sec+'_'+per+'.csv', encoding="utf-8", header=True)

    return df_air
    
def get_hh_savings(df, myC, econ_unit, pol, fstr):

    if myC == 'PH' and econ_unit == 'region': econ_unit = 'province'
    if myC =='BA': 
        _s = pd.DataFrame(index=df.index)
        _s['savings_per_hh'] = 0
    else:
        _s = pd.DataFrame({'c_pc':df.c_pc,'pcwgt':df.pcwgt,econ_unit:df[econ_unit],'ispoor':df.ispoor},index=df.index)
        
    if pol == '_nosavings': return 0
    elif pol == '_nosavingsdata': return df.eval('c/12')
        
    elif myC == 'SL': _s['hh_savings'] = _s['c']/12.

    elif myC == 'PH':

        # Load PSA file with average savings
        f = pd.read_excel(fstr,sheetname='Average Savings',skiprows=3).rename(columns={'Unnamed: 0':'province','Estimate':'p','Estimate.1':'np'})[['province','p','np']]

        # Load dictionaries so that these province names match those in df
        ph_prov_lookup = pd.read_excel('../inputs/PH/FIES_provinces.xlsx',usecols=['province_upper','province_AIR'],index_col='province_upper')['province_AIR'].to_dict()
        AIR_prov_corrections = {'Tawi-Tawi':'Tawi-tawi',
                                'North Cotabato':'Cotabato',
                                'COTABATO':'Cotabato',
                                'DAVAO':'Davao',
                                'COTABATO CITY':'Cotabato',
                                'ISABELA CITY':'Isabela',
                                'MANILA':'Manila',
                                'NCR-2ND DIST.':'NCR-2nd Dist.',
                                'NCR-3RD DIST.':'NCR-3rd Dist.',
                                'NCR-4TH DIST.':'NCR-4th Dist.',
                                'SAMAR (WESTERN)':'Samar (Western)'
                                }
        f['province'].replace(ph_prov_lookup,inplace=True)
        f['province'].replace(AIR_prov_corrections,inplace=True)

        # Manipulate for ease of merge
        f = f.reset_index().set_index('province').drop('index',axis=1)
        f.columns.name = 'pnp'
        f = f.stack().to_frame()
        f.columns = ['avg_savings']
        
        f = f.reset_index().set_index(['province'])
        
        f['ispoor'] = 0
        f.loc[f.pnp=='p','ispoor'] = 1
        f = f.drop('pnp',axis=1)

        f = f.reset_index().set_index(['province','ispoor']).dropna()

        # Poor in some provinces report negative savings...
        f['avg_savings'] = f['avg_savings'].mean(level=['province','ispoor']).clip(lower=0.)

        f = f.reset_index().set_index('province')
        _s = _s.reset_index().set_index('province')
        
        f['c_mean'] = 0
        f.loc[f.ispoor==0,'c_mean'] = _s.loc[_s.ispoor==0,['c','pcwgt']].prod(axis=1).sum(level='province')/_s.loc[_s.ispoor==0,'pcwgt'].sum(level='province')
        f.loc[f.ispoor==1,'c_mean'] = _s.loc[_s.ispoor==1,['c','pcwgt']].prod(axis=1).sum(level='province')/_s.loc[_s.ispoor==1,'pcwgt'].sum(level='province')

        try: f.to_csv('tmp/provincial_savings_avg.csv')
        except: pass
        assert(f['c_mean'].shape[0] == f['c_mean'].dropna().shape[0])

        # Put it back together
        _s = pd.merge(_s.reset_index(),f.reset_index(),on=['province','ispoor']).set_index('index').sort_index()
        _s = _s.mean(level='index')
    
        _s['hh_savings'] = _s.eval('avg_savings*c/c_mean')
        
    elif myC == 'BA':
        _s['hh_savings'] = df['savings_per_hh']*df['hhwgt']/df['pcwgt']
    else:
        # Without data: we tried giving hh savings = 6 months' income if they report spending on savings or investments, 1 month if not
        _s = (temp[['axfin','c']].prod(axis=1)/2.).clip(lower=temp['c']/12.)
        
    
    return _s['hh_savings']


def get_subnational_gdp_macro(myCountry,_hr,avg_prod_k):
    hr_init = _hr.shape[0]

    if myCountry == 'PH':

        grdp = pd.read_csv('../inputs/PH/phil_grdp.csv',usecols=['region','2015'])
        grdp.columns = ['_region','grdp']
        grdp['region_lower'] = grdp['_region'].str.lower()

        grdp['grdp_assets'] = grdp['grdp'].str.replace(',','').astype('int')*1000./avg_prod_k

        _hr = _hr.reset_index()
        _hr['region_lower'] = _hr['region'].str.lower()
        _hr = pd.merge(_hr.reset_index(),grdp.reset_index(),on=['region_lower']).reset_index().set_index(['region','hazard','rp']).sort_index()

        print(_hr.shape[0],hr_init)
        assert(_hr.shape[0] == hr_init)
        return _hr['grdp_assets']
