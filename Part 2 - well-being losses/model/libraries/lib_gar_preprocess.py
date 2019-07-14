mport pandas as pd
from replace_with_warning import replace_with_warning
from pandas_helper import broadcast_simple
import numpy as np

from res_ind_lib import average_over_rp

###########################
# Load fa for wind or surge
def get_dk_over_k_from_file(path='inputs/GAR_data_surge.csv'):
    gardata_surge = pd.read_csv(path).dropna(axis=1, how='all')

    #these are part of France & the UK
    gardata_surge.Country.replace(['GUF', 'GLP', 'MTQ', 'MYT', 'REU'],'FRA', inplace=True)
    gardata_surge.Country.replace(['FLK', 'GIB', 'MSR'],'GBR', inplace=True)

    gardata_surge = gardata_surge.set_index(replace_with_warning(gardata_surge.Country, iso3_to_wb))[['AAL', 'Exposed Value (from GED)']]

    return gardata_surge['AAL'].sum(level=0)/gardata_surge['Exposed Value (from GED)'].sum(level=0)

#######################
# RP (as str) to floats
def str_to_float(s):
    try: return (float(s))
    except ValueError: return s

def gar_preprocessing():
    global iso3_to_wb
    iso3_to_wb = pd.read_csv('inputs/iso3_to_wb_name.csv', index_col='iso3', squeeze=True)

    #Names to WB names
    any_to_wb = pd.read_csv('inputs/any_name_to_wb_name.csv',index_col='any',squeeze=True)

    #######
    # AAL
    #
    #agg data
    gar_aal_data = pd.read_csv('inputs/GAR15 results feb 2016_AALmundo.csv', "latin-1", thousands=',', )

    # These are part of France and the UKxs
    gar_aal_data.ISO.replace(['GUF', 'GLP', 'MTQ', 'MYT', 'REU'],'FRA', inplace=True)
    gar_aal_data.ISO.replace(['FLK', 'GIB', 'MSR'],'GBR', inplace=True)

    # WB spellings
    gar_aal_data = gar_aal_data.set_index(replace_with_warning(gar_aal_data.ISO,iso3_to_wb)).drop(['ISO','Country'],axis=1)
    gar_aal_data.index.name='country'

    #aggregates UK and France pieces to one country
    gar_aal_data =  gar_aal_data.sum(level='country')

    #takes exposed value out
    gar_exposed_value = gar_aal_data.pop('EXPOSED VALUE')

    #rename hazards
    gar_aal_data = gar_aal_data.rename(columns=lambda s:(s.lower()))
    gar_aal_data= gar_aal_data.rename(columns = {'tropical cyclones':'cyclones', 'riverine floods':'flood', 'storm surge':'surge'})

    #gar_aal_data
    AAL = (gar_aal_data.T/gar_exposed_value).T

    # wind and surge
    aal_surge = get_dk_over_k_from_file('inputs/GAR_data_surge.csv')
    aal_wind = get_dk_over_k_from_file('inputs/GAR_data_wind.csv')
    aal_recomputed =aal_surge +aal_wind

    #print(((AAL.cyclones - aal_recomputed)/AAL.cyclones).abs().sort_values(ascending=False).head())
    c='Costa Rica'
    f = (aal_surge/aal_recomputed).mean()
    aal_surge.ix[c] = f*AAL.cyclones[c]
    aal_wind.ix[c] = (1-f)*AAL.cyclones[c]

    #AAL SPLIT
    AAL_splitted = AAL.drop('cyclones',axis=1).assign(wind= aal_wind, surge = aal_surge).rename(columns=dict(floods='flood')).stack()
    AAL_splitted.index.names=['country', 'hazard']

    # PMLs and Exposed value
    gardata =pd.read_csv('inputs/GAR15 results feb 2016_PML mundo.csv', encoding='latin-1', header=[0,1,2], index_col=0)#.sort_index(axis=1)
    gardata.index.name = 'country'

    gardata = gardata.reset_index()
    gardata.columns.names=['hazard','rp','what']

    # These are part of france / the UK
    gardata.country = gardata.country.replace(["French Guiana", "Guadeloupe", "Martinique", "Mayotte", "Reunion"],"France")
    gardata.country = gardata.country.replace(["Falkland Islands (Malvinas)", "Gibraltar", "Montserrat"],"United Kingdom")
    gardata.country = replace_with_warning(gardata.country, any_to_wb.dropna())
    gardata =gardata.set_index("country")

    #Other format for zero
    gardata = gardata.replace("---",0).astype(float)

    #aggregates france and uk
    gardata = gardata.sum(level="country")

    #Looks at the exposed value, for comparison
    exposed_value_GAR = gardata.pop("EXPOSED VALUE").squeeze()
    # df["exposed_value_GAR"].head()

    #rename hazards
    gardata = gardata.rename(columns=lambda s:(s.lower()))
    gardata= gardata.rename(columns = {"wind":"wind", "riverine floods":"flood", "storm surge":"surge"})

    gardata = gardata.rename(columns=str_to_float)

    gardata.head()

    #Asset losses per event
    dK_gar = gardata.swaplevel("what","hazard",axis=1)["million us$"].swaplevel("rp","hazard",axis=1).stack(level=["hazard","rp"], dropna=False).sort_index()

    #Exposed value per event
    ev_gar=  broadcast_simple(exposed_value_GAR,dK_gar.index)

    #Fraction of value destroyed
    frac_value_destroyed_gar = dK_gar/ev_gar

    frac_value_destroyed_gar.dropna().unstack("country").shape

    frac_value_destroyed_gar.to_csv("intermediate/frac_value_destroyed_gar.csv", encoding="utf-8", header=True)

    #Check. should be 0
    #print((exposed_value_GAR - gar_exposed_value).abs().sort_values(ascending=True).head(5))

    capital_losses_from_GAR_events = pd.read_csv("intermediate/capital_losses_from_GAR_events.csv", index_col=["country","hazard","rp"], squeeze=True)

    frac_cap_distroyed_from_events = capital_losses_from_GAR_events/ev_gar.median(level="country")
    frac_cap_distroyed_from_events.to_csv("intermediate/frac_cap_distroyed_from_events.csv", header=True)
    #print((average_over_rp(frac_cap_distroyed_from_events).squeeze()/ AAL_splitted ).replace(np.inf,np.nan).dropna().sort_values())

    # adds event to PML to complete AAL (old bad method)
    #print(frac_value_destroyed_gar.head())
    #print(AAL_splitted.head())

    paf= (AAL_splitted / average_over_rp(frac_value_destroyed_gar).squeeze()).replace(np.inf,np.nan).sort_values(ascending=False)

    #print(paf[paf>5])
    #print(paf.dropna().tail(30))
    #print(frac_value_destroyed_gar["Japan"]["flood"]*1e6)

    # add frequent events
    last_rp = 20
    new_rp = 1

    added_proba = 1/new_rp - 1/last_rp

    new_frac_destroyed = (AAL_splitted - average_over_rp(frac_value_destroyed_gar).squeeze())/(added_proba)

    #REMOVES 'tsunamis' and 'earthquakes' from this thing
    #new_frac_destroyed = pd.DataFrame(new_frac_destroyed).query("hazard not in ['tsunami', 'earthquake']").squeeze()
    #print(new_frac_destroyed.describe())

    hop = frac_value_destroyed_gar.unstack()
    hop[new_rp]=   new_frac_destroyed
    hop= hop.sort_index(axis=1)
    #print(hop.head())

    frac_value_destroyed_gar_completed = hop.stack()
    #print(frac_value_destroyed_gar_completed.head(50))
    # ^ this shows earthquake was not updated

    #double check. expecting zeroes expect for quakes and tsunamis:
    (average_over_rp(frac_value_destroyed_gar_completed).squeeze() - AAL_splitted).abs().sort_values(ascending=False).sample(10)

    # places where new values are higher than values for 20-yr RP
    test = frac_value_destroyed_gar_completed.unstack().replace(0,np.nan).dropna().assign(test=lambda x:x[new_rp]/x[20]).test

    max_relative_exp = .8

    overflow_frequent_countries = test[test>max_relative_exp].index
    print("overflow in {n} (country, event)".format(n=len(overflow_frequent_countries)))
    #print(test[overflow_frequent_countries].sort_values(ascending=False))

    # for this places, add infrequent events
    hop=frac_value_destroyed_gar_completed.unstack()

    hop[1]=hop[1].clip(upper=max_relative_exp*hop[20])
    frac_value_destroyed_gar_completed = hop.stack()
    #^ changed from: frac_value_destroyed_gar = hop.stack()
    #print(frac_value_destroyed_gar_completed)

    new_rp = 2000
    added_proba = 1/2000

    new_frac_destroyed = (AAL_splitted - average_over_rp(frac_value_destroyed_gar_completed).squeeze())/(added_proba)

    #REMOVES 'tsunamis' and 'earthquakes' from this thing
    # new_frac_destroyed = pd.DataFrame(new_frac_destroyed).query("hazard in ['tsunami', 'earthquake']").squeeze()

    hop = frac_value_destroyed_gar_completed.unstack()
    hop[new_rp]=   new_frac_destroyed
    hop= hop.sort_index(axis=1)

    frac_value_destroyed_gar_completed = hop.stack()
    #frac_value_destroyed_gar_completed.head(10)

    test = frac_value_destroyed_gar_completed.unstack().replace(0,np.nan).dropna().assign(test=lambda x:x[new_rp]/x[1500]).test
    #print(frac_value_destroyed_gar_completed["United States"])

    pd.DataFrame((average_over_rp(frac_value_destroyed_gar).squeeze()/AAL_splitted).replace(0,np.nan).dropna().sort_values())

    print('GAR preprocessing script: writing out intermediate/frac_value_destroyed_gar_completed.csv')
    frac_value_destroyed_gar_completed.to_csv("intermediate/frac_value_destroyed_gar_completed.csv", encoding="utf-8", header=True)
