import os
import pandas as pd

#Import package for data analysis
from replace_with_warning import *
from lib_gather_data import *
from maps_lib import *

#ploting capacities
import matplotlib as mpl
import matplotlib.pyplot as plt 


#Default options for plots: 
#this controls the font used in the legend
font = {'family' : 'sans serif',
    'size'   : 26}
plt.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 20

for myCountry in [['FJ','Fiji','Fijians'],
                  ['PH','the Philippines','Filipinos']]:
    
    model  = os.getcwd() #get current directory
    output = model+'/../output_country/'+myCountry[0]+'/'

    if myCountry[0] == 'PH': economy = 'province'
    elif myCountry[0] == 'FJ': economy = 'Division'

    event_level = [economy, 'hazard', 'rp']
    df = pd.read_csv(output+'results_tax_fiji_SPS_.csv', index_col=[economy,'hazard','rp'])

    # Sum with RPs
    df_prov_mh = sum_with_rp(myCountry[0],df[['risk','risk_to_assets']],['risk','risk_to_assets'],sum_provinces=False)
    df_prov_mh['resilience'] = df['resilience'].mean(level=economy)

    #df_prov_mh['risk_to_assets'] = df_prov_mh['risk_to_assets'].clip(upper=0.06)
    df_prov_mh['risk'] = df_prov_mh['risk']

    # path to the blank map 
    svg_file_path = '../map_files/'+myCountry[0]+'/BlankSimpleMap.svg'
    inp_res = 800

    make_map_from_svg(
        df_prov_mh.risk_to_assets, #data 
        svg_file_path,                  #path to blank map
        outname=myCountry[0]+'_asset_risk_',  #base name for output  (will create img/map_of_asset_risk.png, img/legend_of_asset_risk.png, etc.)
        color_maper=plt.cm.get_cmap('Blues'), #color scheme (from matplotlib. Chose them from http://colorbrewer2.org/)
        label='Annual asset losses (% of GDP)',
        new_title='Map of asset risk in '+myCountry[1],  #title for the colored SVG
        do_qualitative=False,
        res=inp_res)
    
    make_map_from_svg(
        df_prov_mh.resilience, 
        svg_file_path,
        outname=myCountry[0]+'_se_resilience_',
        color_maper=plt.cm.get_cmap('RdYlGn'), 
        label='Socio-economic capacity (%)',
        new_title='Map of socio-economic resilience in '+myCountry[1],
        do_qualitative=False,
        res=inp_res)

    make_map_from_svg(
        df_prov_mh.risk,
        svg_file_path,
        outname=myCountry[0]+'_welfare_risk_',
        color_maper=plt.cm.get_cmap('Purples'), 
        label='Annual well-being losses [% of regional GDP]',
        new_title='Map of welfare risk in '+myCountry[1],
        do_qualitative=False,
        res=inp_res)
