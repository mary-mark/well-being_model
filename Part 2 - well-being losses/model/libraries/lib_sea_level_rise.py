import os
import pandas as pd

def get_SLR_hazard(myC,df):
    
    if myC != 'FJ': return df

    init_index = df.index.names
    # ^ record initial index of input df so that it can be returned in same format...    

    # Load additional files
    #tikina_all = pd.read_excel(os.getcwd()+'/../../country_docs/FJ/fji_pcode_list_v03.xls',sheetname='NEW TIKINA ALPHABETICAL',usecols=['TIKINA','PROVINCE','Admin 2 Pcode'])
    #tikina_all.columns = ['tikina','province','admin_2_pcode']

    tikina_no_coast = pd.read_excel(os.getcwd()+'/../../country_docs/FJ/tikina_no_coast.xlsx',usecols=['tikina_no_coast'])
    tikina_no_coast.columns = ['Tikina']
    tikina_no_coast['has_coast'] = 0

    slr_exposure = pd.read_csv(os.getcwd()+'/../inputs/FJ/fiji_results_exposure_v1.csv',usecols=['time','assets_below_1p0','assets_below_2p0'])
    slr_exposure['assets_below_1p0'] *= (1.E6/0.48)
    slr_exposure['assets_below_2p0'] *= (1.E6/0.48)
    # ^ exposure values for 2030/50/2100, all in millions of USD. PCRAFI/Exposure info in FJD

    # Manipulate df (tikina-level exposure info)
    df = df.reset_index().set_index(['Tikina','asset_class'])['Exp_Value'].mean(level=['Tikina','asset_class']).sum(level='Tikina')
    # ^ leaving out 'value_destroyed' here

    # merge files
    df = pd.merge(df.reset_index(),tikina_no_coast.reset_index(),on=['Tikina'],how='outer').reset_index().set_index('Tikina')
    df = df.drop(['level_0','index'],axis=1)
    df['has_coast'].fillna(1,inplace=True)

    assets_1m_2030 = slr_exposure.loc[slr_exposure.time==2030,'assets_below_1p0'].squeeze()
    assets_2m_2030 = slr_exposure.loc[slr_exposure.time==2030,['assets_below_1p0','assets_below_2p0']].sum(axis=1).squeeze()

    df['fa_slr1_2030'] = (assets_1m_2030*df[['Exp_Value','has_coast']].prod(axis=1)/df[['Exp_Value','has_coast']].prod(axis=1).sum()).fillna(0.)
    df['fa_slr2_2030'] = (assets_2m_2030*df[['Exp_Value','has_coast']].prod(axis=1)/df[['Exp_Value','has_coast']].prod(axis=1).sum()).fillna(0.)

    #svg_file_path = '../map_files/FJ/FIJI_Tikina_2.svg'
    #make_map_from_svg(
    #    tikina_all.has_coast, #data
    #    svg_file_path,                  #path to blank map
    #    outname='coasts',  #base name for output  (will create img/map_of_asset_risk.png, img/legend_of_asset_risk.png, etc.)
    #    color_maper=plt.cm.get_cmap('Blues'), #color scheme (from matplotlib. Chose them from http://colorbrewer2.org/)
    #    label='coasts',
    #    new_title='Map of coasts in Fiji',  #title for the colored SVG
    #    do_qualitative=False,
    #    res=2500)

    return df#.reset_index().set_index(init_index)
