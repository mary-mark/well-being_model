import os
import pandas as pd
from lib_gather_data import *
from lib_country_dir import *
import matplotlib.pyplot as plt
#import moviepy.editor as mpy 

from maps_lib import *

import seaborn as sns
sns.set_style('darkgrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)
greys_pal = sns.color_palette('Greys', n_colors=9)
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

global model
model  = os.getcwd()
inputs = model+'/../inputs/FJ/' # get inputs data directory

pi = 0.3333
eta = 1.5
my_cap, my_inc, my_welf = np.linspace(0.0001,100000.1,100), [], []
for k in my_cap:
    k = float(k)
    i = k*pi
    my_inc.append(i)
    my_welf.append(((i**(1-eta))/(1-eta)))

ax = plt.gca()
fig = ax.get_figure()

print(my_cap)
print(my_inc)

plt.scatter(my_cap,my_inc,'k-', label='Household income',color=q_colors[0],zorder=100,alpha=0.85)
plt.scatter(my_cap,my_welf,'k-',label='Household well-being',color=q_colors[2],zorder=100,alpha=0.85)

plt.xlabel('Household assets [USD PPP]')
plt.ylabel('Household assets')
#leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
#leg.get_frame().set_color('white')
#leg.get_frame().set_edgecolor(greys_pal[7])
#leg.get_frame().set_linewidth(0.2)
fig.savefig('/Users/brian/Desktop/k_i_w.pdf',format='pdf')
plt.clf()
plt.close('all')
assert(False)

# Poverty definitions
urban_pov_line = 55.12*52
rural_pov_line = 49.50*52

# Agriculture (Fishing & Farming) & Tourism
df = pd.read_excel(inputs+'HIES 2013-14 Income Data.xlsx',
                   usecols=['HHID','Division','AE','HHsize','Ethnicity','Weight','Sector','TotalAgri',
                            'Fishprawnscrabsetc','NWages','TotalBusiness','TotalIncome']).set_index('HHID')

df_dem = pd.read_excel(inputs+'HIES 2013-14 Demographic Data.xlsx',sheetname='Income Earners',
                       usecols=['Hhid','weight','Occupation','Comp Name Purpose','Person No','Gender']).set_index('Hhid')
df_dem = df_dem.rename(columns={'Person No':'Person_No'}) 

df_cons = pd.read_excel(inputs+'HIES 2013-14 Consumption Data.xlsx',sheetname='by_Division',
                        usecols=['HHID','DIVISION 1: FOOD & NON-ALCOHOLIC BEVERAGES + Imputed Food']).set_index('HHID')

df_cons = df_cons.rename(columns={'DIVISION 1: FOOD & NON-ALCOHOLIC BEVERAGES + Imputed Food':'food_cons'}) 
df = pd.concat([df,df_cons,df_dem.loc[df_dem.Person_No == '01']],axis=1,join='inner')
df['frac_inc_foodcons'] = 100.*df['food_cons']/df['TotalIncome']


#df.loc[df.frac_inc_foodcons >= 100].to_csv('~/Desktop/too_much.csv')
#print(df.loc[df.frac_inc_foodcons >= 1])

df_tourism = df_dem.loc[((df_dem['Occupation'].str.contains('TOUR'))
                         |(df_dem['Occupation'].str.contains('HOTEL'))
                         |(df_dem['Occupation'].str.contains('RESORT'))
                         |(df_dem['Occupation'].str.contains('SPA'))
                         |(df_dem['Occupation'].str.contains('RENTAL'))
                         |(df_dem['Occupation'].str.contains('TAXI'))
                         |(df_dem['Occupation'].str.contains('WAITER'))
                         |(df_dem['Occupation'].str.contains('WAITRESS'))
                         |(df_dem['Comp Name Purpose'].str.contains('TOUR'))
                         |(df_dem['Comp Name Purpose'].str.contains('RESORT'))
                         |(df_dem['Comp Name Purpose'].str.contains('HOTEL'))
                         |(df_dem['Comp Name Purpose'].str.contains('SPA'))
                         |(df_dem['Comp Name Purpose'].str.contains('RENTAL'))
                         |(df_dem['Comp Name Purpose'].str.contains('GIFT SHOP'))
                         |(df_dem['Comp Name Purpose'].str.contains('TAXI')))].sum(level='Hhid')
df_tourism['tourism_inc_frac'] = df_tourism['weight']/df_dem['weight'].sum(level='Hhid')
df_tourism['tourism_income'] = 1
df_tourism = pd.concat([df_tourism,df[['TotalIncome','AE','HHsize','Weight']]], axis=1, join='inner')

print(int(df_tourism['weight'].sum()),'tourism jobs in Fiji.')
print(int(df_tourism[['Weight','HHsize']].prod(axis=1).sum()),'people in households that gain income from tourism.')

df['inc_pAE'] = df['TotalIncome']/df['AE']
df['ispoor'] = False
df.loc[(df.Sector=='Urban')&(df.inc_pAE<urban_pov_line),'ispoor'] = True
df.loc[(df.Sector=='Rural')&(df.inc_pAE<rural_pov_line),'ispoor'] = True
#df['TotalAgri'] -= df['Fishprawnscrabsetc']
df['frac_inc_agri'] = df['TotalAgri']/df['TotalIncome']
df['frac_inc_fish'] = df['Fishprawnscrabsetc']/df['TotalIncome']

###################
# Food consumption
frac_food_poor = round(df.loc[(df.ispoor==True),['frac_inc_foodcons','Weight']].prod(axis=1).sum()/df.loc[(df.ispoor==True),'Weight'].sum(),1)
frac_food_rich = round(df.loc[(df.ispoor==False),['frac_inc_foodcons','Weight']].prod(axis=1).sum()/df.loc[(df.ispoor==False),'Weight'].sum(),1)

print('\n\nPoor Fijian hh spend '+str(frac_food_poor)+'% of income on food (avg).')
print('Non-por Fijian hh spend '+str(frac_food_rich)+'% of income on food (avg).\n\n')

ax = df.loc[(df.ispoor==True)&(df.frac_inc_foodcons<100)&(df.TotalIncome<50000)].plot.scatter('inc_pAE','frac_inc_foodcons',color=q_colors[1],edgecolor=q_colors[0],linewidth=1,
                                                                                       label='Households in poverty',alpha=0.15)
df.loc[(df.ispoor==False)&(df.frac_inc_foodcons<100)&(df.TotalIncome<50000)].plot.scatter('inc_pAE','frac_inc_foodcons',color=q_colors[1],edgecolor=q_colors[2],
                                                                                          label='Households above poverty',alpha=0.15,ax=ax)
fig = ax.get_figure()

plt.plot([0,3000],[frac_food_poor,frac_food_poor],'k-',lw=1.5,color=greys_pal[7],zorder=100,alpha=0.85)
ax.annotate('On average, Fijian households in poverty\n  spend '+str(frac_food_poor)+'% of their incomes on food.',
            xy=(3200,frac_food_poor),xycoords='data',ha='left',va='center',fontsize=7,annotation_clip=False,weight='bold')
plt.plot([0,6000],[frac_food_rich,frac_food_rich],'k-',lw=1.5,color=greys_pal[7],zorder=100,alpha=0.85)
ax.annotate('On average, non-poor Fijian households\n  spend '+str(frac_food_rich)+'% of their incomes on food.',
            xy=(6200,frac_food_rich),xycoords='data',ha='left',va='center',fontsize=7,annotation_clip=False,weight='bold')

plt.xlim(0,12000)
plt.ylim(0,100)
plt.xlabel(r'Household income per adult equivalent [FJD yr$^{-1}$]')
plt.ylabel('Percentage of income spent on food')
leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
leg.get_frame().set_color('white')
leg.get_frame().set_edgecolor(greys_pal[7])
leg.get_frame().set_linewidth(0.2)
fig.savefig('../output_plots/FJ/sectoral/food_cons_vs_income.pdf',format='pdf')
plt.clf()
plt.close('all')

# Look at increased food costs
my_xlim = None
my_ylim = None
movie_array = []

df['pov_line'] = 0
df.loc[df.Sector=='Urban','pov_line'] = urban_pov_line
df.loc[df.Sector=='Rural','pov_line'] = rural_pov_line

for nfood, dfood in enumerate(np.linspace(0.00,1.00,21)):

    dfood_str = str(5*int(nfood))
    ax = plt.gca()

    df['inc_less_food'] = (df['inc_pAE']-(dfood*df['food_cons'])/df['AE']).clip(lower=0,upper=20000)

    ci_heights, ci_bins = np.histogram(df['inc_pAE'].clip(lower=0,upper=20000), bins=50, range=(0,20000), weights=df[['Weight','AE']].prod(axis=1))
    cf_heights, cf_bins = np.histogram(df['inc_less_food'], bins=50, range=(0,20000), weights=df[['Weight','AE']].prod(axis=1))
    #ci_heights, ci_bins = np.histogram((df['inc_pAE']-df['food_cons']/df['AE']).clip(lower=0,upper=20000), bins=50, weights=df[['Weight','AE']].prod(axis=1))
    #cf_heights, ci_bins = np.histogram((df['inc_pAE']-(1.+dfood)*df['food_cons']/df['AE']).clip(lower=0,upper=20000), bins=50, weights=df[['Weight','AE']].prod(axis=1))
    
    n_fp = int(df.loc[(df.inc_pAE > df.pov_line)&(df.inc_less_food <= df.pov_line),['Weight','AE']].prod(axis=1).sum())
    if n_fp > 1000:
        n_food_pov = str(n_fp)[:-3]+','+str(n_fp)[-3:]
    else: n_food_pov = str(n_fp)

    ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[1],label='Initial household consumption',alpha=0.4)
    ax.bar(ci_bins[:-1], cf_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],label='Household consumption\nless food price increase',alpha=0.4)
    fig = ax.get_figure()
    if my_xlim == None: my_xlim = [xlim for xlim in ax.get_xlim()]
    if my_ylim == None:  
        my_ylim = [ylim for ylim in ax.get_ylim()]
        my_ylim[1] *= 1.25
    plt.xlim(my_xlim[0],my_xlim[1])
    plt.ylim(my_ylim[0],my_ylim[1])
    leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
    leg.get_frame().set_color('white')
    leg.get_frame().set_edgecolor(greys_pal[7])
    leg.get_frame().set_linewidth(0.2)
    ax.annotate('A '+dfood_str+'% increase in food prices would\n\nmove '+n_food_pov+' Fijians into poverty.',
                xy=(6200,50000),xycoords='data',ha='left',va='center',fontsize=12,annotation_clip=False,color=greys_pal[7],weight='bold')

    plt.plot([rural_pov_line,rural_pov_line],[my_ylim[0],my_ylim[1]],'k-',lw=1.5,color=greys_pal[7],zorder=100,alpha=0.85)
    fig.savefig('../output_plots/FJ/sectoral/food_inc'+dfood_str+'_vs_income.png',format='png')
    plt.clf()
    plt.close('all')

    #movie_array.append('../output_plots/FJ/sectoral/food_inc'+dfood_str+'_vs_income.png')

assert(False)

##########
# Tourism
df['tourism_income'] = 0
df.ix[df_tourism.index,'tourism_income'] = df_tourism['tourism_income']

p_inc_tour_poor = round(100.*df.loc[(df.ispoor==True)&(df.tourism_income>0),['Weight','HHsize']].prod(axis=1).sum()/df.loc[df.ispoor==True,['Weight','HHsize']].prod(axis=1).sum(),1)
p_inc_tour_rich = round(100.*df.loc[(df.ispoor==False)&(df.tourism_income>0),['Weight','HHsize']].prod(axis=1).sum()/df.loc[df.ispoor==False,['Weight','HHsize']].prod(axis=1).sum(),1)

print('Odds of a Fijian in poverty having income from tourism:',p_inc_tour_poor,'%')
print('Odds of a Fijian above poverty having income from tourism:',p_inc_tour_rich,'%')

################################
# Look at poverty effect of reducing income from agriculture
df['new_inc_pAE'] = df['inc_pAE']*(1-0.1*df['frac_inc_agri'])#-0.1*df['frac_inc_fish'])
agloss = np.linspace(0.01,1.,100)
initial_nPov = df.loc[df.ispoor == True,['Weight','HHsize']].prod(axis=1).sum()
print(initial_nPov)
new_nPov = []
for agl in agloss:
    new_nPov.append(df.loc[((df.Sector == 'Urban')&((df.TotalIncome-agl*df.TotalAgri)/df.AE <= urban_pov_line)
                            |(df.Sector == 'Rural')&((df.TotalIncome-agl*df.TotalAgri)/df.AE <= rural_pov_line)),['Weight','HHsize']].prod(axis=1).sum() - initial_nPov)
ax = plt.scatter(agloss*100,new_nPov,alpha=0.6)
fig = ax.get_figure()
plt.xlim(0,102)
plt.ylim(0,105000)
plt.xlabel('Agricultural income reduction [%]')
plt.ylabel('Fijians entering poverty [capita]')
fig.savefig('../output_plots/FJ/sectoral/ag_inc_poverty_incidence.pdf',format='pdf')#+'.pdf',format='pdf')
plt.clf()

# Calculate numbers
total_hh_fishermen = df.loc[df.Fishprawnscrabsetc > 0,'Weight'].sum()
frac_hh_fishermen_ita = df.loc[(df.Fishprawnscrabsetc > 0)&(df.Ethnicity=='ITAUKEI'),'Weight'].sum()/total_hh_fishermen
frac_hh_fishermen_ind = df.loc[(df.Fishprawnscrabsetc > 0)&(df.Ethnicity=='INDIAN-FIJIAN'),'Weight'].sum()/total_hh_fishermen

total_hh_farmers = df.loc[df.TotalAgri > 0,'Weight'].sum()
frac_hh_farmers_ita = df.loc[(df.TotalAgri > 0)&(df.Ethnicity=='ITAUKEI'),'Weight'].sum()/total_hh_farmers
frac_hh_farmers_ind = df.loc[(df.TotalAgri > 0)&(df.Ethnicity=='INDIAN-FIJIAN'),'Weight'].sum()/total_hh_farmers

mean_finc_ag_poor = round(100.*df.loc[df.ispoor==True,['frac_inc_agri','Weight','HHsize']].prod(axis=1).sum()/df.loc[df.ispoor==True,['Weight','HHsize']].prod(axis=1).sum(),1)
mean_finc_ag_rich = round(100.*df.loc[df.ispoor==False,['frac_inc_agri','Weight','HHsize']].prod(axis=1).sum()/df.loc[df.ispoor==False,['Weight','HHsize']].prod(axis=1).sum(),1)

print('On average, '+str(mean_finc_ag_poor)+'% of poor peoples\' incomes come from agriculture')
print('On average, '+str(mean_finc_ag_rich)+'% of rich peoples\' incomes come from agriculture')

p_inc_ag_poor = round(100.*df.loc[(df.ispoor==True)&(df.frac_inc_agri>0),['Weight','HHsize']].prod(axis=1).sum()/df.loc[df.ispoor==True,['Weight','HHsize']].prod(axis=1).sum(),1)
p_inc_ag_rich = round(100.*df.loc[(df.ispoor==False)&(df.frac_inc_agri>0),['Weight','HHsize']].prod(axis=1).sum()/df.loc[df.ispoor==False,['Weight','HHsize']].prod(axis=1).sum(),1)

print('Odds of a Fijian in poverty having income from ag:',p_inc_ag_poor,'%')
print('Odds of a Fijian above poverty having income from ag:',p_inc_ag_rich,'%')

# clear first drawimg
#ax.clear()  # clear drawing axes
#cb.ax.clear()  # clear colorbar axes

# replace with new drawing
# 1. drawing new contour at drawing axes
#c_new = ax.contour(Z)  
## 2. create new colorbar for new contour at colorbar axes
#cb = ax.get_figure().colorbar()
#cb.ax.clear()

# Scatter plot of total income (all sources) vs frac of income from ag
ax = df.ix[(df.TotalIncome<10000)&(df.frac_inc_agri>0)].plot.hexbin('inc_pAE','frac_inc_agri',gridsize=25,alpha=1.)
cb = plt.gcf().get_axes()[1]
#c = ax.contourf(Z)   # contour fill c
#cb = fig.colorbar(c)  # colorbar for contour c
cb.clear()

plt.plot([rural_pov_line,rural_pov_line],[0.,1.],'k-',lw=1.5,color=greys_pal[7],zorder=100,alpha=0.85)
fig = ax.get_figure()

plt.xlabel(r'Total income [FJD yr$^{-1}$ per AE]')
plt.ylabel('Fraction of income from agriculture\n(in households with agricultural income)')
fig.savefig('../output_plots/FJ/sectoral/ag_income_frac_vs_total.pdf',format='pdf')#+'.pdf',format='pdf')
plt.clf()
plt.close('all')

# Scatter plot of total income (all sources) vs frac of income from fisheries
ax = df.ix[(df.TotalIncome<10000)&(df.frac_inc_fish>0)].plot.hexbin('inc_pAE','frac_inc_fish',gridsize=25,alpha=1.)
fig = ax.get_figure()
fig.savefig('../output_plots/FJ/sectoral/fish_income_frac_vs_total.pdf',format='pdf')#+'.pdf',format='pdf')
plt.clf()
plt.close('all')

# Plot reduced income from agriculture
ci_heights, ci_bins = np.histogram(df['inc_pAE'].clip(upper=20000), bins=50, weights=df[['Weight','AE']].prod(axis=1))
cf_heights, cf_bins = np.histogram(df['new_inc_pAE'].clip(upper=20000), bins=50, weights=df[['Weight','AE']].prod(axis=1))
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[1],alpha=0.4)
ax.bar(ci_bins[:-1], cf_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
# --> annotations
fig = ax.get_figure()
plt.plot([rural_pov_line,rural_pov_line],[0.,70000],'k-',lw=1.5,color=greys_pal[7],zorder=100,alpha=0.85)
plt.xlabel(r'Income [FJD yr$^{-1}$]')
plt.ylabel('Population')
plt.ylim(0,65000)
plt.legend(loc='best')
fig.savefig('../output_plots/FJ/sectoral/income_ag_reductions.pdf',format='pdf')
plt.clf()

# Plot total income from Fishprawnscrabsetc:
ax = plt.gca()
ci_heights, ci_bins = np.histogram(df.loc[df.Fishprawnscrabsetc > 0,'Fishprawnscrabsetc'].clip(upper=20000), bins=50, weights=df.loc[df.Fishprawnscrabsetc > 0,'Weight'])
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
# --> annotations
fig = ax.get_figure()
plt.xlabel(r'Income from fisheries [FJD yr$^{-1}$]')
plt.ylabel('Population')
plt.legend(loc='best')
ax.annotate('Total income: '+str(round(df[['Fishprawnscrabsetc','Weight']].prod(axis=1).sum()/1.E6,1))+'M FJD per year',
            xy=(10000,650),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
fm_str = str(int(total_hh_fishermen))
ax.annotate(fm_str[:2]+','+fm_str[-3:]+' Fijian households gain some\nincome from fisheries',xy=(10000,600),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_hh_fishermen_ita*100.,1))+'% i-Taukei',xy=(10000,525),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_hh_fishermen_ind*100.,1))+'% Indian-Fijian',xy=(10000,475),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
fig.savefig('../output_plots/FJ/sectoral/fisheries_income.pdf',format='pdf')#+'.pdf',format='pdf')
plt.clf()

# Plot fraction of income from Fishprawnscrabsetc:
ax = plt.gca()
ci_heights, ci_bins = np.histogram(df.loc[df.Fishprawnscrabsetc > 0,'Fishprawnscrabsetc']/df.loc[df.Fishprawnscrabsetc > 0,'TotalIncome'], 
                                   bins=50, weights=df.loc[df.Fishprawnscrabsetc > 0,'Weight'])
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
# --> annotations
fig = ax.get_figure()
plt.xlabel(r'Fraction of income from fisheries')
plt.ylabel('Population')
plt.legend(loc='best')
fig.savefig('../output_plots/FJ/sectoral/fisheries_income_frac.pdf',format='pdf')#+'.pdf',format='pdf')
plt.clf()

# Plot total income from Farming:
ax = plt.gca()
ci_heights, ci_bins = np.histogram(df.loc[df.TotalAgri > 0,'TotalAgri'].clip(upper=20000), bins=50, weights=df.loc[df.TotalAgri > 0,'Weight'])
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)    
# --> annotations
fig = ax.get_figure()
plt.xlabel(r'Income from agriculture [FJD yr$^{-1}$]')
plt.ylabel('Population')
plt.legend(loc='best')
ax.annotate('Total income: '+str(round(df[['TotalAgri','Weight']].prod(axis=1).sum()/1.E6,1))+'M FJD per year',xy=(10000,3200),
            xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
fm_hh = str(int(total_hh_farmers))
ax.annotate(fm_hh[:2]+','+fm_hh[2:]+' Fijian households gain\nsome income from agriculture',xy=(10000,3000),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_hh_farmers_ita*100.,1))+'% i-Taukei',xy=(10000,2700),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')
ax.annotate(str(round(frac_hh_farmers_ind*100.,1))+'% Indian-Fijian',xy=(10000,2500),xycoords='data',ha='left',va='top',fontsize=9,annotation_clip=False,weight='bold')

fig.savefig('../output_plots/FJ/sectoral/agri_income.pdf',format='pdf')#+'.pdf',format='pdf')
plt.clf()

# Plot fraction of income from agriculture (non-fish):
ax = plt.gca()
ci_heights, ci_bins = np.histogram(df.loc[df.TotalAgri > 0,'TotalAgri']/df.loc[df.TotalAgri > 0,'TotalIncome'], 
                                   bins=50, weights=df.loc[df.TotalAgri > 0,'Weight'])
ax.bar(ci_bins[:-1], ci_heights, width=(ci_bins[1]-ci_bins[0]), facecolor=q_colors[0],alpha=0.4)
# --> annotations   
fig = ax.get_figure()
plt.xlabel(r'Fraction of income from agriculture')
plt.ylabel('Population')
plt.legend(loc='best')
fig.savefig('../output_plots/FJ/sectoral/agri_income_frac.pdf',format='pdf')#+'.pdf',format='pdf')
plt.clf()

# SAVE files
df = df.reset_index()
prov_code = pd.read_excel(inputs+'Fiji_provinces.xlsx')[['code','name']].set_index('code').squeeze() 
df['Division'] = df.Division.replace(prov_code)
df = df.set_index('Division')

ag_df = df[['Weight','TotalIncome']].prod(axis=1).sum(level='Division').to_frame(name='TotalIncome')
ag_df['AgriIncome'] = df[['Weight','TotalAgri']].prod(axis=1).sum(level='Division')
ag_df['FishIncome'] = df[['Weight','Fishprawnscrabsetc']].prod(axis=1).sum(level='Division')
ag_df.to_csv('~/Desktop/my_plots/FJ_ag_incomes.csv')

#print(df)
assert(False)

# LOAD FILES (by hazard, asset class) and merge hazards
# load all building values

#df_bld_edu_tc =   pd.read_csv(inputs+'fiji_tc_buildings_edu_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
#desc_str = 'education'
df_bld_edu_tc =   pd.read_csv(inputs+'fiji_tc_buildings_health_tikina.csv').set_index('Tikina').drop('Country_ID',axis=1)
desc_str = 'health'

df_bld_edu_tc['Division'] = (df_bld_edu_tc['Tikina_ID']/100).astype('int')
prov_code = pd.read_excel(inputs+'Fiji_provinces.xlsx')[['code','name']].set_index('code').squeeze() 
df_bld_edu_tc['Division'] = df_bld_edu_tc.Division.replace(prov_code)
df_bld_edu_tc.drop('Tikina_ID',axis=1,inplace=True)

df_bld_edu_tc = df_bld_edu_tc.reset_index().set_index(['Division','Tikina']).rename(columns={'exceed_2':2475,'exceed_5':975,'exceed_10':475,
                                                                                             'exceed_20':224,'exceed_40':100,'exceed_50':72,
                                                                                             'exceed_65':50,'exceed_90':22,'exceed_99':10,'AAL':1})

df_bld_edu_tc = df_bld_edu_tc.stack()
df_bld_edu_tc /= 1.E6 # put into millions
df_bld_edu_tc = df_bld_edu_tc.unstack()

df_bld_edu_tc = df_bld_edu_tc.reset_index().set_index(['Division','Tikina','Exp_Value']).stack().to_frame(name='losses')
df_bld_edu_tc.index.names = ['Division','Tikina','Exp_Value','rp']
df_bld_edu_tc = df_bld_edu_tc.reset_index().set_index(['Division','Tikina','rp'])

summed = sum_with_rp('FJ',df_bld_edu_tc,['losses'],sum_provinces=False,national=False)

df_bld_edu_tc.sum(level=['Division','rp']).to_csv('~/Desktop/my_plots/'+desc_str+'_assets.csv')
summed.to_csv('~/Desktop/my_plots/'+desc_str+'_assets_AAL.csv')


df_bld_edu_tc['Exp_Value'] /= 100. # map code multiplies by 100 for a percentage
make_map_from_svg(
    df_bld_edu_tc['Exp_Value'].sum(level=['Division','rp']).mean(level='Division'), 
    '../map_files/FJ/BlankSimpleMap.svg',
    outname='FJ_'+desc_str+'_assets',
    color_maper=plt.cm.get_cmap('Blues'),
    label = desc_str[0].upper()+desc_str[1:]+' assets [million USD]',
    new_title = desc_str[0].upper()+desc_str[1:]+' assets [million USD]',
    do_qualitative=False,
    res=2000)
