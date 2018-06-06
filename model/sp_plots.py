import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os, glob

from libraries.lib_common_plotting_functions import title_legend_labels

import seaborn as sns
sns.set_style('darkgrid')
mpl.rcParams['legend.facecolor'] = 'white'

myCountry = 'SL'
out = '../output_plots/'+myCountry+'/'
to_usd = 1./153

def get_sp_costs(myCountry):
    sp_pols, sp_files = [], {}

    dir = '../output_country/'+myCountry+'/sp_costs'
    
    for f in glob.glob(dir+'*'):
        
        sp_pol = f.replace(dir+'_','').replace('.csv','')
        sp_pols.append(sp_pol)

        sp_files[sp_pol] = pd.read_csv(f,index_col=['district','hazard','rp'])

    return sp_pols, sp_files

def get_sort_order(sp_pols,sp_files,crit):
    sort_order = []

    while len(sort_order) != len(sp_pols):
        val = [0,'']
        for _sp in sp_pols:

            if _sp not in sort_order and val[0] <= sp_files[_sp][crit].mean(): 
                val[0] = sp_files[_sp][crit].mean()
                val[1] = _sp

        sort_order.append(val[1])

    return sort_order
            

sp_pols, sp_files = get_sp_costs(myCountry)

# Plot costs of each SP when event occurs
sort_order = get_sort_order(sp_pols,sp_files,'avg_natl_cost')
print(sort_order)

for _sp in sort_order:

    cost = ' (annual cost = '+str(round(sp_files[_sp]['avg_natl_cost'].mean()*to_usd/1.E6,2))+'M)'
    if _sp == 'no': cost = ''
    
    plt.plot(sp_files[_sp].sum(level='rp').index,sp_files[_sp]['event_cost'].sum(level='rp')*to_usd/1.E6,label=_sp+cost)

title_legend_labels(plt,'Sri Lanka','Return Period [years]','SP cost when event occurs [M USD]',[1,1000])
plt.xscale('log')
plt.gca().get_figure().savefig(out+'sp_costs.pdf',format='pdf')

