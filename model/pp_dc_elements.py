#This script processes data outputs for the resilience indicator multihazard model for the Philippines. Developed by Brian Walsh.
from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#Import packages for data analysis
#from libraries.lib_compute_resilience_and_risk import 
#from new_process_data import haz_dict
from libraries.replace_with_warning import *
from libraries.lib_country_dir import *
from libraries.lib_gather_data import *
from libraries.maps_lib import *
from libraries.lib_average_over_rp import average_over_rp

from scipy.stats import norm
import matplotlib.mlab as mlab

import matplotlib.patches as patches
from pandas import isnull
import pandas as pd
import numpy as np
import os, time
import pylab
import sys

#Aesthetics
import seaborn as sns
import brewer2mpl as brew
from matplotlib import colors
sns.set_style('darkgrid')
brew_pal = brew.get_map('Set1', 'qualitative', 8).mpl_colors
sns_pal = sns.color_palette('Set1', n_colors=17, desat=None)
greys_pal = sns.color_palette('Greys', n_colors=9)
reds_pal = sns.color_palette('Reds', n_colors=9)
q_labels = ['Q1 (Poorest)','Q2','Q3','Q4','Q5 (Wealthiest)']
q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

reg_pal = sns.color_palette(['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c',
                             '#008080','#e6beff','#fabebe','#800000','#808000','#000080','#808080','#000000'],n_colors=17,desat=None)

params = {'savefig.bbox': 'tight', #or 'standard'
          #'savefig.pad_inches': 0.1 
          'xtick.labelsize': 8,
          'ytick.labelsize': 14,
          'legend.fontsize': 12,
          'legend.facecolor': 'white',
          #'legend.linewidth': 2, 
          'legend.fancybox': True,
          'savefig.facecolor': 'white',   # figure facecolor when saving
          #'savefig.edgecolor': 'white'    # figure edgecolor when saving
          };plt.rcParams.update(params)
font = {'family' : 'sans serif', 'size'   : 8};plt.rc('font', **font)

import warnings
warnings.filterwarnings('always',category=UserWarning)

myCountry = 'SL'
economy = 'district'
if len(sys.argv) < 2:
    print('Could list country. Using SL.')
else: 
    myCountry = sys.argv[1]
    economy = {'PH':'region','SL':'district','FJ':'division'}[myCountry]
print('Running at',economy,'level in',myCountry)

model  = os.getcwd() #get current directory
output = model+'/../output_country/'+myCountry+'/'

def export_legend(legend, filename='../check_plots/legend.pdf', expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox,format='pdf')
    

# Load output files
pol_str = ''#'_v95'#could be {'_v95'}
base_str = 'no'
pds_str = 'unif_poor'

macro = pd.read_csv(output+'macro_tax_'+pds_str+'_.csv')

try:
    _hh = pd.read_csv('/Users/brian/Desktop/BANK/hh_resilience_model/check_plots/test_hh.csv').reset_index()
except:
    print('loading iah_full to get just 1 hh')
    _hh = pd.read_csv(output+'iah_full_tax_'+pds_str+'_'+pol_str+'.csv').reset_index()
    #_hh = _hh.loc[(_hh.hhid==153829114)&(_hh.hazard=='EQ')&(_hh.rp==100)&(_hh.affected_cat=='a')&(_hh.helped_cat=='helped')]
    _hh = _hh.loc[(_hh.hazard=='EQ')&(_hh.affected_cat=='a')].head(1)
    _hh.to_csv('/Users/brian/Desktop/BANK/hh_resilience_model/check_plots/test_hh.csv')

_hh['dc_post_reco'] = 0

# k recovery
const_dk_reco = float(_hh['hh_reco_rate'])
const_dk_reco_time = np.log(1/0.05)/float(_hh['hh_reco_rate'])
#const_pds     = (np.log(1/0.05)/3.)*2. # PDS consumed in first half year of recovery 
const_prod_k  = float(macro.avg_prod_k.mean())

print(_hh.head())

c   = float(_hh['c'])
di0 = float(_hh['di0'])
dc0 = float(_hh['dc0'])
#pds = float(_hh['help_received'])*3
savings_usage = dc0*0.5

k     = float(_hh['k'])
dk0   = float(_hh['dk0'])
dkprv = float(_hh['dk_private'])
dkpub = float(_hh['dk_public'])

c_t       = [] 
dc_k_t    = []
dc_reco_t = []
#dc_pds_t  = []

cw_t = []
w_t = []
ie = 1.5

t_lins = np.linspace(0,10,200)
for t in t_lins:
    c_t.append(c)

    dc1 = di0*np.e**(-(t)*const_dk_reco)
    dc_k_t.append(dc1)

    dc2 = dkprv*const_dk_reco*np.e**(-(t)*const_dk_reco)
    dc_reco_t.append(dc2)

    #dc3 = pds*const_pds*np.e**(-(t)*const_pds)
    #dc_pds_t.append(dc3)

    cw_t.append(c**(1-ie)/(1-ie))
    w_t.append(c**(1-ie)/(1-ie)-(c-min(savings_usage,(dc1+dc2)))**(1-ie)/(1-ie))

#plt.plot(t_lins,cw_t)
plt.fill_between(t_lins,0,w_t,color=sns_pal[1])
plt.draw()
plt.xlim(0,4)
plt.xlabel(r'Time $t$ after disaster ($\tau_h \equiv$ '+str(round(const_dk_reco_time,2))+') [years]')
plt.ylabel('Welfare loss')
#plt.ylim(0,-.1)
fig=plt.gcf()
fig.savefig('/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/Figures/dw.pdf',format='pdf')
plt.cla()

#step_dt*((1.-(temp['dc0']/temp['c'])*math.e**(-i_dt*const_reco_rate)+temp['help_received']*const_pds_rate*math.e**(-i_dt*const_pds_rate))**(1-const_ie)-1)*math.e**(-i_dt*const_rho)
# Indicate k(t): private and public 

# Lost income from capital
plt.fill_between(t_lins,c_t,[i-j for i,j in zip(c_t,dc_k_t)],facecolor=reds_pal[3],alpha=0.45)
plt.scatter(0,c_t[0]-dc_k_t[0],color=reds_pal[3],zorder=100)
plt.annotate('Income\nlosses',[-0.50,(c_t[0]+(c_t[0]-dc_k_t[0]))/2.],fontsize=9,ha='center',va='center',weight='bold')

# Reconstruction costs
plt.fill_between(t_lins,[i-j for i,j in zip(c_t,dc_k_t)],[i-j-k for i,j,k in zip(c_t,dc_k_t,dc_reco_t)],facecolor=reds_pal[4],alpha=0.45)
plt.scatter(0,c_t[0]-dc_k_t[0]-dc_reco_t[0],color=reds_pal[4],zorder=100)
plt.annotate('Reconstruction\ncosts',[-0.50,((c_t[0]-dc_k_t[0])+(c_t[0]-dc_k_t[0]-0.5*dc_reco_t[0]))/2.],fontsize=9,ha='center',va='center',weight='bold')

plt.gca().add_patch(patches.Rectangle((1.13,c_t[0]-1.12*dc_reco_t[15]),2.03,15000,facecolor='white',zorder=98,clip_on=False))
plt.gca().annotate(r'Area = total value of destroyed assets',
                   xy=(0.50,c_t[0]-dc_reco_t[15]), xycoords='data',
                   xytext=(1.20,c_t[0]-dc_reco_t[15]), textcoords='data', fontsize=10,
                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-0.05",lw=1.5),
                   ha='left',va='center',zorder=99)

# PDS
#plt.fill_between(t_lins,c_t,[i+j for i,j in zip(c_t,dc_pds_t)],facecolor=sns_pal[2],alpha=0.45)
#plt.annotate('PDS\nspend down',[-0.50,(c_t[0]+(c_t[0]+dc_pds_t[0]))/2.],fontsize=9,ha='center',va='center',weight='bold')

#plt.gca().add_patch(patches.Rectangle((0.73,c_t[0]+0.50*dc_pds_t[0]),1.40,15000,facecolor='white',zorder=98,clip_on=False))
#plt.gca().annotate(r'Area = total value of PDS',
#                   xy=(0.10,c_t[0]+0.50*dc_pds_t[0]), xycoords='data',
#                   xytext=(0.80,c_t[0]+0.75*dc_pds_t[0]), textcoords='data', fontsize=10,
#                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0.05",lw=1.5),
#                   ha='left',va='center',zorder=99)

# C net of everything except savings 
net_c_t = [i-j-k for i,j,k in zip(c_t,dc_k_t,dc_reco_t)]

# savings usage
plt.plot([t_lins[0],t_lins[7]],[savings_usage,savings_usage],color=greys_pal[7])
plt.fill_between(t_lins[:8],[savings_usage for i in t_lins[:8]],net_c_t[:8],facecolor=greys_pal[4])

plt.gca().add_patch(patches.Rectangle((0.73,0.84*savings_usage),1.89,15000,facecolor='white',zorder=98,clip_on=False))
plt.gca().annotate(r'Area = total value of savings + PDS',
                   xy=(0.10,0.9*savings_usage), xycoords='data',
                   xytext=(0.80,0.9*savings_usage), textcoords='data', fontsize=10,
                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-0.05",lw=1.5),
                   ha='left',va='center',zorder=99)

plt.gca().annotate('Savings + PDS\nexpenditure',
                   xy=(0.09,0.82*savings_usage), xycoords='data',
                   xytext=(-0.5,0.52*savings_usage), textcoords='data', fontsize=9,
                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-0.10",lw=1.5),
                   ha='center',va='center',zorder=99,weight='bold')


_c_net = []
for i in net_c_t:
    _c_net.append(max(savings_usage,i))

plt.plot(t_lins,_c_net,color=reds_pal[8],ls='--',lw=2.5,zorder=100,label='Household consumption')
plt.plot([-1,0],[c_t[0],c_t[0]],color=reds_pal[8],ls='--',lw=2.5,zorder=100,label='')
plt.plot([0,0],[c_t[0],savings_usage],color=reds_pal[8],ls='--',lw=2.5,zorder=100,label='')
leg = plt.gca().legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=1.0)

# poverty line
#plt.plot([-10,t_lins[-1]],[_hh.pov_line,_hh.pov_line])

# Draw c
plt.plot([-1,5],[c,c],color=greys_pal[8])

plt.xlim(-1,4)
#plt.ylim((c-dc0)*0.98,c*1.02)

plt.xlabel(r'Time $t$ after disaster ($\tau_h \equiv$ '+str(round(const_dk_reco_time,2))+') [years]')
plt.ylabel(r'Household consumption ($c_h$)')
plt.xticks([-1,0,1,2,3,4],['-1',r'$t_0$','1','2','3','4'])
plt.yticks([c_t[0]],[r'$c_0$'])

plt.draw()
fig=plt.gcf()
fig.savefig('/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/stephane_edits/Figures/dc.pdf',format='pdf')

plt.clf()
plt.close('all')

# Draw k
plt.plot([-1,5],[k,k],color=greys_pal[8])

# points at t0
plt.scatter(0,k-dk0,color=reds_pal[5],zorder=100)
plt.scatter(0,k-dkprv,color=reds_pal[3],zorder=100)
# Annotate 
plt.annotate('Private\nasset\nlosses',[-0.65,k-dkprv/2.],fontsize=9,ha='center',va='center',weight='bold')
plt.annotate(r'$\Delta k^{prv}_0$',[-0.2,k-dkprv/2.],fontsize=10,ha='center',va='center')
plt.plot([-0.2,-0.2],[k-dkprv,k-1.1*dkprv/2.],color=reds_pal[3])
plt.plot([-0.2,-0.2],[k-0.9*dkprv/2.,k],color=reds_pal[3])
plt.plot([-0.22,-0.18],[k,k],color=reds_pal[3])
plt.plot([-0.22,-0.18],[k-dkprv*0.997,k-dkprv*0.997],color=reds_pal[3],zorder=100)

plt.annotate('Public\nasset\nlosses',[-0.65,k-dk0+dkpub/2.],fontsize=9,ha='center',va='center',weight='bold')
plt.annotate(r'$\Delta k^{pub}_0$',[-0.2,k-dk0+dkpub/2.],fontsize=10,ha='center',va='center')
plt.plot([-0.2,-0.2],  [k-dkprv-dkpub,(k-dkprv)-1.5*dkpub/2.],color=reds_pal[5])
plt.plot([-0.2,-0.2],  [(k-dkprv)-0.5*dkpub/2.,k-dkprv],color=reds_pal[5])
plt.plot([-0.22,-0.18],[k-dkprv*1.003,k-dkprv*1.003],color=reds_pal[5])
plt.plot([-0.22,-0.18],[k-dkprv-dkpub,k-dkprv-dkpub],color=reds_pal[5])

plt.annotate('Disaster\n'+r'(t = t$_0$)',[0,k*1.005],fontsize=9,ha='center',weight='bold')
plt.plot([0,0],[k-dk0,k],color=sns_pal[0])

# k recovery
k_t     = []
dk0_t   = []
dkprv_t = []
dkpub_t = []

for t in t_lins:
    k_t.append(k)
    dk0_t.append(k-(dk0*np.e**(-t*const_dk_reco)))
    dkprv_t.append(k-(dkprv*np.e**(-t*const_dk_reco)))
    dkpub_t.append(k-(dkpub*np.e**(-t*const_dk_reco)))


# Indicate k(t): private and public 
plt.fill_between(t_lins,k_t,dkprv_t,facecolor=reds_pal[3],alpha=0.45)
plt.fill_between(t_lins,dkprv_t,[i-(k-j) for i,j,k in zip(dkprv_t,dkpub_t,k_t)],facecolor=reds_pal[5],alpha=0.45)

plt.plot([2.1,2.1],[k-0.05*dk0,k],color=reds_pal[8])
plt.plot([2.08,2.12],[k-0.05*dk0,k-0.05*dk0],color=reds_pal[8],zorder=100)
plt.plot([2.08,2.12],[k,k],color=reds_pal[8],zorder=100)

plt.gca().add_patch(patches.Rectangle((2.45,k-0.12*dk0),1.40,11000,facecolor='white',zorder=98,clip_on=False))
plt.gca().annotate(r'$\Delta k_h|_{t=\tau_h}$ = 0.05$\times\Delta k_0$',
                   xy=(2.1,k-0.025*dk0), xycoords='data',
                   xytext=(2.5,k-0.075*dk0), textcoords='data', fontsize=10,
                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-0.05",lw=1.5),
                   ha='left',va='center',zorder=99)

plt.plot(t_lins,dk0_t,color=reds_pal[8],ls='--',lw=2.5,label='Household capital')
plt.plot([-1,0],[k,k],color=reds_pal[8],ls='--',lw=2.5,label='')
plt.plot([0,0],[k,dk0_t[0]],color=reds_pal[8],ls='--',lw=2.5,label='')

leg = plt.gca().legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=1.0)

plt.gca().add_patch(patches.Rectangle((1.40,dk0_t[10]*1.003),1.50,11000,facecolor='white',zorder=98))
plt.gca().annotate(r'$\Delta k_h(t) = \Delta k_0e^{-R_{\tau}\cdot t}$',
                   xy=(t_lins[20],dk0_t[20]), xycoords='data',
                   xytext=(1.45,dk0_t[10]*1.01), textcoords='data', fontsize=12,
                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0.2",lw=1.5),
                   ha='left',va='center',zorder=99)

plt.xlim(-1,5)
plt.ylim((k-dk0)*0.98,k*1.02)

plt.xlabel(r'Time $t$ after disaster ($\tau_h \equiv '+str(round(const_dk_reco_time,2))+'$) [years]')
#plt.xlabel(r'Time $t$ after disaster (years)')
plt.ylabel(r'Effective household capital ($k_h$)')
plt.xticks([-1,0,1,2,3,4],['-1',r'$t_0$','1','2','3','4'])
plt.yticks([k_t[0]],[r'$k_h$'])

plt.draw()
fig=plt.gcf()
fig.savefig('/Users/brian/Desktop/Dropbox/Bank/unbreakable_writeup/stephane_edits/Figures/dk.pdf',format='pdf')

summary_df = pd.read_csv('../output_country/'+myCountry+'/my_summary_no.csv').reset_index()
summary_df['res_mean'] = summary_df.groupby([economy,'hazard'])['res_tot'].transform('median')
summary_df = summary_df.loc[(summary_df.dk_tot>=0.1)&(summary_df.res_tot<=1.25*summary_df.res_mean)].sort_values(by=['hazard',economy,'rp'])

all_regions = np.unique(summary_df[economy].dropna())

haz_dict = {'SS':'Storm surge','PF':'Precipitation flood','HU':'Hurricane','EQ':'Earthquake'}
xax = [['rp',' return period [years]'],['dk_tot',' asset losses [PhP]'],['dw_tot',' welfare losses [PhP]']]
plotz = []
regz = []

for ix in xax:
    for iHaz in ['SS','PF','HU','EQ']:

        _ = summary_df.loc[(summary_df.hazard==iHaz)]

        regions = _.groupby(economy)
    
        fig,ax = plt.subplots()
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
        col_ix = 0
        for name, iReg in regions:

            while name != all_regions[col_ix]:
                col_ix += 1
        
            try: plotz.append(ax.semilogx(iReg[ix[0]],(100*iReg.res_tot), marker='.', linestyle='', ms=9, label=name,color=reg_pal[col_ix]))
            except: plotz.append(ax.semilogx(iReg[ix[0]],(100*iReg.res_tot), marker='.', linestyle='', ms=9, label=name))
            
            regz.append(name)
            col_ix+=1

        plt.xlabel(haz_dict[iHaz]+ix[1],fontsize=16)
        plt.ylabel('Socio-economic capacity [%]',fontsize=16)
        plt.ylim(0,250)

        if False:
            leg = ax.legend(loc='best',labelspacing=0.75,ncol=4,fontsize=6,borderpad=0.75,fancybox=True,frameon=True,framealpha=1.0,title=economy[0].upper()+economy[1:])
            export_legend(leg,'../check_plots/regional_legend.pdf')            

        col_ix = 0
        for name, iReg in regions:

            while name != all_regions[col_ix]:
                col_ix += 1

            try: ax.plot(iReg[ix[0]],(100*iReg.res_tot),color=reg_pal[col_ix])
            except: ax.plot(iReg[ix[0]],(100*iReg.res_tot))
            col_ix+=1

        fig.savefig('../check_plots/reg_resilience_vs_'+ix[0]+'_'+iHaz+'.pdf',format='pdf')
        #plt.clf()
        plt.close('all')


summary_df = pd.read_csv('../output_country/'+myCountry+'/my_summary_no.csv').reset_index().set_index([economy,'hazard','rp'])[['dk_tot','dw_tot','res_tot']]
v_df = pd.read_csv('../output_country/'+myCountry+'/fa_v.csv').reset_index().set_index([economy,'hazard','rp'])
summary_df['fa'] = v_df['fa'].round(2).clip(upper=0.95)

summary_df.loc[summary_df.fa==0.95,'dk_famax'] = summary_df.loc[summary_df.fa==0.95,'dk_tot']
summary_df.loc[summary_df.fa==0.95,'dw_famax'] = summary_df.loc[summary_df.fa==0.95,'dw_tot']

summary_df['dk_famax'] = summary_df['dk_famax'].fillna(0)
summary_df['dw_famax'] = summary_df['dw_famax'].fillna(0)

rp_sum,_ = average_over_rp(summary_df[['dk_tot','dw_tot','dk_famax','dw_famax']])

rp_sum['res_tot'] = rp_sum['dk_tot']/rp_sum['dw_tot']
rp_sum['res_fa095'] = (rp_sum['dk_famax']/rp_sum['dw_famax']).fillna(rp_sum['res_tot'])
rp_sum['res_fa_less_095'] = (rp_sum['dk_tot']-rp_sum['dk_famax'])/(rp_sum['dw_tot']-rp_sum['dw_famax'])

_df = rp_sum[['res_tot','res_fa095','res_fa_less_095']].stack()
_df = _df.reset_index()
_df.columns = [economy,'hazard','res_type','res']

_df = _df.reset_index().set_index([economy,'hazard'])
_df['res_sort'] = _df.loc[_df.res_type=='res_tot','res']
_df = _df.reset_index()

print(_df.head())

for _,_h in _df.groupby('hazard'):

    ax = _h.boxplot('res',by=economy)
    fig = plt.gcf()
    plt.title('Regional resilience to '+_)
    plt.xlabel(economy[0].upper()+economy[1:])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        tick.label.set_rotation('vertical')

    plt.ylabel('Resilience')

    fig.savefig('../check_plots/resilience_'+_+'.pdf',format='pdf')

    plt.cla()
    plt.close('all')
