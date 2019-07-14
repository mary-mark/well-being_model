import os
import pandas as pd
from libraries.lib_gather_data import *
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)
greys_pal = sns.color_palette('Greys', n_colors=9)

def plot_simple_hist(df,cols,labels,fout,nBins=50,uclip=None,lclip=None,xlab='Income',logy=False):
    q_colors = [sns_pal[0],sns_pal[1],sns_pal[2],sns_pal[3],sns_pal[5]]

    plt.cla()
    ax = plt.gca()

    bin0 = None
    weighted = True
    for nCol, aCol in enumerate(cols):
        try:
            heights, bins = np.histogram(df[aCol].clip(upper=uclip,lower=lclip), bins=nBins, weights=df['hhwgt'])
        except:
            heights, bins = np.histogram(df[aCol].clip(upper=uclip,lower=lclip), bins=nBins)
            weighted = False
  
        if bin0 is None: bin0 = bins
        ax.bar(bin0[:-1], heights, width=(bin0[1]-bin0[0]), label=labels[nCol], facecolor=q_colors[nCol],alpha=0.4)
        
    fig = ax.get_figure()
    if logy: plt.yscale('log', nonposy='clip')
    plt.xlabel(xlab)
    if weighted: plt.ylabel('Households')
    if not weighted: plt.ylabel('HIES entries')

    try:
        leg = ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=9,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
        leg.get_frame().set_color('white')
        leg.get_frame().set_edgecolor(greys_pal[7])
        leg.get_frame().set_linewidth(0.2)
    except: pass

    fig.savefig(fout,format='pdf')#+'.pdf',format='pdf')
