import matplotlib.pyplot as plt
import math

params = {'savefig.bbox': 'tight',
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.fontsize': 9,
          'legend.facecolor': 'white',
          'legend.fancybox': True
          }
font = {'family':'sans serif', 'size':16}
plt.rcParams.update(params)
plt.rc('font', **font)

#######################
_rho = 0.06
_eta = 1.5

_k   = [  100,   100]
_v   = [0.5, 0.5]
_fa  = [0.25, 0.25]
_pi  = [0.336,  0.50]
#######################

t_reco = [1,2,3,4,5] 
dwdt = [[],[]]
for _rt in t_reco:
    
    dwdt[0].append(math.log(20)*_k[0]**(1-_eta)*_v[0]*_fa[0]*(_pi[0]/_rho-_v[0]*_fa[0]*(_rt*_pi[0]+math.log(20))/(_rt*_pi[0]+math.log(20)))**(-_eta)*(_pi[0]-_rho)/(_rt*_rho+math.log(20))**2)
    dwdt[1].append(math.log(20)*_k[1]**(1-_eta)*_v[0]*_fa[1]*(_pi[1]/_rho-_v[0]*_fa[1]*(_rt*_pi[1]+math.log(20))/(_rt*_pi[1]+math.log(20)))**(-_eta)*(_pi[1]-_rho)/(_rt*_rho+math.log(20))**2)

plt.plot(t_reco,dwdt[0],label='C0')
plt.plot(t_reco,dwdt[1],label='C1')
plt.draw()
plt.legend()

plt.xlabel('Reconstruction time')
plt.ylabel(r'$\frac{dw}{dt}$')
plt.ylim(0)

fig = plt.gcf()
fig.savefig('/Users/brian/Desktop/BANK/hh_resilience_model/check_plots/dwdt.pdf',format='pdf')

