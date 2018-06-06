import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

####################
####################
# Numerical attempt:

t_interval,_dt = np.linspace(0,10,52*10,retstep=True)

####################
# These don't change
eta = 1.5
pov_line = 10

####################
# These can change 
rng = [8.,4]
####################

pi = 0.333
pi_vals = [pi]#pi + pi/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ avg productivity of cap

rho = 0.06
rho_vals = [rho]#rho + rho/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ discount rate

c = 10*pov_line
c_vals = [c]#c + c/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ consumption expressed as multiple of poverty line

dk0 = 0.1*(c/pi)
dk0_vals = dk0 + dk0/rng[0]*np.array(range(-1*rng[1],rng[1]+1))
# ^ expressed as fraction of assets (= c/pi)

t_reco = 3.
t_reco_vals = t_reco + t_reco/rng[0]*np.array(range(-1*rng[1],rng[1]+1))

########################
# Calculate
dw_sums = []
dw_dict = {}
dw_val_dict = {}

for _pi in pi_vals:
    for _rho in rho_vals:
        for _c in c_vals:
            for _dk0 in dk0_vals:
                for _t_reco in t_reco_vals:
                    _R = np.log(1/0.05)/_t_reco

                    dw_tot = 0

                    print([_pi,_rho,_c,_dk0,_t_reco])

                    const = -_c**(1-eta)/(1-eta)
                    
                    for _t in t_interval:
                        
                        _dw = const*((1-(_dk0/_c)*(_pi+_R)*np.exp(-_t*_R))**(1-eta)-1)*np.exp(-_t*_rho)
                        dw_tot += _dw

                    dw_dict[len(dw_sums)] = [_pi,_rho,_c,_dk0,_t_reco]
                    dw_val_dict[dw_tot] = len(dw_sums)

                    dw_sums.append(dw_tot)
                    
plt.scatter(range(len(dw_sums)),dw_sums,s=5)
plt.xlabel('Scenario')
plt.xticks([])
plt.ylabel(r'$\int$dw')
plt.gcf().savefig('dw_optimization.pdf',format='pdf')

print(min(dw_sums),dw_dict[dw_val_dict[min(dw_sums)]])


####################
# Analytical:
c = 100.
dc = 10.

alpha = (1+rho)**(1/(1-eta))

#tau = -(pi*(alpha-2)+2)/(pi+alpha)
tau = pi*(c/dc-1)*(alpha-1)/(2*pi+alpha+1)

t_reco = np.log(1/0.05)/tau
print('tau = ', tau)
print('t_reco = ',t_reco)

####################
# Numero-analytical:
plt.cla()
plt.grid(True)

xmax = 5
tr_rng = np.array(np.linspace(0.5,xmax,1000))
vals = []

dc_rng = np.array(np.linspace(0.5,20.,1000))

for _ in dc_rng:
    
    #tr = np.log(1/0.05)/_
    #vals.append(tr/pi*(1+pi+alpha)-np.e**(-tr))
    #cval = c/dc*(alpha-1)-alpha    

    __ = pi*(alpha-1)/(pi+alpha+1)*((c/_)-1)
    vals.append(np.log(1/0.05)/__)

plt.xticks(1*np.array(range(0,21)))

plt.plot(dc_rng,vals)
#plt.plot(tr_rng,[cval for i in tr_rng])
plt.gcf().savefig('dev/optimize.pdf',format='pdf')


print('alpha = ',alpha)

####################
# Numero-analytical:
plt.cla()
vals2 = []
for tr in tr_rng:
    
    _ = (1-1/(1+rho))*(c/dc)**(1-eta)-(c/dc-np.log(1./0.05)/(tr*pi))**(1-eta)+1/(1+rho)*(c/dc+np.log(1/0.05)/(tr*pi)*(1+pi))**(1+eta)
    vals2.append(_)

plt.plot(tr_rng,vals2)
#plt.plot(tr_rng,[cval for i in tr_rng])
plt.gcf().savefig('dev/optimize2.pdf',format='pdf')

plt.close('all')


#####################
#
def welf(c,eta=1.5,t=0,rho=0.06):
    return(c**(1-eta)/(1-eta)*(1/(1+rho*t)))

print('')
