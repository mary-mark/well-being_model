import pandas as pd
import numpy as np

def smart_savers(c,dk,lam,pi,Vsav):

    #_a = temp[['dc0_prv','dc0_pub','help_received','sav_f','hh_reco_rate']].copy()
    #
    #_a['sav_offset_to'] = 0.25*(_a['dc0_prv']+_a['dc0_pub']-_a['help_received']*const_pds_rate) 
    ## ^ will remain at this level for hh that don't reconstruct
    #
    #savings_offset = '((dc0_prv+dc0_pub-help_received*@const_pds_rate-sav_f*((help_received*(@const_pds_rate)**2-dc0_prv*hh_reco_rate-dc0_pub*@const_pub_reco_rate)/(help_received*@const_pds_rate-dc0_prv-dc0_pub)))/(1.-sav_f*((help_received*(@const_pds_rate)**2-dc0_prv*hh_reco_rate-dc0_pub*@const_pub_reco_rate)/(help_received*@const_pds_rate-dc0_prv-dc0_pub)**2)))'
    #
    #_a.loc[(_a.hh_reco_rate!=0),'sav_offset_to'] = 0.65*_a.loc[(_a.hh_reco_rate!=0)].eval(savings_offset)
    
    # sav_offset_to is going to be used to determine dc_net
    # Lower clip = 0: dc can't go negative
    # Upper clip = (c-c_min): hh is obliged to stay out of subsistence if at all possible

    #_a['max'] = temp.eval('c-c_min').clip(lower=0.)
    #return _a['sav_offset_to'].clip(lower=0., upper=_a['max'])

    if dk == 0: return 0,10
    if lam == 0: return Vsav/10,10

    gamma = 0.02*dk*pi
    last_result = None

    while True:
        
        beta = gamma/(dk*(pi+lam))
        result = dk*(pi+lam)*(beta-1)-gamma*np.log(beta)+lam*Vsav
        
        try:
            if (last_result < 0 and result > 0) or (last_result > 0 and result < 0):

                _t = -np.log(beta)/lam
                #print('RESULT!:\ngamma = ',gamma,'& beta = ',beta,' & t = ',_t)
                #print('CHECK:',-dk*(pi+lam)*np.e**(-lam*_t),' gamma = ',gamma)
                return int(gamma),round(_t,3)

        except: pass

        last_result = result
        gamma += 0.02*dk*pi
        if (gamma > c): return 0,10

def optimize_reco(pi, rho, v, verbose=False):

    if v == 0: return 0

    eta = 1.5

    last_integ = None
    last_lambda = None
    
    _l = 0.0
    while True:

        if pi-(pi+_l)*v < 0: assert(False)
        
        x_max = 15
        dt_step = 52*x_max

        integ = 0
        for _t in np.linspace(0,x_max,dt_step):

            integ += np.e**(-_t*(rho+_l)) * ((pi+_l)*_t-1) * (pi-(pi+_l)*v*np.e**(-_l*_t))**(-eta)

        if last_integ and ((last_integ < 0 and integ > 0) or (last_integ > 0 and integ < 0)):
            #print('\n Found the Minimum!\n lambda = ',last_lambda,'--> integ = ',last_integ)
            #print('lambda = ',_l,'--> integ = ',integ)
            return (_l+last_lambda)/2
            
        last_integ = integ
        last_lambda = _l
        _l += 0.01
        
#print(optimize_reco())
#smart_savers(None,None,None,None)
