import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import optimize

def f(x,dk,pi,lambd,sav):
    beta = x/(dk*(pi+lambd))
    expression = dk*(pi+lambd)*(1-beta) + x * np.log(beta) - lambd*sav
    return expression

def f_w_labour(x,dk,pi,lambd,Vsav,di_labour,hh_share,k_str,v,p_rent,k_h,dt):
    x = min(0,x)
    x = max(10,x)
    labour_index = int(round(x*52,0))
    labour_income = di_labour.loc[np.arange(0,labour_index)].sum()*dt 
    if labour_index == 520: labour_index = 519
    di_t = di_labour.loc[labour_index]
    gamma = (di_t + k_h*v*pi*np.exp(-lambd*x)
        - v*p_rent*np.exp(-lambd*x) 
        + lambd*v*k_str*hh_share*np.exp(-lambd*x))
    expression = (gamma*x + Vsav - labour_income 
        - k_h*v*pi*(1-np.exp(-lambd*x))/lambd 
        + v*p_rent*(1-np.exp(-lambd*x))/lambd 
        - hh_share*v*k_str*(1-np.exp(-lambd*x)))
    return expression

def smart_savers(c,k,dk,lambd,pi,Vsav,idx,di_labour = None,
    k_oth = None,v=None,hh_share = None, k_str = None,k_h = None, p_rent = None,
    labourIncome = False):

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

    if labourIncome:
        print 'With labour income, Iter:', idx
        t_min = 0.
        t_max = 10.
        n_steps = 52.*10.
        int_dt, dt = np.linspace(t_min,t_max,num=n_steps,endpoint=True,retstep=True)
        if dk == 0: return 0,10,0 # no loss in consumption
        if (di_labour.sum()*dt + k_h*v*pi*(1-np.exp(-lambd*t_max))/lambd 
            + v*p_rent*(np.exp(-lambd*t_max)-1)/lambd 
            + v*k_str*hh_share*(1-np.exp(-lambd*t_max))) < Vsav: 
            return 0,10,0 # no loss in consumption since you can pay it out of savings
        if lambd == 0: 
            print('Lambda',0)
            return 999999,10,Vsav/10.
       
        '''
        print('TRYING NETWON METHOD')
        _t = optimize.newton(f_w_labour,4,args=(dk,pi,lambd,Vsav,di_labour,hh_share,k_str,v,p_rent,k_h,dt),maxiter = 10000, tol = 1e-8)
        labour_index = int(round(_t*52,0))
        labour_income = di_labour.loc[np.arange(0,labour_index)].sum()*dt 
        gamma = (labour_income + k_h*v*pi*np.exp(-lambd*_t) 
            - v*p_rent*np.exp(-lambd*_t) 
            + lambd*v*k_str*hh_share*np.exp(-lambd*_t))
        
        if _t == np.float('nan'):
            raise('THIS IS A NAN')
        elif gamma == np.float('nan'):
            raise('THIS IS A NAN')
        elif (gamma > c): 
            raise('gamma > c')
        elif(_t<0):
            raise('_t<0')
        else:
            print int(gamma),round(_t,3)
            return int(gamma),round(_t,3)
        '''
        t_possible = np.arange(0,10,0.05) 
        for x in t_possible:
            labour_index = int(round(x*52,0))
            labour_income = di_labour.loc[np.arange(0,labour_index)].sum()*dt 
            if labour_index == 520: labour_index = 519
            di_t = di_labour.loc[labour_index]
            gamma = (di_t + k_h*v*pi*np.exp(-lambd*x)- v*p_rent*np.exp(-lambd*x)  + lambd*v*k_str*hh_share*np.exp(-lambd*x))
            result = (gamma*x + Vsav - labour_income - k_h*v*pi*(1-np.exp(-lambd*x))/lambd  + v*p_rent*(1-np.exp(-lambd*x))/lambd - hh_share*v*k_str*(1-np.exp(-lambd*x)))  
            if x == 0: last_result = result
            if (last_result < 0 and result > 0) or (last_result > 0 and result < 0):
                _t = x
                if _t == np.float('nan'): raise('THIS IS A NAN')
                elif gamma == np.float('nan'): raise('THIS IS A NAN')
                elif (gamma > c):  raise('gamma > c')
                elif(_t<0): raise('_t<0')
                else:
                    print int(gamma),round(_t,3),0
                    return int(gamma),round(_t,3),0
                     #print('RESULT!:\ngamma = ',gamma,'& beta = ',beta,' & t = ',_t)
                     #print('CHECK:',-dk*(pi+lam)*np.e**(-lam*_t),' gamma = ',gamma)
            last_result = result
        print('COULDNT FIND SOLUTION results',last_result, result)
        return 999999,10,Vsav/10.
        '''
        x = 2.01
        labour_index = int(round(x*52,0))
        labour_income = di_labour.loc[np.arange(0,labour_index)].sum()*dt 
        di_t = di_labour.loc[labour_index]
        gamma = (di_t + k_h*v*pi*np.exp(-lambd*x)- v*p_rent*np.exp(-lambd*x)  + lambd*v*k_str*hh_share*np.exp(-lambd*x))
        print int(gamma),round(_t,3)
        return int(gamma),round(_t,3) 
        '''
        
    #### LABOUR INCOME NOT INCLUDED - ORIGINAL MODEL
    else:
        print 'Iter:', idx
        if dk == 0: return 0,10 # no loss in consumption
        if dk*np.exp(-10*lambd)*(np.exp(10*lambd)-1)*(lambd+pi)/lambd < Vsav: return 0,10 # no loss in consumption since you can pay it out of savings
        if lambd == 0: 
            print 'Lambda',0
            return Vsav/10,10
       
        try:
            gamma = optimize.newton(f,3,args=(dk,pi,lambd,Vsav),maxiter = 50000, tol = 1e-3)
            beta = gamma/(dk*(pi+lambd))
            _t = -np.log(beta)/lambd

            if gamma == np.float('nan'):
                print 'THIS IS A NAN'

            if (gamma > c): 
                return 0,10
            elif(_t<0):
                return 0,10
            else:
                return int(gamma),round(_t,3)
        except:
            print 'OOPS'
            gamma = 0.02*dk*pi
            last_result = None
            while True:
                beta = gamma/(dk*(pi+lambd))
                result = dk*(pi+lambd)*(beta-1)-gamma*np.log(beta)+lambd*Vsav
                try:
                    if (last_result < 0 and result > 0) or (last_result > 0 and result < 0):
                        _t = -np.log(beta)/lambd
                     #print('RESULT!:\ngamma = ',gamma,'& beta = ',beta,' & t = ',_t)
                     #print('CHECK:',-dk*(pi+lam)*np.e**(-lam*_t),' gamma = ',gamma)
                        if _t>=0:
                            return int(gamma),round(_t,3)
                        else:
                            return 0,10
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
        

def optimize_reco_w_labour_and_physical_constraint(pi, rho, myCountry,df_v,ind_names,L_recov_dic,L_pcinc_dic,d_constr_recon_rate,fault_identifier,verbose=False,suffix = ''):
    # This function computes the optimal reconstruction rate for each of the census tracts in df_v
    try: 
        nsims = len(df_v.index.get_level_values(0).unique())
        input_list = pickle.load(open('optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/"+myCountry+"_recon_rate_opt_w_labour_"+fault_identifier+"_"+str(nsims)+"sims.p",'rb'))
        recon_rate_opt = input_list[0]
        lambda_opt = input_list[1]
        lambda_physical = input_list[2]
        lambda_constr = input_list[3]
        recovery_constrain = input_list[4]
        
    except:
        print('Was not able to load v to hh_reco_rate library from optimization_libs/'+myCountry+'_v_to_reco_rate.p')
#         v_to_reco_rate = {}
     #   if suffix == '_retrofit':
     #   	d_recovery_rate = pickle.load(open('optimization_libs/HH_physical_recovery_rate_'
     #   		+fault_identifier+'_DEC2018'+suffix+'.p','rb'))
     #   else:
        d_recovery_rate = pickle.load(open('optimization_libs/HH_physical_recovery_rate_'
        		+fault_identifier+'_DEC2018'+suffix+'.p','rb'))

        nsims = len(df_v.index.get_level_values(0).unique())
        tracts = df_v.index.get_level_values(1).unique()
        recon_rate_opt = pd.DataFrame(index = df_v.index)
        recon_rate_opt['lambda'] = 0
        
        # For plotting
        lambda_opt = np.zeros(nsims* len(tracts))
        recovery_constrain = np.zeros(nsims* len(tracts))
        lambda_physical = np.zeros(nsims* len(tracts))
        lambda_constr = np.zeros(nsims* len(tracts))
        
        count = 0
        x_max = 10
        dt_steps = 13*x_max
        
        # Set up the time vector to iterate over
        t_recov = np.append(np.linspace(30,11,num = 20),np.linspace(10,4./52,num = 10*13))
        t_recov = np.append(t_recov,1./52)
        t_recov = np.append(t_recov,1./365)
        index_for_labour = np.append(np.array([0,1]),np.arange(4,520,4))
        t_for_labour = np.append(np.array([1./365,1./52]),np.linspace(4./52,10,num = 10*13))
        t_for_labour = t_for_labour[0:-1]
                        
        for sim in np.arange(1,nsims+1):
            sim
            for i,tract in enumerate(tracts):
                labour_recovery = np.zeros(52*x_max)
            	# Get income recovery curve:
                for ind in ind_names:
                    labour_recovery += L_recov_dic[(ind,sim)]*L_pcinc_dic[ind][slice(None),tract].values[0]

                v =  df_v.loc[(sim,tract)].v
                v_ins =  df_v.loc[(sim,tract)].v_ins
                mort_pc = df_v.loc[(sim,tract)].mort_pc
                #pcinc_tot =  df_v.loc[(sim,tract)].pcinc_tot - mort_pc
                c_pc =  df_v.loc[(sim,tract)].pcinc_tot - mort_pc
                k_pc_h =  df_v.loc[(sim,tract)].k_pc_h
                k_pc_str =  df_v.loc[(sim,tract)].k_pc_str
                p_rent_pc =  df_v.loc[(sim,tract)].p_rent_pc
                hh_share =  df_v.loc[(sim,tract)].hh_share
                c_pc_min =  df_v.loc[(sim,tract)].c_pc_min

                rate_physical = d_recovery_rate[(tract,sim)]
                rate_constr = 9999#d_constr_recon_rate[(tract,sim)]
                lambda_physical[count] = rate_physical
                lambda_constr[count] = rate_constr

                # If there is no asset loss assume the recovery rate is maximum (1 day)
                if v == 0:
                    recon_rate_opt.loc[(sim,tract),'lambda']  = 1093.44
                    lambda_opt[count] = 1093.44 
                    recovery_constrain[count] = 0

                else: 
                    rate_c_min = (c_pc - labour_recovery[0] - k_pc_h*v*pi + v*p_rent_pc) /(v*k_pc_str*hh_share)
                    print 'Rate for minimum consumption:',rate_c_min
                    eta = 1.5
                    last_integ = None
                    last_lambda = None
                    
                    for t in t_recov:
                        _l = np.log(1/0.05)/t
                        
                        integ = 0
                        for j,_t in enumerate(t_for_labour):
                            integ += np.e**(-_t*(rho+_l)) * (c_pc - labour_recovery[index_for_labour[j]] - k_pc_h*v*pi*np.e**(-_l*_t) + v*p_rent_pc*np.e**(-_l*_t) - _l*v*k_pc_str*hh_share*np.e**(-_l*_t))**(-eta) * (_t*v*k_pc_h*pi - _t*v*p_rent_pc + _l*_t*v*k_pc_str*hh_share - v*k_pc_str*hh_share)

                        if last_integ and ((last_integ < 0 and integ > 0) or (last_integ > 0 and integ < 0)):
                        #print('\n Found the Minimum!\n lambda = ',last_lambda,'--> integ = ',last_integ)
                        #print('lambda = ',_l,'--> integ = ',integ)

                            print 'Sim:',sim,'Tract:',i,'#',tract,'Rate consumption:',(_l+last_lambda)/2
                            recon_rate_opt.loc[(sim,tract),'lambda'] = (_l+last_lambda)/2
                            lambda_opt[count] =  (_l+last_lambda)/2
                            recovery_constrain[count] = 1 # 1 is for optimized consumption constraint
                            #print ('Found the Minimum',recon_rate_opt.loc[(sim,tract),'lambda'],'Physical',rate_physical)
                            break

                        elif _l > rate_physical:
                            
                            print 'Sim:',sim,'Tract:',i,'#',tract,'Rate physical:',rate_physical
                            recon_rate_opt.loc[(sim,tract),'lambda']  = rate_physical
                            lambda_opt[count] = rate_physical
                            recovery_constrain[count] = 2 # 2 is constrainted by physical rate
                            break

                        elif _l >= rate_c_min:
                            print 'Sim:',sim,'Tract:',i,'#',tract,'Rate c_min',rate_c_min
                            recon_rate_opt.loc[(sim,tract),'lambda']  = rate_c_min
                            lambda_opt[count] = rate_c_min
                            recovery_constrain[count] = 3 # 3 recovery rate is such that your consumption is not <0
                            break
                        
                       # elif _l >= rate_constr:
                       #     print 'Sim:',sim,'Tract:',i,'#',tract,'Rate construction sector',rate_constr
                       #     recon_rate_opt.loc[(sim,tract),'lambda']  = rate_constr
                        #    lambda_opt[count] = rate_constr
                        #    recovery_constrain[count] = 4 # 2 is constrainted by construction sector
                        #    break


                        if  (c_pc - labour_recovery[0] - k_pc_h*v*pi + v*p_rent_pc- _l*v*k_pc_str*hh_share) < 0: 
                            print _l,'MINIMUM',rate_c_min
                            assert(False)

                        last_integ = integ
                        last_lambda = _l
                        
                    # while True:
                    #         if (k_pc*pi - labour_recovery[0] - k_pc_oth*v*pi - _l*loss_pc*hh_share) < 0: 
                    #         assert(False)
                    #     if loss_pc == 0:
                    #         lambda_cons[count] = 0
                    #         lambda_opt[count] = rate_physical
                    #         recon_rate_opt.loc[(sim,tract),'lambda']  = rate_physical
                    #         break

                    #     integ = 0
                    #     for j,_t in enumerate(np.linspace(0,x_max,dt_step)):
                    #         integ += np.e**(-_t*(rho+_l)) * (k_pc*pi - labour_recovery[j] - k_pc_oth*v*pi*np.e**(-_l*_t) - _l*loss_pc*hh_share*np.e**(-_l*_t))**(-eta) * (_t*k_pc_oth*v*pi + _l*_t*loss_pc*hh_share -loss_pc*hh_share)

                    #     if last_integ and ((last_integ < 0 and integ > 0) or (last_integ > 0 and integ < 0)):
                    #     #print('\n Found the Minimum!\n lambda = ',last_lambda,'--> integ = ',last_integ)
                    #     #print('lambda = ',_l,'--> integ = ',integ)

                    #         print 'Sim:',sim,'Tract:',i,'#',tract,'Rate consumption:',(_l+last_lambda)/2
                    #         lambda_opt[count] = (_l+last_lambda)/2
                    #         lambda_cons[count] = (_l+last_lambda)/2
                    #         recon_rate_opt.loc[(sim,tract),'lambda']  = (_l+last_lambda)/2
                    #         #print ('Found the Minimum',recon_rate_opt.loc[(sim,tract),'lambda'],'Physical',rate_physical)
                    #         break

                    #     last_integ = integ
                    #     last_lambda = _l
                    #     _l += 0.01

                    #     if _l > rate_physical:
                        	
                    #     	print 'Sim:',sim,'Tract:',i,'#',tract,'Rate physical:',rate_physical
                    #     	lambda_cons[count] = last_lambda
                    #     	lambda_opt[count] = rate_physical
                    #     	recon_rate_opt.loc[(sim,tract),'lambda']  = rate_physical
                    #     	break
                    #     elif _l > rate_c_min:
                    #     	print 'Sim:',sim,'Tract:',i,'#',tract,'Rate c_min',rate_c_min
                    #     	lambda_cons[count] = last_lambda
                    #     	lambda_opt[count] = rate_c_min
                    #     	recon_rate_opt.loc[(sim,tract),'lambda']  = rate_c_min
                    #     	break


                    #     if _l >= 22:
                    #     	_l = 1093
               
                count += 1
        pickle.dump([recon_rate_opt, lambda_opt, lambda_physical,lambda_constr,recovery_constrain],open( 'optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/"+myCountry+"_recon_rate_opt_w_labour_"+fault_identifier+"_"+str(nsims)+"sims.p", "wb" ))
    return recon_rate_opt, lambda_opt, lambda_physical,lambda_constr,recovery_constrain

def get_physical_recon_rate(pi, rho, myCountry,df_v,fault_identifier, verbose=False,suffix = ''):
    # This function computes the optimal reconstruction rate for each of the census tracts in df_v
    try: 
        nsims = len(df_v.index.get_level_values(0).unique())
        physical_recon_rate= pickle.load(open('optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/"+myCountry+"_physical_recon_rate_"+fault_identifier+'_'+str(nsims)+"sims.p",'rb'))

    except:
#         print('Was not able to load v to hh_reco_rate library from optimization_libs/'+myCountry+'_v_to_reco_rate.p')
#         v_to_reco_rate = {}
        d_recovery_rate = pickle.load(open('optimization_libs/HH_physical_recovery_rate_'+fault_identifier+'_DEC2018'+suffix+'.p','rb'))
        nsims = len(df_v.index.get_level_values(0).unique())
        tracts = df_v.index.get_level_values(1).unique()
        physical_recon_rate = pd.DataFrame(index = df_v.index)
        physical_recon_rate['recon_rate_physical'] = 0

        for sim in np.arange(1,nsims+1):
            sim
            for i,tract in enumerate(tracts):
                print 'Sim:',sim,'Tract:',i,'#', tract
                physical_recon_rate.loc[(sim,tract),'recon_rate_physical'] = d_recovery_rate[(tract,sim)]
        
        pickle.dump(physical_recon_rate,open( 'optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/"+myCountry+"_physical_recon_rate_"+fault_identifier+'_'+str(nsims)+"sims.p", "wb" ))
    return physical_recon_rate


def get_constr_sector_recon_rate(df_v,myCountry,fault_identifier,verbose=False):
    # This function computes the recovery rates constrained by the construction industry
    try: 
        nsims = len(df_v.index.get_level_values(0).unique())
        d_constr_recon_rate= pickle.load(open('optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/"+myCountry+"_constr_sector_recon_rate"+str(nsims)+"sims.p",'rb'))

    except:
        df_HH_recon= pd.read_csv('../inputs/BA/ARIO_v7_1_HH_reconstr_demand_satisfaction_'+fault_identifier+'_DEC2018.csv')
#                
        nsims = len(df_v.index.get_level_values(0).unique())
        tracts = df_v.index.get_level_values(1).unique()
        
        
        t_min = 0.
        t_max = 10.
        n_steps = 52.*10.
        int_dt, dt = np.linspace(t_min,t_max,num=n_steps,endpoint=True,retstep=True)
        df_v['loss_fraction'] = 0
        df_v['threshold'] = df_v.asset_loss_tot*0.05       
        df_v['recon_needs'] = df_v.asset_loss_tot.copy() 
        df_v['t_95'] = -999
        
        d_constr_recon_rate = {}
        
        for sim in np.arange(1,nsims+1):
            scale_factor = max(1,df_v.asset_loss_tot[sim].sum()/(df_HH_recon.iloc[sim-1,1]*1e6))
            print 'Getting the proportional losses for sim:', sim
            print 'Scale_factor:',scale_factor
            df_v.loss_fraction.loc[sim,slice(None)] = df_v.asset_loss_tot.loc[sim,slice(None)]/df_v.asset_loss_tot[sim].sum()
            
            for i,t in enumerate(int_dt):
                df_v.recon_needs.loc[sim,slice(None)] = df_v.recon_needs.loc[sim,slice(None)] - df_v.loss_fraction.loc[sim,slice(None)]*df_HH_recon.iloc[sim-1,i+2]*1e6*scale_factor
                df_v.t_95.loc[(df_v.t_95 == -999) & (df_v.recon_needs <= df_v.threshold)] = t
        
        for sim in np.arange(1,nsims+1):
            for i,tract in enumerate(tracts):
                print 'Sim:',sim,'Tract:',i,'#', tract
                d_constr_recon_rate[(tract,sim)] = np.log(1/0.05)/df_v.t_95.loc[(sim,tract)]
        
        #for sim in np.arange(1,nsims+1):
         #   sim
          #  for i,tract in enumerate(tracts):
           #     print 'Sim:',sim,'Tract:',i,'#', tract
            #    physical_recon_rate.loc[(sim,tract),'recon_rate_physical'] = d_recovery_rate[(tract,sim)]
        
        pickle.dump(d_constr_recon_rate,open( 'optimization_libs/'+myCountry+suffix+'_'+str(nsims)+"sims/"+myCountry+"_constr_sector_recon_rate"+str(nsims)+"sims.p", "wb" ))
    return d_constr_recon_rate
