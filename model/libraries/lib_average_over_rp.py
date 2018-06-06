import pandas as pd
import numpy as np

def average_over_rp(df,default_rp='default_rp',protection=None):        
    """Aggregation of the outputs over return periods"""    
    if protection is None:
        protection=pd.Series(0,index=df.index)        

    #just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values('rp'):
        print('default_rp detected, dropping rp')
        return (df.T/protection).T.reset_index('rp',drop=True)
           
    df=df.copy().reset_index('rp')

    #computes frequency of each return period
    return_periods=np.unique(df['rp'].dropna())

    proba = pd.Series(np.diff(np.append(1/return_periods,0)[::-1])[::-1],index=return_periods) #removes 0 from the rps 

    #matches return periods and their frequency
    proba_serie=df['rp'].replace(proba).rename('prob')
    proba_serie1 = pd.concat([df.rp,proba_serie],axis=1)
    #    print(proba_serie.shape)
    #    print(df.rp.shape)
    #    print(protection)
    #removes events below the protection level
    #    proba_serie[protection>df.rp] =0

    #handles cases with multi index and single index (works around pandas limitation)
    idxlevels = list(range(df.index.nlevels))
    if idxlevels==[0]:
        idxlevels =0
#    print(idxlevels)
#    print(get_list_of_index_names(df))
#    print(df.head(10))
    #average weighted by proba
    averaged = df.mul(proba_serie,axis=0).sum(level=idxlevels).drop('rp',axis=1) # frequency times each variables in the columns including rp.
    return averaged,proba_serie1 #here drop rp.
	
	
def average_over_rp1(df,default_rp,protection=None):        
    """Aggregation of the outputs over return periods"""    
    if protection is None:
        protection=pd.Series(0,index=df.index)

    #just drops rp index if df contains default_rp
    if default_rp in df.index.get_level_values('rp'):
        print('default_rp detected, dropping rp')
        return (df.T/protection).T.reset_index('rp',drop=True)
           
    df=df.copy().reset_index('rp')
    protection=protection.copy().reset_index('rp',drop=True)
    
    #computes frequency of each return period
    return_periods=np.unique(df['rp'].dropna())

    proba = pd.Series(np.diff(np.append(1/return_periods,0)[::-1])[::-1],index=return_periods) #removes 0 from the rps 

    #matches return periods and their frequency
    proba_serie=df['rp'].replace(proba)
#    print(proba_serie.shape)
#    print(df.rp.shape)
#    print(protection)
    #removes events below the protection level
    proba_serie[protection>df.rp] =0

    #handles cases with multi index and single index (works around pandas limitation)
    idxlevels = list(range(df.index.nlevels))
    if idxlevels==[0]:
        idxlevels =0

    #average weighted by proba
    averaged = df.mul(proba_serie,axis=0)#.sum(level=idxlevels) # frequency times each variables in the columns including rp.
    return averaged.drop('rp',axis=1) #here drop rp.
