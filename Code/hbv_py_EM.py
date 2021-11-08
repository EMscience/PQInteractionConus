# Edom Moges 
# Environmental Systems Dynamics Laboratory (ESDL)
# University of California Berkeley
# This script contains the HBV model along with performance metrics and latin hypercube sampler.
# Modified from John_Craven.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def hbv_main(n_days,params,air_temp,prec, dpem,monthly,month,snowThr,cA):

    Tsnow_thresh = snowThr
    ca = cA 
    
    #Initialize arrays for the simiulation
    snow      = np.zeros(n_days)  #
    liq_water = np.zeros(n_days)  #
    pe        = np.zeros(n_days)  #
    soil      = np.zeros(n_days)  #
    ea        = np.zeros(n_days)  #
    dq        = np.zeros(n_days)  #
    s1        = np.zeros(n_days)  #
    s2        = np.zeros(n_days)  #
    q         = np.zeros(n_days)  #
    qm        = np.zeros(n_days)  #
    
    #Set parameters
    d    = params[0]  #
    fc   = params[1]  #
    beta = params[2]  #
    c    = params[3]  #
    k0   = params[4]  #
    l    = params[5]  #
    k1   = params[6]  #
    k2   = params[7]  #
    kp   = params[8]  #
    pwp  = params[9] * fc #

    
    
    for i in np.arange(1,n_days):
        
        if air_temp[i] < Tsnow_thresh:  # snow accumulation
            snow[i] = snow[i-1] + prec[i]
            liq_water[i] = 0.0
            
        # ET computation
            pe[i] = min(2*dpem[month[i]], max(0, (1 + c*(air_temp[i] - monthly[month[i]] ) )*dpem[month[i]] )  )
            
            if soil[i-1] > pwp:
                ea[i] = pe[i]
            else:
                ea[i] = pe[i] * min(1, soil[i-1]/pwp)
                
        # soil w balance
            dq[i] = liq_water[i]*( (soil[i-1]/fc)**beta)
            soil[i] = max(0, soil[i-1] + liq_water[i] - dq[i] - ea[i] )
           
        # Routing
            s1[i] = max(0, s1[i-1] + dq[i] - max(0, s1[i-1]-l)*k0 - s1[i-1]*k1 - s1[i-1]*kp  ) 
            s2[i] = max(0, s2[i-1] + s1[i-1]*kp - s2[i-1]*k2 )
        
        # Runoff           
            q[i] = max(0, (s1[i-1]-l))*k0 + s1[i-1]*k1 + s2[i-1]*k2
            qm[i] = (q[i]*ca*1000.0)/(24.0*3600.0)
            
        else:    # snow melt for high Temp
            
            snow[i] = max(0, snow[i-1] - d*(air_temp[i] - snowThr) )
            liq_water[i] = prec[i] + min(snow[i-1], d*(air_temp[i] - snowThr))
            
        # ET computation 

            pe[i] = min(2*dpem[month[i]], max(0, (1 + c*(air_temp[i] - monthly[month[i]] ) )*dpem[month[i]] )  )
            if soil[i-1] > pwp:
                ea[i] = pe[i]
            else:
                ea[i] = pe[i] * min(1, soil[i-1]/pwp)
            
        # soil water balance
            dq[i] = liq_water[i]*( (soil[i-1]/fc)**beta)
            soil[i] = max(0, soil[i-1] + liq_water[i] - dq[i] - ea[i] )
            
        # Routing
            s1[i] = max(0, s1[i-1] + dq[i] - max(0, s1[i-1]-l)*k0 - s1[i-1]*k1 - s1[i-1]*kp  ) 
            s2[i] = max(0, s2[i-1] + s1[i-1]*kp - s2[i-1]*k2 )
            
        # Runoff           
            q[i] = max(0, (s1[i-1]-l))*k0 + s1[i-1]*k1 + s2[i-1]*k2
            qm[i] = (q[i]*ca*1000.0)/(24.0*3600.0)
            
            
        
       
    return qm, soil, ea


def logNS(o,s):
    """
    Nash Sutcliffe efficiency coefficient in Log space
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient in log space
    """
    eps = 1
    return 1 - sum((np.log(s+eps)-np.log(o+eps))**2)/sum((np.log(o+eps)-np.mean(np.log(o+eps)))**2) 



def NS(o,s):
    """
    Nash Sutcliffe efficiency coefficient
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2) 

def lhssample(n,p): # Latin hypercube sampling
    x = np.random.uniform(size=[n,p])
    for i in range(0,p):
        x[:,i] = (np.argsort(x[:,i])+0.5)/n
    return x

# np.random.seed(50) # seed the random generator
# x = lhssample(100,2)*2+1 #range between 1 and 2 100 parameters in 2D
