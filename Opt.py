import math
from scipy.optimize import fmin


def funcSogRpm(par, sog):
    '''
    par:[a,b,c,d]
    0<=sog<=p
    rpm=a sog+b

    sog>p
    rpm=c sog+d

    sog=p
    ap+b=cp+d
    p = (d-b)/(a-c)
    '''
    rpm=0
    if sog >= 0 and sog <= p:
        rpm = par[0] * sog + par[1]
    elif sog > p:
        rpm = par[2] * sog + par[3]
    else:
        rpm = sys.info_float.max
    return rpm
    
def resSogRpm(par=[250,0,250,0], sog, rpm):
    return funcSogRpm(par, sog) - rpm

def fitSogRpm(sog, rpm, par0=[250,0,250,0]):
    return scipy.optimize.leastsq(resSogRpm, par0,args=(sog,rpm))

