import math
from scipy.optimize import fmin


def resSogRpm(par=[250,0,250,0], sog, rpm):
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
    res=0
    if sog >= 0 and sog <= p:
        res=par[0] * sog + par[1] - rpm
    elif sog > p:
        res=par[2] * sog + par[3] - rpm
    else:
        res = sys.info_float.max
    return res

def fitSogRpm(sog, rpm, par0=[250,0,250,0]):
    return scipy.optimize.leastsq(resSogRpm, par0,args=(sog,rpm))

