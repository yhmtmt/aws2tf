import math
import numpy as np
import scipy.optimize
import sys
import matplotlib.pyplot as plt
import pdb

pdb.set_trace()

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
    p=(par[3]-par[1])/(par[0]-par[2])
    rpm=0
    if sog >= 0 and sog <= p:
        rpm = par[0] * sog + par[1]
    elif sog > p:
        rpm = par[2] * sog + par[3]
    else:
        rpm = sys.float_info.max
    return rpm
    
def resSogRpm(par, sog, rpm):
    rpm_p = np.array([funcSogRpm(par, sog[i]) for i in range(rpm.shape[0])])
    res = rpm_p - rpm
    return res

def fitSogRpm(sog, rpm, par0=[250.0,0.0,250.0,0.0]):
    return scipy.optimize.least_squares(resSogRpm, par0, args=(sog,rpm))

if __name__ == '__main__':
    #load sog/rpm data
    data=np.loadtxt("/home/ubuntu/matumoto/aws/plot/sogrpm/sogrpm.csv", delimiter=",")
    data=np.transpose(data)
    res=fitSogRpm(data[0], data[1], par0=[250.0,0.0,150.0,1000.0])
    par=res.x
    print("a=%f b=%f c=%f d=%f res=%f" % (par[0], par[1], par[2], par[3], res.cost))
    
    sog=np.array([float(i) for i in range(0,25)])
    rpm=np.array([funcSogRpm(par, float(i)) for i in range(0,25)])       
    
    plt.figure(figsize=(8,5))
    plt.scatter(data[0], data[1], label="data", alpha=0.3)
    plt.plot(sog, rpm, label="fit", linewidth=10, alpha=0.7)
    plt.xlabel("SOG(kts)")
    plt.ylabel("Rev(RPM)")
    plt.grid(True)
    plt.show()
