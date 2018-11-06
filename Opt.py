import math
import numpy as np
import scipy.optimize
import sys
import matplotlib.pyplot as plt


############## model for sog/rpm relationship ################
# sog->rpm function
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

# residual rpm - funcSogRpm(sog)
def resSogRpm(par, sog, rpm):
    rpm_p = np.array([funcSogRpm(par, sog[i]) for i in range(rpm.shape[0])])
    res = rpm_p - rpm
    return res

# fitter for funcSogRpm using least_squares
def fitSogRpm(sog, rpm, par0=[250.0,0.0,250.0,0.0]):
    return scipy.optimize.least_squares(resSogRpm, par0, args=(sog,rpm))


######### calculation of linear acceleration parameter ############
# -m du - X_du du -X_u u -X_uu |u|u + K_l u n + k_q |n| n = 0
#
# u > 0
#   funcSogRpm(u) > n
#      n > 0 (acceleration) ---<1>
#      n < 0 (inverse thrust deceleration) ---<2>
#   funcSogRpm(u) < n
#      n > 0 (passive deceleration) ---<3>
#      n < 0 (no possibility because n > funcSogRpm(u) > 0)
# u < 0
#   funcSogRpm(u) > n
#      n > 0 (no possibility because n < funcSogRpm(u) < 0)
#      n < 0 (negative acceleration) ---<4>
#   funcSogRpm(u) < n
#      n > 0 (inverse thrust negative deceleration) ---<5>
#      n < 0 (passive negative deceleration) ---<6>
#
# par: m, X_du, X_u, X_uu, K_l, K_q
def resXAclPVelPRev(par, du, u, n): # <1>, <3> 
    return -par[0] * du - par[1] * du - par[2] * u + par[3] * u * u - par[4] * u * n + par[5] * n * n

def resXAclPVelNRev(par, du, u, n): # <2>
    return -par[0] * du - par[1] * du - par[2] * u + par[3] * u * u - par[4] * u * n - par[5] * n * n

def rexXAclNVelNRev(par, du, u, n): # <4>, <6>
    return -par[0] * du - par[1] * du - par[2] * u - par[3] * u * u - par[4] * u * n - par[5] * n * n

def rexXAclNVelPRev(par, du, u, n): # <5>
    return -par[0] * du - par[1] * du - par[2] * u - par[3] * u * u - par[4] * u * n + par[5] * n * n



if __name__ == '__main__':
    import pdb
    pdb.set_trace()

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
