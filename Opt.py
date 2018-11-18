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
    if sog <= p:
        rpm = par[0] * sog + par[1]
    elif sog > p:
        rpm = par[2] * sog + par[3]
    else:
        rpm = sys.float_info.max
    return rpm

def cSog(par):
    return (par[3]-par[1])/(par[0]-par[2])

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
# par: X_du, X_u, X_uu, K_l, K_q
# X_du: the added mass
# X_u: linear drag coefficient
# X_uu: quadratic drag coefficient
# K_l: linear thrust coefficient
# K_q: quadratic thrust coefficient
def resXAclPVelPRev(par, m, du, u, n): # <1>, <3> 
    return -m * du - par[0] * du - par[1] * u + par[2] * u * u - par[3] * u * n + par[4] * n * n

def fitXAclPVelPRev(m, du, u, n, par0=[100.0,100,0,100.0,100.0]):
    return scipy.optimize.least_squares(resXAclPVelPRev,
                                        par0, args=(m, du, u, n))

def resXAclPVelNRev(par, m, du, u, n): # <2>
    return -m * du - par[0] * du - par[1] * u + par[2] * u * u - par[3] * u * n - par[4] * n * n

def fitXAclPVelNRev(m, du, u, n, par0=[100.0,100,0,100.0,100.0]):
    return scipy.optimize.least_squares(resXAclPVelNRev,
                                        par0, args=(m, du, u, n))

def resXAclNVelNRev(par, m, du, u, n): # <4>, <6>
    return -m * du - par[0] * du - par[1] * u - par[2] * u * u - par[3] * u * n - par[4] * n * n

def fitXAclNVelNRev(m, du, u, n, par0=[100.0,100,0,100.0,100.0]):
    return scipy.optimize.least_squares(resXAclNVelNRev,
                                        par0, args=(m, du, u, n))

def resXAclNVelPRev(par, m, du, u, n): # <5>
    return -m * du - par[0] * du - par[1] * u - par[2] * u * u - par[3] * u * n + par[4] * n * n

def fitXAclNVelPRev(m, du, u, n, par0=[100.0,100,0,100.0,100.0]):
    return scipy.optimize.least_squares(resXAclNVelPRev,
                                        par0, args=(m, du, u, n))

if __name__ == '__main__':
    import pdb
    pdb.set_trace()

    #load sog/rpm data
    data=np.loadtxt("/mnt/d/aws/proc/sogrpm/un.csv", delimiter=",")
    data=np.transpose(data)
    res=fitSogRpm(data[0], data[1], par0=[500.0,0.0,300.0,1000.0])
    par=res.x
    print("a=%f b=%f c=%f d=%f res=%f" % (par[0], par[1], par[2], par[3], res.cost))

    umax = np.max(data[0])
    umin = np.min(data[0])
    u=np.array([float(i) for i in range(int(umin-0.5),int(umax+0.5))])
    n=np.array([funcSogRpm(par, float(i)) for i in range(int(umin-0.5),int(umax+0.5))])       
    
    plt.scatter(data[0], data[1], label="data", alpha=0.3)
    plt.plot(u, n, label="fit", color='r', linewidth=3)
    plt.xlabel("u(m/s)")
    plt.ylabel("Rev(RPM)")
    plt.grid(True)
    plt.show()
