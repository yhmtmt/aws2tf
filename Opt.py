import math
import numpy as np
import scipy.optimize
import sys
import matplotlib.pyplot as plt

############## model for eng/rev relationship ################
def funcengrevf(par, eng, is_up=True):
    # parameters
    # r0 : idling rev
    # rp : planing rev
    # rf : final rev
    # e0d: idling ctrl (down mode)
    # e0 : idling ctrl (up mode)
    # epd: planing ctrl (down mode)
    # ep : planing ctrl (up mode)
    # ef : final ctrl
    # dd0,dd1,dd2, dp0,dp1,dp2 : quadratic function in down mode
    # ud0,ud1,ud2, up0,up1,up2 : quadratic function in up mode
    r0 = par[0]
    rp = par[1]
    rf = par[2]
    e0d = par[3]
    e0 = par[4]
    epd = par[5]
    ep = par[6]
    ef = par[7]
    dd0 = par[8]
    dd1 = par[9]
    dd2 = par[10]
    dp0 = par[11]
    dp1 = par[12]
    dp2 = par[13]
    ud0 = par[14]
    ud1 = par[15]
    ud2 = par[16]
    up0 = par[17]
    up1 = par[18]
    up2 = par[19]
    
    if eng < e0d:
        return r0
    if eng >= ef:
        return rf

    eng2 = eng * eng
    
    if is_up:
        if eng < ep:
            return dd0 * eng2 + dd1 * eng + dd2
        else:
            return dp0 * eng2 + dp1 * eng + dp2
    else:
        if eng < epd:
            return ud0 * eng2 + ud1 * eng + ud2
        else:
            return up0 * eng2 + up1 * eng + up2

    return 0

def resengrevf(par, eng, rev):
    rev_p = np.array([funcengrevf(par, eng[i]) for i in range(n.shape[0])])
    res = n_p - rev
    return res

def fitengrevf(eng, rev, par0=[700,3000,5500,
                               180,185,
                               192.5,195,
                               210,
                               0,184,-32420,0,143,-24500,
                               0,230,-41850,0,167,-29500]):
    return scipy.optimize.least_squares(resengrevf, par0, arg(eng, rev))

############## model for sog/rpm relationship ################
# u->n function
def funcun(par, u):
    '''
    par:[a,b,c,d]
    u < 0
    n = au

    0<=u<=p
    n = bu

    u>p
    n= cu+d

    n=p
    ap+b=cp+d
    p = d/(b-c)

    '''
    p=cu(par)
    n=0
    if u < 0:
        n = par[0] * u
    elif u <= p:
        n = par[1] * u
    else:
        n = par[2] * u + par[3]
        
    return n

def cu(par):
    return par[3]/(par[1]-par[2])

# residual n - funcun(u)
def resun(par, u, n):
    n_p = np.array([funcun(par, u[i]) for i in range(n.shape[0])])
    res = n_p - n
    return res

# fitter for funcun using least_squares
def fitun(u, n, par0=[250.0,0.0,250.0,0.0]):
    return scipy.optimize.least_squares(resun, par0, args=(u,n))


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
    data=np.loadtxt("/home/ubuntu/matumoto/aws/proc/sogrpm/un.csv", delimiter=",")
    data=np.transpose(data)
    _rx=[]
    _ry=[]

    for i in range(len(data[1])):
        if data[1][i] >= 0 and data[0][i] < 0:
            continue
        _rx.append(data[0][i])
        _ry.append(data[1][i])
            
    rx = np.array(_rx)
    ry = np.array(_ry)
    
    res=fitun(rx, ry, par0=[700,500.0,300.0,1000.0])
    par=res.x
    print("a=%f b=%f c=%f d=%f res=%f" % (par[0], par[1], par[2], par[3], res.cost))

    umax = np.max(data[0])
    umin = np.min(data[0])
    u=np.array([float(i) for i in range(int(umin-0.5),int(umax+0.5))])
    n=np.array([funcun(par, float(i)) for i in range(int(umin-0.5),int(umax+0.5))])       
    
    plt.scatter(rx, ry, label="data", alpha=0.3)
    plt.plot(u, n, label="fit", color='r', linewidth=3)
    plt.xlabel("u(m/s)")
    plt.ylabel("Rev(RPM)")
    plt.grid(True)
    plt.show()
