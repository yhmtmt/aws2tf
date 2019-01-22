import math
import numpy as np
import scipy.optimize
import sys
import matplotlib.pyplot as plt

############## model for eng/rev relationship ################
def qeq2pt(x0,y0,x1,y1,c):
    '''
    return a,b related to a function f(x)=ax^2+bx+c
    pass through (x0,y0) and (x1,y1) and given c.
    '''
    x0x1=x0*x1
    x0x1x1=x0x1*x1
    x0x0x1=x0*x0x1
    base=1/(x0x1x1-x0x0x1)
    cx0=c*x0
    cx0x0=cx0*x0
    x1cmy0=x1*(c-y0)
    x1x1cmy0=x1*x1cmy0
    x0y1=x0*y1
    x0x0y1=x0*x0y1
    a=(-cx0+x1cmy0+x0y1)*base
    b=-(-cx0x0+x1x1cmy0+x0x0y1)*base
    return a,b

def qeq2pt2(x0,y0,x1,y1,a):
    x1mx0=x1-x0
    x0x0=x0*x0
    x1x1=x1*x1
    base = 1/(x1-x0)
    b=(a * (x0x0 - x1x1) - y0 + y1) * base
    c=-(-a*x0*x1x1 + x1*(a*x0x0-y0) + x0*y1) * base
    return b,c

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

    '''
    dd0,dd1=qeq2pt(e0d,r0,epd,rp, par[8])
    dd2 = par[8]    
    dp0,dp1=qeq2pt(epd,rp,ef,rf, par[9])    
    dp2 = par[9]    
    ud0,ud1=qeq2pt(e0,r0,ep,rp, par[10])
    ud2 = par[10]
    up0,up1=qeq2pt(ep,rp,ef,rf, par[11])
    up2 = par[11]
    '''
    dd1,dd2=qeq2pt2(e0d,r0,epd,rp, par[8])
    dd0 = par[8]    
    dp1,dp2=qeq2pt2(epd,rp,ef,rf, par[9])    
    dp0 = par[9]    
    ud1,ud2=qeq2pt2(e0,r0,ep,rp, par[10])
    ud0 = par[10]
    up1,up2=qeq2pt2(ep,rp,ef,rf, par[11])
    up0 = par[11]
   
    if eng < e0d:
        return r0
    if eng >= ef:
        return rf

    eng2 = eng * eng
    
    if is_up:
        if eng < e0:
            return r0
        else:
            if eng < ep:
                return ud0 * eng2 + ud1 * eng + ud2
            else:
                return up0 * eng2 + up1 * eng + up2
    else:
        if eng < epd:
            return dd0 * eng2 + dd1 * eng + dd2
        else:
            return dp0 * eng2 + dp1 * eng + dp2

    return 0

def funcengrevb(par, eng, is_up=True):
    #parameters
    # r0 : idling rev
    # rf : final rev
    # e0d : idling ctrl(down mode)
    # e0 : idling ctrl(up mode)
    # ef : final ctrl
    # d0, d1, d2 : quadratic function in down mode
    # u0, u1, u2 : quadratic function in up mode
    r0=par[0]
    rf=par[1]
    e0d=par[2]
    e0=par[3]
    ef=par[4]
    d0,d1=qeq2pt(e0d,r0,ef,rf,par[5])
    d2=par[5]
    u0,u1=qeq2pt(e0,r0,ef,rf,par[6])
    u2=par[6]
    
    if eng > e0:
        return r0
    if eng < ef:
        return rf

    eng2 = eng * eng
    
    if is_up:
        return u0 * eng2 + u1 * eng + u2
    else:
        if eng > e0d:
            return r0
        else:
            return d0 * eng2 + d1 * eng + d2
    return 0
    
    
def resengrevf(par, eng, rev):
    rev_p = np.array([funcengrevf(par, eng[i][0], eng[i][1] > 0) for i in range(rev.shape[0])])
    res = rev_p - rev
    
    return res

def resengrevb(par, eng, rev):
    rev_p = np.array([funcengrevb(par, eng[i][0], eng[i][1] > 0) for i in range(rev.shape[0])])
    res = rev_p - rev
    return res


def fitengrevf(eng, rev,
               par0=[700,3000,5500,
                     190,195,
                     205,210,
                     230,
                     0,0,
                     0,0]):
    '''
    eng: (eng_ctrl_val,up_or_down) array
    rev: (rpm) array
    '''
    return scipy.optimize.least_squares(resengrevf, par0,
                                        args=(eng, rev),verbose=2, loss='huber')

def fitengrevb(eng, rev,par0=[700,3500,60,65,50,0,0]):
    '''
    eng: (eng_ctrl_val,up_or_down) array

    rev: (rpm) array
    '''
    return scipy.optimize.least_squares(resengrevb, par0,
                                        args=(eng, rev))


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

    par=[700,3000,5500,
         190,195,
         210,215,
         230,
         -1,-2,-3,-4]
    x=[i for i in range(175,230)]
    y=[funcengrevf(par,float(i),True) for i in range(175,230)]
    yd=[funcengrevf(par,float(i),False) for i in range(175,230)]
    plt.plot(x, y, label="up", color='r', linewidth=3)
    plt.plot(x, yd, label="down", color='b', linewidth=3)
    plt.xlabel("u(m/s)")
    plt.ylabel("Rev(RPM)")
    plt.grid(True)
    plt.show()
    
