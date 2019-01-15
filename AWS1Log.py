import sys
import os
import re           # for regular expression
import subprocess   # for command execution 
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import requests
import Odet
import cv2
import ldAWS1Log as ldl
import Opt as opt
import pyawssim as sim
from scipy import signal


#pdb.set_trace()
# Ship coordinate
# Axes X: bow
#      Y: right
#      Z: bottom
#
# Camera coordinate
# Axes X: right
#      Y: down
#      Z: front

# Mr: rotation around X (starboard down -> positive roll)
# 1   0  0
# 0  cr -sr
# 0  sr cr

# Mp: rotation around Y (bow up -> positive pitch)
# cp 0 sp
# 0  1  0
# -sp 0  cp

# My: rotation around Z (bow right -> positive yaw)
# cy -sy 0
# sy  cy 0
# 0   0  1

# Raw: AHRS attitude to the world (A=ArApAy)
# Ras: AHRS attitude to the ship (B=BrBpBy)
#    idealy B=I, where I is identity
#    Note that ship attitude to the world is B^tA
# Rcs: Camera attitude relative to ship (C=CrCpCy)
# P: Projection matrix

# Mw: point in world coordinate  
# Ma: point in AHRS coordinate Ma=Ras(Ms-Ta)
# Ta: AHRS position on the ship (ship coordinate)
# Ms: point in ship coordinate  Ms=Ras^tA(Mw-Ts)
# Ts: Ship position in the world (world coordinate)
# Mc: point in camera coordinate Mc=C(Ms-Tc)
# Tc: Camera position on the ship (ship coordinate)
# m: point in image coordinate, sm=PMc (s is scale factor)
# Note: the camera has sensor resolution 1936,1216 but the ROI is 1920,1080 placed at the image center. (-8, -68) should be added to (cx, cy)
# Note: camera parameters are fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
Campar = [ 1.4138767346226757e+003, 1.4139390383098676e+003,
       9.6319441141882567e+002, 5.7628496708437842e+002,
       -9.0039760307679337e-002, 1.1452599777072203e-001, 0., 0.,
       3.4755734835861862e-001, 9.0919984164909914e-002,
       -5.8278957247821707e-002, 4.9679287555607182e-001 ]
Pcam=np.array([[Campar[0],0.0,Campar[2]-8.0],[0.0,Campar[1],Campar[3]-68.0],[0.0,0.0,1.0]])
distCoeffs=np.array(Campar[4:12])
mapx,mapy=cv2.initUndistortRectifyMap(Pcam, distCoeffs, np.eye(3),Pcam, (1920,1080), cv2.CV_32FC1)

rearth=6378136.6

def distHorizon(height):
    ''' 
    calculates distance to the horizon for given height of the observer
    The units of the argument and return value is in meter
    '''
    return 3.57e3*math.sqrt(height)

def genHorizonPoints(height):
    thstep=(10.0/180.0)*math.pi
    D = distHorizon(height) * rearth / (rearth + height)
    Z = height * rearth / (rearth + height)
    angles=np.array([thstep * i for i in range(0,36)])
    c = D * np.cos(angles)
    s = D * np.sin(angles)                    
    return np.stack([c,s,[Z]*36])    

def genRmat(roll, pitch, yaw):
    ''' 
    calculate rotation matrix for given [roll pitch yaw] 
    Matrix is multiplied in the order RrRpRy
    Note that the unit of the arguments should be radian.
    '''
    theta = np.array([roll,pitch,yaw])
    s = np.sin(theta)
    c = np.cos(theta)

    Rr=np.array([[1,0,0],[0,c[0],-s[0]],[0,s[0],c[0]]])
    Rp=np.array([[c[1],0,s[1]],[0,1,0],[-s[1],0,c[1]]])
    Ry=np.array([[c[2],-s[2],0],[s[2],c[2],0],[0,0,1]])
    return np.matmul(Ry,np.matmul(Rp,Rr))

def projPoints(Pcam, R, T, M):
    '''
    Pcam: projection matrix
    R: Rotation matrix
    M: 3D Points
    '''
    sm=np.matmul(Pcam, np.transpose(np.transpose(np.dot(R, M)) - T))
    m=np.divide(sm[0:2],sm[2])
    return np.vstack((m,sm[2]))

#RrRpRy
#(RrRpRy)^t=(RpRy)^tRr^t=Ry^tRp^tRr^t
Raw=np.eye(3)
Ras=np.eye(3)
Rcs=genRmat(0.5*math.pi, 0.0, 0.5*math.pi)

Ta=np.array([2.0,0,0])
Tc=np.array([0,0,2.0])
Ts=np.array([0,0,0])
horizon=genHorizonPoints(Tc[2])
R=np.matmul(np.matmul(Raw, Ras), Rcs).transpose()
#Camera position in the world:T=R^tTc+Ts
T=np.matmul(R.transpose(), Tc) + Ts
#sm=P(RMw-T)
m=projPoints(Pcam, R, T, horizon)

par_engr=['rpm','trim']
str_engr=[["Engine Rev", "RPM"], ["Engine Trim", "None"]]
par_engd=['valt','temp','frate']
str_engd=[["Alternator Output", "V"], ["Engine Temperature", "DegC"], ["Fuel Consumption", "L/h"]]
par_stpos=['lat','lon','alt']
str_stpos=[["Latitude", "Deg"], ["Longitude", "Deg"], ["Altitude", "m"]]
par_stvel=['cog','sog', 'dcog', 'dsog']
str_stvel=[["Course Over Ground","Deg"], ["Speed Over Ground", "kts"], ["Rate of Cource Change", "deg/s"], ["Acceleration", "m/ss"]]
par_stdp=['depth']
str_stdp=[["Depth","m"]]
par_statt=['roll','pitch','yaw','droll', 'dpitch', 'dyaw', 'byaw']
str_statt=[["Roll","Deg"],["Pitch","Deg"],["Yaw","Deg"],["Roll Rate", "Deg/s"],["Pitch Rate", "Deg/s"],["Yaw Rate", "Deg/s"], ["Yaw Bias", "Deg"]]
par_9dof=['mx','my','mz','ax','ay','az','gx','gy','gz']
str_9dof=[["Magnetic Field in X", "None"],["Magnetic Field in Y", "None"],["Magnetic Field in Z", "None"],
            ["Acceleration in X", "None"],  ["Acceleration in Y", "None"],  ["Acceleration in Z", "None"],
            ["Angular Velocity in X", "None"],["Angular Velocity in Y", "None"],["Angular Velocity in Z","None"]]
par_cstat=['meng','seng','rud']
str_cstat=[["Main Engine Throttle Control","None"], ["Sub Engine Throttle Control","None"],["Rudder Control", "None"]]
par_cinst=['acs','meng','seng','rud']
str_cinst=[["Control Source", "None"], ["Main Engine Throttle Control","None"], ["Sub Engine Throttle Control","None"],["Rudder Control", "None"]]

par_model_state=['ueng', 'urud', 'gamma', 'delta', 'sdelta', 'srudder', 'u', 'v', 'r', 'n', 'psi']
str_model_state=[['Engine Control Instruction', 'None'], ['Rudder Control Instruction', 'None'], ['Gear Lever Position', 'None'], ['Throttle Lever Position', 'None'], ['Throttle Lever slack', 'None'], ['Rudder Slack', 'None'], ['Speed in X', 'm/s'],['Speed in Y', 'm/s'],['Yaw rate', 'rad/s'], ['Propeller Rotation', 'rpm'], ['Rudder Angle', 'rad']]

class AWS1Log:
    ''' AWS1 log '''
    def __init__(self):
        self.apinst = []
        self.uiinst = []
        self.ctrlst = []
        self.stpos = []
        self.stvel = []
        self.statt = []
        self.st9dof = []
        self.stdp = []
        self.engr = []
        self.engd = []
        self.strm = {'t':None, 'strm':None}
        self.model_state={}
        self.mdl_eng_ctrl = sim.c_model_engine_ctrl();
        self.mdl_rud_ctrl = sim.c_model_rudder_ctrl();
        self.mdl_3dof_ahd = sim.c_model_3dof();
        self.mdl_3dof_ahp = sim.c_model_3dof();
        self.mdl_3dof_as = sim.c_model_3dof();
        self.mdl_obf_ahd = sim.c_model_outboard_force();
        self.mdl_obf_ahp = sim.c_model_outboard_force();
        self.mdl_obf_as = sim.c_model_outboard_force();
        self.mdl_3dof_ahd.alloc_param(0)
        self.mdl_3dof_ahp.alloc_param(1)
        self.mdl_3dof_as.alloc_param(2)
        self.mdl_obf_ahd.alloc_param(0)
        self.mdl_obf_ahp.alloc_param(1)
        self.mdl_obf_as.alloc_param(2)
        
        self.mdl_params={}
        self.yaw_bias=0
        self.yaw_bias_max=0
        self.yaw_bias_min=0
        self.yaw_bias_dev=0
        
    def update_model_param(self):
        self.mdl_3dof_ahd.init()
        self.mdl_3dof_ahp.init()
        self.mdl_3dof_as.init()
        self.mdl_obf_ahd.init()
        self.mdl_obf_ahp.init()
        self.mdl_obf_as.init()        
        
    def load_model_param(self, path_model_param):
        with open(path_model_param) as file:
            self.mdl_params={}
            for line in file:
                line = line.strip();
                if len(line) == 0:
                    continue;

                parval=line
                for i in range(len(line)):
                    if(line[i] == '#'):                        
                        parval,exp=line.split('#')
                
                if len(parval) == 0:
                    continue;

                par=''
                val=''
                for i in range(len(line)):
                    if(line[i] == '='):
                        par,val = parval.split('=')                        
                        self.mdl_params[par]=float(val)
                    
            self.mdl_eng_ctrl.set_params(self.mdl_params)
            self.mdl_rud_ctrl.set_params(self.mdl_params)
            self.mdl_3dof_ahd.set_params(self.mdl_params)
            self.mdl_3dof_ahp.set_params(self.mdl_params)
            self.mdl_3dof_as.set_params(self.mdl_params)
            self.mdl_obf_ahd.set_params(self.mdl_params)
            self.mdl_obf_ahp.set_params(self.mdl_params)
            self.mdl_obf_as.set_params(self.mdl_params)
            self.update_model_param()
            
    def save_model_param(self, path_model_param):
        with open(path_model_param,mode='w') as file:
            for par,val in self.mdl_params.items():
                txt="%s=%0.12f\n" % (par, val)
                file.write(txt)
    
    def load(self, path_aws1_log, log_time=-1, dt=0.1):
        data,log_time=ldl.loadLog(path_aws1_log, log_time)
        self.apinst = data['apinst']
        self.uiinst = data['uiinst']
        self.ctrlst = data['ctrlst']
        self.stpos = data['stpos']
        self.stvel = data['stvel']
        self.statt = data['statt']
        self.st9dof = data['st9dof']
        self.stdp = data['stdp']
        self.engr = data['engr']
        self.engd = data['engd']
        self.strm = data['strm']
        
        #calculate model state
        # velocity in body fixed coordinate (u, v, r)
        # rudder angle in rad
        # gear position
        # propeller rev rpm x gratio x gear {-1,0,1}
        lapinst,tapinst = ldl.getListAndTime(par_cinst, self.apinst)
        luiinst,tuiinst = ldl.getListAndTime(par_cinst, self.uiinst)      
        lengr,tengr = ldl.getListAndTime(par_engr, self.engr)        
        lstatt,tstatt = ldl.getListAndTime(par_statt, self.statt)
        lstvel,tstvel = ldl.getListAndTime(par_stvel, self.stvel)

        terr = ldl.getErrorAtt(tstatt, lstatt)
        
        # calculate yaw bias
        self.yaw_bias, self.yaw_bias_max, self.yaw_bias_min, self.yaw_bias_dev = ldl.estimateYawBias(0, sys.float_info.max, tstvel, lstvel, tstatt, lstatt, terr)
        
        ts=0.0
        def getTimeEnd(t):
            if len(t) == 0:
                return 0
            return t[-1]
        
        te=max([getTimeEnd(tapinst),getTimeEnd(tuiinst),getTimeEnd(tengr),getTimeEnd(tstatt),getTimeEnd(tstvel)])
        tcur=ts
        iapinst = ldl.seekLogTime(tapinst, ts)
        iuiinst = ldl.seekLogTime(tuiinst, ts)
        istvel = ldl.seekLogTime(tstvel, ts)
        istatt = ldl.seekLogTime(tstatt, ts)
        iengr = ldl.seekLogTime(tengr, ts)

        gamma = 0.0
        delta = 0.0
        sdelta= 0.0
        srudder = 0.0
        psi=0.0
        len_seq = int(te/dt)
        vecs=np.zeros((len_seq, len(par_model_state)+1))
        radian = math.pi / 180.0
        mps = 1852.0 / 3600.0
        for i in range(len_seq):
            iapinst = ldl.seekNextDataIndex(tcur, iapinst, tapinst)
            vapinst = ldl.itpltDataVec(lapinst, tcur, tapinst, iapinst)
            iuiinst = ldl.seekNextDataIndex(tcur, iuiinst, tuiinst)
            vuiinst = ldl.itpltDataVec(luiinst, tcur, tuiinst, iuiinst)
            istvel = ldl.seekNextDataIndex(tcur, istvel, tstvel)
            vstvel = ldl.itpltDataVec(lstvel, tcur, tstvel, istvel, [True, False, False, False])
            istatt = ldl.seekNextDataIndex(tcur, istatt, tstatt)
            vstatt = ldl.itpltDataVec(lstatt, tcur, tstatt, istatt, [False, False, True, False, False, False, False])
            iengr = ldl.seekNextDataIndex(tcur, iengr, tengr)
            vengr = ldl.itpltDataVec(lengr, tcur, tengr, iengr)

            sog = vstvel[1] * mps
            vstatt[6] = self.yaw_bias
            beta = (vstvel[0] - (vstatt[2] + vstatt[6]))
            if(beta > 180):
                beta -= 360
            elif (beta < -180):
                beta += 360
            beta *= radian
            u = sog * math.cos(beta)
            v = sog * math.sin(beta)
            r = vstatt[5]
            n = (vengr[0] if (gamma >= 1.0) else (-vengr[0] if (gamma <= -1.0) else 0))
            ueng=vuiinst[1] if vuiinst[0]==0 else vapinst[1]
            urud=vuiinst[3] if vuiinst[0]==0 else vapinst[3]
            
            vecs[i][0] = tcur
            vecs[i][1] = ueng;
            vecs[i][2] = urud;
            vecs[i][3] = gamma;
            vecs[i][4] = delta;
            vecs[i][5] = sdelta;
            vecs[i][6] = srudder;
            vecs[i][7] = u;
            vecs[i][8] = v;
            vecs[i][9] = r;
            vecs[i][10] = n;
            vecs[i][11] = psi;
            gamma_new, delta_new, sdelta_new = self.mdl_eng_ctrl.update(int(ueng), gamma, delta, sdelta, dt)
            psi_new, srudder_new = self.mdl_rud_ctrl.update(int(urud), psi, srudder, dt)
            gamma=gamma_new
            delta=delta_new
            sdelta=sdelta_new
            psi=psi_new
            srudder=srudder_new
            
            tcur+=dt

        vecs=np.transpose(vecs)
        self.model_state['t'] = vecs[0]
        for i in range(len(par_model_state)):
            self.model_state[par_model_state[i]] = vecs[i+1]
            
        return log_time

    def getRelSogRpmAcl(self, ts=0.0, te=sys.float_info.max):
        lstatt,tstatt = ldl.getListAndTime(par_statt, self.statt)
        terr=ldl.getErrorAtt(tstatt, lstatt)
        lstvel,tstvel = ldl.getListAndTime(par_stvel, self.stvel)
        lctrlst,tctrlst = ldl.getListAndTime(par_cstat, self.ctrlst)
        lengr,tengr = ldl.getListAndTime(par_engr, self.engr)
        
        return ldl.getRelSogRpmAcl(ts, te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr)

    def getRelundu(self, ts=0.0, te=sys.float_info.max):
        terr=ldl.getErrorAtt(tstatt, lstatt)        
        lmdl,tmdl=ldl.getListAndTime(par_model_state, self.model_state)
        return ldl.getRelundu(ts, te, tmdl, lmdl, terr)
    
    def play(self, ts, te, dt=0.1):
        # seek head for all data section
        lapinst,tapinst = ldl.getListAndTime(par_cinst, self.apinst)
        luiinst,tuiinst = ldl.getListAndTime(par_cinst, self.uiinst)
        lctrlst,tctrlst = ldl.getListAndTime(par_cstat, self.ctrlst)
        lstpos,tstpos = ldl.getListAndTime(par_stpos, self.stpos)
        lstvel,tstvel = ldl.getListAndTime(par_stvel, self.stvel)
        lstatt,tstatt = ldl.getListAndTime(par_statt, self.statt)
        lst9dof,tst9dof = ldl.getListAndTime(par_9dof, self.st9dof)
        lstdp,tstdp = ldl.getListAndTime(par_stdp, self.stdp)
        lengr,tengr = ldl.getListAndTime(par_engr, self.engr)
        lengd,tengd = ldl.getListAndTime(par_engd, self.engd)
        
        iapinst = ldl.seekLogTime(tapinst, ts)
        iuiinst = ldl.seekLogTime(tuiinst, ts)
        ictrlst = ldl.seekLogTime(tctrlst, ts)
        istpos = ldl.seekLogTime(tstpos, ts)
        istvel = ldl.seekLogTime(tstvel, ts)
        istatt = ldl.seekLogTime(tstatt, ts)
        i9dof = ldl.seekLogTime(tst9dof, ts)
        istdp = ldl.seekLogTime(tstdp, ts)
        iengr = ldl.seekLogTime(tengr, ts)
        iengd = ldl.seekLogTime(tengd, ts) 
        tstrm = self.strm['t']
        istrm = ldl.seekLogTime(tstrm, ts)
        strm = self.strm['strm']
        ifrm = int(strm.get(cv2.CAP_PROP_POS_FRAMES))        
        ret = strm.set(cv2.CAP_PROP_POS_FRAMES, max(0,istrm[1] - 1))        
        ret,frm = strm.read()

        while ifrm < istrm[1]:
            ret,frm = strm.read()
            ifrm += 1
        
        ldl.printTimeHead("apinst", iapinst, tapinst)
        ldl.printTimeHead("uiinst", iuiinst, tuiinst)
        ldl.printTimeHead("ctrlst", ictrlst, tctrlst)
        ldl.printTimeHead("stpos", istpos, tstpos)
        ldl.printTimeHead("stvel", istvel, tstvel)
        ldl.printTimeHead("statt", istatt, tstatt)
        ldl.printTimeHead("st9dof", i9dof, tst9dof)
        ldl.printTimeHead("stdp", istdp, tstdp)
        ldl.printTimeHead("engr", iengr, tengr)
        ldl.printTimeHead("engd", iengd, tengd)
        ldl.printTimeHead("strm", istrm, tstrm)

        tcur = ts

        # Setting up object detector
        odt = Odet.Odet()
        
        bodet=False
        budist=False
        
        while tcur < te:
            print ("Time %fsec" % tcur)
            iapinst = ldl.seekNextDataIndex(tcur, iapinst, tapinst)
            vapinst = ldl.itpltDataVec(lapinst, tcur, tapinst, iapinst)
            ldl.printDataVec("apinst", par_cinst, vapinst)

            iuiinst = ldl.seekNextDataIndex(tcur, iuiinst, tuiinst)
            vuiinst = ldl.itpltDataVec(luiinst, tcur, tuiinst, iuiinst)
            ldl.printDataVec("uiinst", par_cinst, vuiinst)

            ictrlst = ldl.seekNextDataIndex(tcur, ictrlst, tctrlst)
            vctrlst = ldl.itpltDataVec(lctrlst, tcur, tctrlst, ictrlst)
            ldl.printDataVec("ctrlst", par_cstat, vctrlst)

            istpos = ldl.seekNextDataIndex(tcur, istpos, tstpos)
            vstpos = ldl.itpltDataVec(lstpos, tcur, tstpos, istpos)
            ldl.printDataVec("stpos", par_stpos, vstpos)

            istvel = ldl.seekNextDataIndex(tcur, istvel, tstvel)
            vstvel = ldl.itpltDataVec(lstvel, tcur, tstvel, istvel, [True, False, False, False])
            ldl.printDataVec("stvel", par_stvel, vstvel)

            istatt = ldl.seekNextDataIndex(tcur, istatt, tstatt)
            vstatt = ldl.itpltDataVec(lstatt, tcur, tstatt, istatt, [False, False, True, False, False, False, False])
            ldl.printDataVec("statt", par_statt, vstatt)

            i9dof = ldl.seekNextDataIndex(tcur, i9dof, tst9dof)
            vst9dof = ldl.itpltDataVec(lst9dof, tcur, tst9dof, i9dof)
            ldl.printDataVec("st9dof", par_9dof, vst9dof)

            istdp = ldl.seekNextDataIndex(tcur, istdp, tstdp)
            vstdp = ldl.itpltDataVec(lstdp, tcur, tstdp, istdp)
            ldl.printDataVec("stdp", par_stdp, vstdp)

            iengr = ldl.seekNextDataIndex(tcur, iengr, tengr)
            vengr = ldl.itpltDataVec(lengr, tcur, tengr, iengr)
            ldl.printDataVec("engr", par_engr, vengr)

            iengd = ldl.seekNextDataIndex(tcur, iengd, tengd)
            vengd = ldl.itpltDataVec(lengd, tcur, tengd, iengd)
            ldl.printDataVec("engr", par_engd, vengd)
        
            istrm = ldl.seekNextDataIndex(tcur, istrm, tstrm)
            ifrm = int(strm.get(cv2.CAP_PROP_POS_FRAMES))
            bfrmNew=False
            if ifrm < istrm[1]:                
                while ifrm < istrm[1]:
                    ret,frm = strm.read()
                    bfrmNew=True
                    ifrm += 1

            if(bfrmNew):
                if(budist):
                    frm_ud=cv2.remap(frm,mapx,mapy,cv2.INTER_LINEAR)
                else:
                    frm_ud = frm

                if(bodet):
                    odt.proc(frm_ud)
                    
                font=cv2.FONT_HERSHEY_SIMPLEX
                txt="Time %5.2f Frame %06d" % (tcur, ifrm)

                if(budist):                
                    txt+=" Undist"
                    
                cv2.putText(frm_ud, txt, (0, 30), font, 1, (0,255,0), 2, cv2.LINE_AA)
                txt="RUD %03.0f ENG %03.0f REV %04.0f SOG %03.1f" % (vuiinst[3], vuiinst[1],vengr[0], vstvel[1])
                cv2.putText(frm_ud, txt, (0, 60), font, 1, (0,255,0), 2, cv2.LINE_AA)
                # draw horizon
                fac=math.pi/180.0
                Raw=genRmat(vstatt[0]*fac,vstatt[1]*fac,vstatt[2]*fac)
                R=np.matmul(np.matmul(Raw, Ras), Rcs).transpose()
                T=np.matmul(R.transpose(), Tc) + Ts
               
                m=projPoints(Pcam, R, T, horizon)
                imin=imax=0
                for i in range(36):
                    pt0=[int(m[0][i]), int(m[1][i])]
                    
                    if m[2][i] > 0 and pt0[0] > 0 and pt0[0] < 1920 and pt0[1] > 0 and pt0[1] < 1080:
                        cv2.line(frm_ud,(pt0[0],pt0[1]), (pt0[0],pt0[1]-10), (0, 255, 0), 3)
                        txt="%02d" % i
                        cv2.putText(frm_ud, txt, (pt0[0], pt0[1] - 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('frame', frm_ud)
                key = cv2.waitKey(int(dt*1000))
                if key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key == ord('u'):
                    budist=not(budist)
                elif key == ord('d'):
                    bodet=not(bodet)
            tcur += dt

    def proc(self, ts=0, te=sys.float_info.max, path='./'):
        lapinst,tapinst = ldl.getListAndTime(par_cinst, self.apinst)
        luiinst,tuiinst = ldl.getListAndTime(par_cinst, self.uiinst)
        lctrlst,tctrlst = ldl.getListAndTime(par_cstat, self.ctrlst)
        lstpos,tstpos = ldl.getListAndTime(par_stpos, self.stpos)
        lstvel,tstvel = ldl.getListAndTime(par_stvel, self.stvel)
        lstatt,tstatt = ldl.getListAndTime(par_statt, self.statt)
        lst9dof,tst9dof = ldl.getListAndTime(par_9dof, self.st9dof)
        lstdp,tstdp = ldl.getListAndTime(par_stdp, self.stdp)
        lengr,tengr = ldl.getListAndTime(par_engr, self.engr)
        lengd,tengd = ldl.getListAndTime(par_engd, self.engd)
        lmdl,tmdl = ldl.getListAndTime(par_model_state, self.model_state)       
        
        iapinst = ldl.seekLogTime(tapinst, ts)
        iapinstf = ldl.seekLogTime(tapinst, te)
        iuiinst = ldl.seekLogTime(tuiinst, ts)
        iuiinstf = ldl.seekLogTime(tuiinst, te)
        ictrlst = ldl.seekLogTime(tctrlst, ts)
        ictrlstf = ldl.seekLogTime(tctrlst, te)
        istpos = ldl.seekLogTime(tstpos, ts)
        istposf = ldl.seekLogTime(tstpos, te)
        istvel = ldl.seekLogTime(tstvel, ts)
        istvelf = ldl.seekLogTime(tstvel, te)
        istatt = ldl.seekLogTime(tstatt, ts)
        istattf = ldl.seekLogTime(tstatt, te)
        i9dof = ldl.seekLogTime(tst9dof, ts)
        i9doff = ldl.seekLogTime(tst9dof, te)
        istdp = ldl.seekLogTime(tstdp, ts)
        istdpf = ldl.seekLogTime(tstdp, te)
        iengr = ldl.seekLogTime(tengr, ts)
        iengrf = ldl.seekLogTime(tengr, te)
        iengd = ldl.seekLogTime(tengd, ts) 
        iengdf = ldl.seekLogTime(tengd, te) 
        tstrm = self.strm['t']
        istrm = ldl.seekLogTime(tstrm, ts)
        istrmf = ldl.seekLogTime(tstrm, te)        
        strm = self.strm['strm']   
        ifrm = int(strm.get(cv2.CAP_PROP_POS_FRAMES))
        ret = strm.set(cv2.CAP_PROP_POS_FRAMES, max(0,istrm[1] - 1))
        ret,frm = strm.read()
        imdl = ldl.seekLogTime(tmdl, ts)
        imdlf = ldl.seekLogTime(tmdl, te)
        
        while ifrm < istrm[1]:
            ifrm += 1
            ret,frm = strm.read()


        if not os.path.exists(path):
            os.mkdir(path)

        # calculate and save statistics
        csvname=path + "/stat.csv"
        with open(csvname, mode='w') as statcsv:
            str="name, max, min, avg, dev\n"
            statcsv.write(str)
            
            for key in par_cinst:
                ldl.saveStat(statcsv, key, self.apinst[key])

            for key in par_cinst:
                ldl.saveStat(statcsv, key, self.uiinst[key])

            for key in par_cstat:
                ldl.saveStat(statcsv, key, self.ctrlst[key])

            for key in par_stpos:
                ldl.saveStat(statcsv, key, self.stpos[key])

            for key in par_stvel:
                ldl.saveStat(statcsv, key, self.stvel[key])

            for key in par_statt:
                ldl.saveStat(statcsv, key, self.statt[key])

            for key in par_9dof:
                ldl.saveStat(statcsv, key, self.st9dof[key])

            for key in par_stdp:
                ldl.saveStat(statcsv, key, self.stdp[key])

            for key in par_engr:
                ldl.saveStat(statcsv, key, self.engr[key])

            for key in par_engd:
                ldl.saveStat(statcsv, key, self.engd[key])

            for key in par_model_state:
                ldl.saveStat(statcsv, key, self.model_state[key])
                
            ftotal = ldl.integrateData(tengd, self.engd['frate'])
            ftotal /= 3600.0
            ldl.saveStatGiven(statcsv, "ftotal", ftotal, ftotal, ftotal, ftotal)
            ldl.saveStatGiven(statcsv, "yaw_bias", self.yaw_bias_max, self.yaw_bias_min, self.yaw_bias, self.yaw_bias_dev)
            
        # plot data
        ldl.plotDataSection(path, "apinst", par_cinst, str_cinst,
                            lapinst, tapinst, iapinst[0], iapinstf[1])
        ldl.plotDataSection(path, "uiinst", par_cinst, str_cinst,
                            luiinst, tuiinst, iuiinst[0], iuiinstf[1])
        ldl.plotDataSection(path, "ctrlst", par_cstat, str_cstat,
                            lctrlst, tctrlst, ictrlst[0], ictrlstf[1])
        ldl.plotDataSection(path, "stpos", par_stpos, str_stpos,
                            lstpos, tstpos, istpos[0], istposf[1])
        ldl.plotDataSection(path, "stvel", par_stvel, str_stvel,
                            lstvel, tstvel, istvel[0], istvelf[1])
        ldl.plotDataSection(path, "statt", par_statt, str_statt,
                            lstatt, tstatt, istatt[0], istattf[1])
        ldl.plotDataSection(path, "st9doff", par_9dof, str_9dof,
                            lst9dof, tst9dof, i9dof[0], i9doff[1])
        ldl.plotDataSection(path, "stdp", par_stdp, str_stdp,
                            lstdp, tstdp, istdp[0], istdpf[1])
        ldl.plotDataSection(path, "engr", par_engr, str_engr,
                            lengr, tengr, iengr[0], iengrf[1])
        ldl.plotDataSection(path, "engd", par_engd, str_engd,
                            lengd, tengd, iengd[0], iengdf[1])
        ldl.plotDataSection(path, "model_state", par_model_state, str_model_state,
                            lmdl, tmdl, imdl[0], imdlf[1])

        minlat = min(lstpos[0][istpos[0]:istposf[1]]) 
        maxlat = max(lstpos[0][istpos[0]:istposf[1]]) 
        minlon = min(lstpos[1][istpos[0]:istposf[1]]) 
        maxlon = max(lstpos[1][istpos[0]:istposf[1]]) 
        midlat = (maxlat + minlat) * 0.5
        midlon = (maxlon + minlon) * 0.5
        rlat = maxlat - minlat
        rlon = maxlon - minlon
        minlat = midlat - rlat
        maxlat = midlat + rlat
        minlon = midlon - rlon
        maxlon = midlon + rlon

        terr = ldl.getErrorAtt(tstatt, lstatt)
        
        rx,ry=ldl.getRelMengRpm(ts,te, tctrlst, lctrlst, tengr, lengr, terr)
        ldl.plotDataRelation(path, "", "meng", "rpm", str_cstat[0], str_engr[0],
                             rx, ry)
       
        rx,ry=ldl.getRelFieldSogCog(ts,te,tstvel,lstvel,tctrlst,lctrlst, terr)
        ldl.plotDataRelation(path, "", par_stvel[1], par_stvel[0], str_stvel[1],
                             str_stvel[0], rx, ry)
               
        rx,ry=ldl.getRelun(ts,te,tmdl,lmdl,terr)
        ldl.plotun(path, str_model_state[6], str_model_state[0], rx, ry)
               
        rx,ry,rz=ldl.getRelundu(ts, te, tmdl, lmdl, terr)
        ldl.plotundu(path, par_model_state[6], par_model_state[9], "du",
                     str_model_state[6], str_model_state[9], ["Acceleration in X", "m/ss"], rx, ry, rz)
        
        # detect and save turns
        turns= ldl.getStableTurn(ts,te,
                                 tstvel, lstvel,
                                 tstatt, lstatt,
                                 tctrlst, lctrlst,
                                 tmdl, lmdl, terr)
        ldl.saveStableTurn(path, turns)

def solve3DoFModelEx(path_model_param, path_log, logs, path_result, force=False):
    #check logs processed
    log = AWS1Log()
    for log_time in logs:
        if not os.path.exists(path_result+"/"+log_time):          
            log.load(path_log, int(log_time))
            log.proc(0, sys.float_info.max, path_result+"/"+log_time)
    # load initial model parameter
    log.load_model_param(path_model_param)

    # check sogrpm file existence
    path_sogrpm = path_result + "/sogrpm"
    if not os.path.exists(path_sogrpm):
        print("sogrpm result is not found. Now sogrpm is gonna run") 
        procOpSogRpm(path_log, logs, path_result, force=False)

    # loadun parameter
    parun = ldl.loadParun(path_sogrpm)
    
    # calculate mode threshold
    # 0 > u astern (mode 2)
    # 0 < u < uthpd ahead displacement (mode 0)
    # uthpd < u ahead plane (mode 1)
    uthpd = opt.cu(parun)    
    if u < 0:
        m_as = log.mdl_params["m2"]
        rx_as = log.mdl_params["xr2"]
        ry_as = log.mdl_params["yr2"]
    elif u < uthpd:
        m_ad = log.mdl_params["m0"]
        rx_ad = log.mdl_params["xr0"]
        ry_ad = log.mdl_params["yr0"]             
    else:             
        m_ap = log.mdl_params["m1"]
        rx_ap = log.mdl_params["xr1"]
        ry_ap = log.mdl_params["yr1"]

    nsmpl=32
    smpl_ad=None
    smpl_ap=None
    smpl_as=None
    
    # sampling nsmpl vectors for each log
    for log_time in logs:
        # load all u,v,r,phi,n
        t,u,v,r,phi,n=ldl.load_u_v_r_phi_n(path_result)
        
        # apply savgol on u,v,r,n
        u=signal.savgol_filter(u, 9, 3, mode="mirror")
        v=signal.savgol_filter(v, 9, 3, mode="mirror")
        r=signal.savgol_filter(r, 9, 3, mode="mirror")
        n=signal.savgol_filter(n, 9, 3, mode="mirror")
        
        # calculate du,dv,dr
        du=ldl.diffDataVec(t, u)
        dv=ldl.diffDataVec(t, v)
        dr=ldl.diffDataVec(t, r)
        
        # select data range with non zero n
        # split data into three mode        
        tahd = findInRangeTimeRanges(t, n, vmax=6000, vmin=600)
        tpln = findInRangeTimeRanges(t, u, vmax=30, vmin=uthpd)
        tdsp = complementTimeRange(t, tpln)
        tahddsp = intersectTimeRanges(tahd, tdsp) # mode 0
        tahdpln = intersectTimeRanges(tahd, tpln) # mode 1
        tast = findInRangeTimeRanges(t, n, vmax=-600, vmin=6000) # mode 2
        
        uahddsp = ldl.getTimeRangeVecs(t, u, tahddsp)
        duahddsp = ldl.getTimeRangeVecs(t, du, tahddsp)        
        vahddsp = ldl.getTimeRangeVecs(t, v, tahddsp)
        dvahddsp = ldl.getTimeRangeVecs(t, dv, tahddsp)
        rahddsp = ldl.getTimeRangeVecs(t, r, tahddsp)
        drahddsp = ldl.getTimeRangeVecs(t, dr, tahddsp)        
        nahddsp = ldl.getTimeRangeVecs(t, n, tahddsp)
        phiahddsp = ldl.getTimeRangeVecs(t, phi, tahddsp)

        uahdpln = ldl.getTimeRangeVecs(t, u, tahdpln)
        duahdpln = ldl.getTimeRangeVecs(t, du, tahdpln)        
        vahdpln = ldl.getTimeRangeVecs(t, v, tahdpln)
        dvahdpln = ldl.getTimeRangeVecs(t, dv, tahdpln)
        rahdpln = ldl.getTimeRangeVecs(t, r, tahdpln)
        drahdpln = ldl.getTimeRangeVecs(t, dr, tahdpln)        
        nahdpln = ldl.getTimeRangeVecs(t, n, tahdpln)
        phiahdpln = ldl.getTimeRangeVecs(t, phi, tahdpln)
        
        uast = ldl.getTimeRangeVecs(t, u, tast)
        duast = ldl.getTimeRangeVecs(t, du, tast)        
        vast = ldl.getTimeRangeVecs(t, v, tast)
        dvast = ldl.getTimeRangeVecs(t, dv, tast)
        rast = ldl.getTimeRangeVecs(t, r, tast)
        drast = ldl.getTimeRangeVecs(t, dr, tast)        
        nast = ldl.getTimeRangeVecs(t, n, tast)
        phiast = ldl.getTimeRangeVecs(t, phi, tast)

        smpl = ldl.sampleMaxDistPoints(nsmpl, [uast,duast,vast,dvast,rast,drast,phiast,nast])
        if(smpl_as == None):
            smpl_as = smpl
        else:
            smpl_as = np.concatenate(smpl_as, smpl)
        
        smpl = ldl.sampleMaxDistPoints(nsmpl, [uahdpln,duahdpln,vahdpln,dvahdpln,rahdpln,drahdpln,phiahdpln,nahdpln])
        if(smpl_ap == None):
            smpl_ap = smpl
        else:
            smpl_ap = np.concatenate(smpl_ap, smpl)
        
        smpl = ldl.sampleMaxDistPoints(nsmpl, [uahddsp,duahddsp,vahddsp,dvahddsp,rahddsp,drahddsp,phiahddsp,nahddsp])
        if(smpl_ad == None):
            smpl_ad = smpl
        else:
            smpl_ad = np.concatenate(smpl_ad, smpl)

    parstr = ["xg", "yg", "ma_xu", "ma_yv", "ma_nv", "ma_nr", "dl_xu", "dl_yv", "dl_yr", "dl_nv", "dl_nr", "dq_xu", "dq_yv", "dq_yr", "dq_nv", "dq_nr", "CL", "CD", "CTL", "CTQ"]


    def reorder_mdl_param(idx, parxy, parn):
        par={}
        stridx="%d" % idx
        iparxy=0
        for i in range(len(parstr)):
            str=parstr[i]+stridx
            if parstr[i] != 'ma_nr':                
                par[str]=parxy[iparxy]
            else:
                par[str]=parn
        return par
                    
    def set_mdl_param(par):
        for j in range(3):
            stridx="%d" % j            
            for i in range(len(parstr)):
                key=parstr[i]+stridx
                log.mdl_params[key] = par[key]

    def print_mdl_param_update(par):
        for j in range(3):
            stridx="%d" % j            
            for i in range(len(par)):
                key=parstr[i]+stridx
                print(parstr[i]+stridx+(" %0.12f->%0.12f" % (log.mdl_params[key], par[key])))
            
    def is_rank_full(s,eps=1.0e-6):
        for i in range(s.shape[0]):
            if(abs(s[i]) < eps):
                return False
        return True

    def psinv(U, s, V):
        return np.dot(np.dot(np.transpose(V),np.pad(np.diag(1/s), [(0,V.shape[1]-s.shape[0]),(0,U.shape[0]-s.shape[0])],'constant')),np.transpose(U))
            
    eqxy_as=[]
    resxy_as=[]
    for ismpl in range(smpl_as.shape[0]):        
        eqxy,resxy=get3DoFEqXY(smpl_as[ismpl][0], smpl_as[ismpl][1],
                            smpl_as[ismpl][2], smpl_as[ismpl][3],
                            smpl_as[ismpl][4], smpl_as[ismpl][5],
                            smpl_as[ismpl][6], smpl_as[ismpl][7],
                            m_as, rx_as, ry_as)
        eqxy_as.append(eqxy[0])
        eqxy_as.append(eqxy[1])
        resxy_as.append(resxy[0])
        resxy_as.append(resxy[1])
        
    eqxy_as = np.array(eqxy_as)
    resxy_as = np.array(resxy_as)
    Uas,sas,Vas=np.linalg.svd(eqxy_as, full_matrices=True)
    if(is_rank_full(sas)):
        eqxy_as_inv=psinv(Uas,sas,Vas)
        paras=np.dot(eqxy_as_inv, rsxy_as)
        Ndr_as=0.0
        for ismpl in range(smpl_as.shape[0]):
            Ndr_as+=get3DoFEqN(smpl_as[ismpl][0], smpl_as[ismpl][1],
                            smpl_as[ismpl][2], smpl_as[ismpl][3],
                            smpl_as[ismpl][4], smpl_as[ismpl][5],
                            smpl_as[ismpl][6], smpl_as[ismpl][7],
                            m_as, rx_as, ry_as, paras)
        Ndr_as/=smpl_as.shape[0]        

        paras=reorder_mdl_param(2, paras, Ndr_as)
        
    eqxy_ap=[]
    resxy_ap=[]
    for ismpl in range(smpl_ap.shape[0]):
        eqxy,resxy=get3DoFEqXY(smpl_ap[ismpl][0], smpl_ap[ismpl][1],
                            smpl_ap[ismpl][2], smpl_ap[ismpl][3],
                            smpl_ap[ismpl][4], smpl_ap[ismpl][5],
                            smpl_ap[ismpl][6], smpl_ap[ismpl][7],
                               m_ap, rx_ap, ry_ap)
        eqxy_ap.append(eqxy[0])
        eqxy_ap.append(eqxy[1])
        resxy_ap.append(resxy[0])
        resxy_ap.append(resxy[1])
    eqxy_ap = np.array(eqxy_ap)
    resxy_ap = np.array(resxy_ap)
    
    Uap,sap,Vap=np.linalg.svd(eqxy_ap, full_matrices=True)
    if(is_rank_full(sap)):
        eqxy_ap_inv=psinv(Uap,sap,Vap)
        parap=np.dot(eqxy_ap_inv, rsxy_ap)
        Ndr_ap=0.0
        for ismpl in range(smpl_ap.shape[0]):
            Ndr_ap+=get3DoFEqN(smpl_ap[ismpl][0], smpl_ap[ismpl][1],
                            smpl_ap[ismpl][2], smpl_ap[ismpl][3],
                            smpl_ap[ismpl][4], smpl_ap[ismpl][5],
                            smpl_ap[ismpl][6], smpl_ap[ismpl][7],
                            m_ap, rx_ap, ry_ap, parap)
        Ndr_ap/=smpl_ap.shape[0]
        parap = reorder_mdl_param(1, parap, Ndr_ap)
        
    eqxy_ad=[]
    resxy_ad=[]    
    for ismpl in range(smpl_ad.shape[0]):
        eqxy,resxy=get3DoFEqXY(smpl_ad[ismpl][0], smpl_ad[ismpl][1],
                            smpl_ad[ismpl][2], smpl_ad[ismpl][3],
                            smpl_ad[ismpl][4], smpl_ad[ismpl][5],
                            smpl_ad[ismpl][6], smpl_ad[ismpl][7],
                            m_ad, rx_ad, ry_ad)
        eqxy_ad.append(eqxy[0])
        eqxy_ad.append(eqxy[1])
        resxy_ad.append(resxy[0])
        resxy_ad.append(resxy[1])
    eqxy_ad = np.array(eqxy_ad)
    resxy_ad = np.array(resxy_ad)

    Uad,sad,Vad=np.linalg.svd(eqxy_ad, full_matrices=True)
    if(is_rank_full(sad)):
        eqxy_ad_inv=psinv(Uad,sad,Vad)
        parad=np.dot(eqxy_ad_inv, rsxy_ad)
        Ndr_ad=0.0
        for ismpl in range(smpl_ad.shape[0]):
            Ndr_ad+=get3DoFEqN(smpl_ad[ismpl][0], smpl_ad[ismpl][1],
                            smpl_ad[ismpl][2], smpl_ad[ismpl][3],
                            smpl_ad[ismpl][4], smpl_ad[ismpl][5],
                            smpl_ad[ismpl][6], smpl_ad[ismpl][7],
                            m_ad, rx_ad, ry_ad, parad)
        Ndr_ad/=smpl_ad.shape[0]
        parad = reorder_mdl_param(0, parad, Ndr_ad)
        

    par={**parad,**parap,**paras}
    print_mdl_param_update(par)
    set_mdl_param(par)
    log.save_model_param(path_model_param)    
    
    
def solve3DoFModel(path_model_param, path_log, logs, path_result, force=False):
    #check logs processed
    log = AWS1Log()
    for log_time in logs:
        if not os.path.exists(path_result+"/"+log_time):          
            log.load(path_log, int(log_time))
            log.proc(0, sys.float_info.max, path_result+"/"+log_time)
    # load initial model parameter
    log.load_model_param(path_model_param)
    
    # check sogrpm file existence
    path_sogrpm = path_result + "/sogrpm"
    if not os.path.exists(path_sogrpm):
        print("sogrpm result is not found. Now sogrpm is gonna run") 
        procOpSogRpm(path_log, logs, path_result, force=False)

    # loadun parameter
    parun = ldl.loadParun(path_sogrpm)
    
    # calculate mode threshold
    # 0 > u astern
    # 0 < u < uthpd ahead displacement
    # uthpd < u ahead plane)
    uthpd = opt.cu(parun)    

    # -2.0 to 0.0
    #  0.0 to uthpd
    # uthpd to 40.0
    eqstas = []
    rstas = []
    eqstahd = []
    rstahd = []
    eqstahp = []
    rstahp = []
    uset = [ 0.5 * i for i in range(-4, 40)]
    for u in range(-2, 40):
        eql=ldl.getStableStraightEq(float(u), opt.funcun(parun, float(u)))
        eqr=0
        if u < 0:
            eqstas.append(eql)
            rstas.append(eqr)
        elif u < uthpd:
            eqstahd.append(eql)
            rstahd.append(eqr)
        else:        
            eqstahp.append(eql)
            rstahp.append(eqr)

             
    # loadturns
    for log_time in logs:
        turns = ldl.loadStableTurn(path_result + "/" + log_time)
        for turn in turns:            
            u = turn[5]
            v = turn[6]
            r = turn[7]
            psi = turn[8]
            n = turn[9]
            if u < 0:
                m = log.mdl_params["m2"]
                rx = log.mdl_params["xr2"]
                ry = log.mdl_params["yr2"]
            elif u < uthpd:
                m = log.mdl_params["m0"]
                rx = log.mdl_params["xr0"]
                ry = log.mdl_params["yr0"]             
            else:             
                m = log.mdl_params["m1"]
                rx = log.mdl_params["xr1"]
                ry = log.mdl_params["yr1"]
             
            eql,eqr = ldl.getStableTurnEq(turn[5],turn[6],turn[7],turn[8],turn[9], m, rx, ry)
            for ieq in range(len(eql)):
                if u < 0:
                    eqstas.append(eql[ieq])
                    rstas.append(eqr[ieq])
                elif u < uthpd:
                    eqstahd.append(eql[ieq])
                    rstahd.append(eqr[ieq])
                else:        
                    eqstahp.append(eql[ieq])
                    rstahp.append(eqr[ieq])

    is_valid_paras = False
    is_valid_par_ahd = False
    is_valid_par_ahp = False
    def is_rank_full(s,eps=1.0e-6):
        for i in range(s.shape[0]):
            if(abs(s[i]) < eps):
                return False
        return True

    eqstas = np.array(eqstas)
    rstas = np.array(rstas)
    # USVx=b
    # (USV)^t(USV)x=b
    # V^tS^tU^tUSVx=V^tS^tU^tb
    # x=(V^tS^tSV)^-1V^tS^tU^t b
    # x=V^t(S^tS)^-1VV^tS^tU^t b
    # x=V^t(S^tS)^-1S^tU^t b

    def psinv(U, s, V):
        return np.dot(np.dot(np.transpose(V),np.pad(np.diag(1/s), [(0,V.shape[1]-s.shape[0]),(0,U.shape[0]-s.shape[0])],'constant')),np.transpose(U))
    
    if(eqstas.shape[0] >= rstas.shape[0]):
        Uas, sas, Vas = np.linalg.svd(eqstas, full_matrices=True)
        if(is_rank_full(sas)):
            eqstas_inv=psinv(Uas,sas,Vas)
            paras = np.dot(eqstas_inv, rstas)
            is_valid_paras = True
    
    eqstahd = np.array(eqstahd)
    rstahd = np.array(rstahd)
    if(eqstahd.shape[0] >= rstahd.shape[0]):
        Uahd, sahd, Vahd = np.linalg.svd(eqstahd, full_matrices=True)
        if(is_rank_full(sahd)):
            eqstahd_inv = psinv(Uahd, sahd, Vahd)
            parahd = np.dot(eqstahd_inv, rstahd)
            is_valid_parahd = True
             
    eqstahp = np.array(eqstahp)
    rstahp = np.array(rstahp)
    if(eqstahd.shape[0] >= rstahd.shape[0]):
        Uahp, sahp, Vahp = np.linalg.svd(eqstahp, full_matrices=True)
        if(is_rank_full(sas)):
            eqstahp_inv = psinv(Uahp, sahp, Vahp)
            parahp = np.dot(eqstahp_inv, rstahp)            
            is_valid_parahp = True
    
    #   solve if rank satisfied the dof.
    # paramter vectors (ma_yr=ma_nv, and ma_nr is not appeared in this eq set)
    parstr = ["xg", "yg", "ma_xu", "ma_yv", "ma_nv", "dl_xu", "dl_yv", "dl_yr", "dl_nv", "dl_nr", "dq_xu", "dq_yv", "dq_yr", "dq_nv", "dq_nr", "CL", "CD", "CTL", "CTQ"]

    def print_mdl_param_update(idx, par):
        stridx="%d" % idx
        for i in range(len(par)):
            print(parstr[i]+stridx+(" %0.12f->%0.12f" % (log.mdl_params[parstr[i]+stridx], par[i])))
    
                    
    def set_mdl_param(idx, par):
        stridx="%d" % idx
        for i in range(len(par)):
            log.mdl_params[parstr[i]+stridx] = par[i]

    if(is_valid_parahd):
        print("AHD min eigen val %e" % np.min(sahd))
        print_mdl_param_update(0, parahd)
        set_mdl_param(0, parahd)
        
    if(is_valid_parahp):
        print("AHP min eigen val %e" % np.min(sahp))
        print_mdl_param_update(1, parahp)
        set_mdl_param(1, parahp)

    if(is_valid_paras):
        print("AS min eigen val %e" % np.min(sas))        
        print_mdl_param_update(2, paras)
        set_mdl_param(2, paras)

    log.save_model_param(path_model_param)
    
def procOpSogRpm(path_log, logs, path_result, force=False):
    if not os.path.exists(path_result):
        os.mkdir(path_result)

    path_sogrpm = path_result + "/sogrpm"
    if not os.path.exists(path_sogrpm):
        os.mkdir(path_sogrpm)
    elif not force:
        print("%s exists. Overwrite? (y/n)" % path_sogrpm)
        yorn=sys.stdin.readline().strip()
        if yorn != "y":
            return

    log = AWS1Log()
    for log_time in logs:
        if not os.path.exists(path_result+"/"+log_time):          
            log.load(path_log, int(log_time))
            log.proc(0, sys.float_info.max, path_result+"/"+log_time)
    
    rx = np.array([])
    ry = np.array([])
    for log_time in logs:
        data=np.loadtxt(path_result+"/"+log_time+"/un.csv", delimiter=",") 
        data=np.transpose(data)
        if data.shape[0] != 2:
            continue
       
        rx = np.concatenate((rx,data[0]), axis=0)
        ry = np.concatenate((ry,data[1]), axis=0)
        
    ldl.plotun(path_sogrpm, str_model_state[6], str_model_state[9], rx, ry)
    rx = np.array([])
    ry = np.array([])
    rz = np.array([])
    def loadundu(type, rx, ry, rz):
        for log_time in logs:
            data=np.loadtxt(path_result+"/"+log_time+"/" + type + "undu.csv", delimiter=",")
            data=np.transpose(data)
            if data.shape[0] != 3:
                continue;
            rx = np.concatenate((rx,data[0]), axis=0)
            ry = np.concatenate((ry,data[1]), axis=0)
            rz = np.concatenate((rz,data[2]), axis=0)
        return rx, ry, rz

    rx,ry,rz = loadundu("all-", rx, ry, rz)
    ldl.plotundu(path_sogrpm,
                 par_model_state[6], par_model_state[9], "du",
                 str_model_state[6], str_model_state[9], ["Acceleration in x", "m/ss"],
                 rx, ry, rz)

def printStat(path_log, logs, path_result, strpars):
    log = AWS1Log()
    for log_time in logs:
        if not os.path.exists(path_result+"/"+log_time):          
            log.load(path_log, int(log_time))
            log.proc(0, sys.float_info.max, path_result)

    valss=ldl.loadStatCsvs(path_result, logs, strpars)
    print(strpars)
    for vals in valss:
        print(vals)               

def selectLogByCond(path_log, logs, path_result, cond):
    sellogs=[]
    valss=ldl.loadStatCsvs(path_result, logs, ["t", cond[0]])
    for vals in valss:
        val = float(cond[2])
        if cond[1] == "<":
            if vals[1] < val:
                sellogs.append(vals[0])
        elif cond[1] == ">":
            if vals[1] > val:
                sellogs.append(vals[0])
        elif cond[1] == "<=":
            if vals[1] <= val:
                sellogs.append(vals[0])
        elif cond[1] == ">=":
            if vals[1] >= val:
                sellogs.append(vals[0])                                        
                
    return sellogs

if __name__ == '__main__':
    log = AWS1Log()
