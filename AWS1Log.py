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
        self.model_state = {'t':None, 'model_state':None}
        self.mdl_eng_ctrl = sim.model_engine_ctrl();
        self.mdl_rud_ctrl = sim.model_rudder_ctrl();
        self.mdl_params={}
        
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
            
    def save_model_param(self, path_model_param):
        with open(path_model_param,mode='w') as file:
            for par,val in self.mdl_params.items():
                txt="%s=%f\n" % (par, val)
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

        
        ts=0.0
        te=max([tapinst[-1],tuiinst[-1], tengr[-1],tstatt[-1],tstvel[-1]])
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
            vstvel = ldl.itpltDataVec(lstvel, tcur, tstvel, istvel)
            istatt = ldl.seekNextDataIndex(tcur, istatt, tstatt)
            vstatt = ldl.itpltDataVec(lstatt, tcur, tstatt, istatt)
            iengr = ldl.seekNextDataIndex(tcur, iengr, tengr)
            vengr = ldl.itpltDataVec(lengr, tcur, tengr, iengr)

            sog = vstvel[1] * mps
            beta = (vstvel[0] - (vstatt[2] + vstatt[6])) * radian
            u = math.cos(beta)
            v = math.sin(beta)
            r = vstatt[5]
            n = (gamma >= 1.0 ? vengr[0] : (gamma <= -1.0 ? -vengr[0] : 0))            
            ueng=(vuiinst[0]==0 ? vuiinst[1]:vapinst[1])
            urud=(vuiinst[0]==0 ? vuiinst[2]:vapinst[2])
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
            
            self.mdl_eng_ctrl.update(ueng, gamma, delta, sdelta, dt,
                                     gamma_new, delta_new, sdelta_new)
            self.mdl_rud_ctrl.update(urud, psi, srudder, dt,
                                     psi_new, srudder_new)
            gamma=gamma_new
            delta=delta_new
            sdelta=sdelta_new
            srudder=srudder_new
            
            tcur+=dt

        self.model_state['t'] = vecs[0]
        self.model_state['model_state'] = vecs[1:]
        
        return log_time

    def getRelSogRpmAcl(self, ts=0.0, te=sys.float_info.max):
        lstatt,tstatt = ldl.getListAndTime(par_statt, self.statt)
        terr=ldl.getErrorAtt(tstatt, lstatt)
        lstvel,tstvel = ldl.getListAndTime(par_stvel, self.stvel)
        lctrlst,tctrlst = ldl.getListAndTime(par_cstat, self.ctrlst)
        lengr,tengr = ldl.getListAndTime(par_engr, self.engr)
        
        return ldl.getRelSogRpmAcl(ts, te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr)
    
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
        ret = strm.set(cv2.CAP_PROP_POS_FRAMES, max(0,istrm[1] - 1))
        ret,frm = strm.read()
        ifrm = strm.get(cv2.CAP_PROP_POS_FRAMES)
        while ifrm != istrm[1]:
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
            vstvel = ldl.itpltDataVec(lstvel, tcur, tstvel, istvel)
            ldl.printDataVec("stvel", par_stvel, vstvel)

            istatt = ldl.seekNextDataIndex(tcur, istatt, tstatt)
            vstatt = ldl.itpltDataVec(lstatt, tcur, tstatt, istatt)
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
                while ifrm != istrm[1]:
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
        ret = strm.set(cv2.CAP_PROP_POS_FRAMES, max(0,istrm[1] - 1))
        ret,frm = strm.read()
        ifrm = strm.get(cv2.CAP_PROP_POS_FRAMES)
        while ifrm != istrm[1]:
            ret,frm = strm.read()
            ifrm += 1

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

            ftotal = ldl.integrateData(tengd, self.engd['frate'])
            ftotal /= 3600.0
            str="ftotal,%f,%f,%f,%f\n" % (ftotal, ftotal, ftotal, ftotal)
            statcsv.write(str)

        # plot data
        ldl.plotDataSection(path, par_cinst, str_cinst,
                                lapinst, tapinst, iapinst[0], iapinstf[1])
        ldl.plotDataSection(path, par_cinst, str_cinst,
                                luiinst, tuiinst, iuiinst[0], iuiinstf[1])
        ldl.plotDataSection(path, par_cstat, str_cstat,
                                lctrlst, tctrlst, ictrlst[0], ictrlstf[1])
        ldl.plotDataSection(path, par_stpos, str_stpos,
                                lstpos, tstpos, istpos[0], istposf[1])
        ldl.plotDataSection(path, par_stvel, str_stvel,
                                lstvel, tstvel, istvel[0], istvelf[1])
        ldl.plotDataSection(path, par_statt, str_statt,
                                lstatt, tstatt, istatt[0], istattf[1])
        ldl.plotDataSection(path, par_9dof, str_9dof,
                                lst9dof, tst9dof, i9dof[0], i9doff[1])
        ldl.plotDataSection(path, par_stdp, str_stdp,
                                lstdp, tstdp, istdp[0], istdpf[1])
        ldl.plotDataSection(path, par_engr, str_engr,
                                lengr, tengr, iengr[0], iengrf[1])
        ldl.plotDataSection(path, par_engd, str_engd,
                                lengd, tengd, iengd[0], iengdf[1])

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
        ldl.plotDataRelation(path, "meng", "rpm", str_cstat[0], str_engr[0], rx, ry)
        rx,ry=ldl.getRelFieldSogCog(ts,te,tstvel,lstvel,tctrlst,lctrlst, terr)
        ldl.plotDataRelation(path, par_stvel[1], par_stvel[0], str_stvel[1], str_stvel[0], rx, ry)
        rx,ry=ldl.getRelSogRpm(ts,te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr)
        ldl.plotSogRpm(path, par_stvel[1], par_engr[0], str_stvel[1], str_engr[0], rx, ry)
        
        rx,ry,rz=ldl.getRelSogRpmAcl(ts,te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr)
        ldl.plotSogRpmAcl(path, par_stvel[1], par_engr[0], par_stvel[3], str_stvel[1], str_engr[0], str_stvel[3], rx, ry, rz)
        

def plotOpSogRpm(path_log, logs, path_result, force=False):
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
            log.plot(0, sys.float_info.max, path_result)
    
    rx = np.array([])
    ry = np.array([])
    for log_time in logs:
        data=np.loadtxt(path_result+"/"+log_time+"/sogrpm.csv", delimiter=",")     
        data=np.transpose(data)
        if data.shape[0] != 2:
            continue
       
        rx = np.concatenate((rx,data[0]), axis=0)
        ry = np.concatenate((ry,data[1]), axis=0)
        
    ldl.plotSogRpm(path_sogrpm, par_stvel[1], par_engr[0],
                             str_stvel[1], str_engr[0], rx, ry)
    rx = np.array([])
    ry = np.array([])
    rz = np.array([])
    for log_time in logs:
        data=np.loadtxt(path_result+"/"+log_time+"/sogrpmdsog.csv", delimiter=",")
        data=np.transpose(data)
        if data.shape[0] != 3:
            continue;
        rx = np.concatenate((rx,data[0]), axis=0)
        ry = np.concatenate((ry,data[1]), axis=0)
        rz = np.concatenate((rz,data[2]), axis=0)
        
    ldl.plotSogRpmAcl(path_sogrpm,
                          par_stvel[1], par_engr[0], par_stvel[3],         
                          str_stvel[1], str_engr[0], str_stvel[3],
                          rx, ry, rz)

def printStat(path_log, logs, path_result, strpars):
    log = AWS1Log()
    for log_time in logs:
        if not os.path.exists(path_result+"/"+log_time):          
            log.load(path_log, int(log_time))
            log.plot(0, sys.float_info.max, path_result)

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
