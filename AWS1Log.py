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
par_statt=['roll','pitch','yaw','droll', 'dpitch', 'dyaw']
str_statt=[["Roll","Deg"],["Pitch","Deg"],["Yaw","Deg"],["Roll Rate", "Deg/s"],["Pitch Rate", "Deg/s"],["Yaw Rate", "Deg/s"]]
par_9dof=['mx','my','mz','ax','ay','az','gx','gy','gz']
str_9dof=[["Magnetic Field in X", "None"],["Magnetic Field in Y", "None"],["Magnetic Field in Z", "None"],
            ["Acceleration in X", "None"],  ["Acceleration in Y", "None"],  ["Acceleration in Z", "None"],
            ["Angular Velocity in X", "None"],["Angular Velocity in Y", "None"],["Angular Velocity in Z","None"]]
par_cstat=['meng','seng','rud']
str_cstat=[["Main Engine Throttle Control","None"], ["Sub Engine Throttle Control","None"],["Rudder Control", "None"]]
par_cinst=['acs','meng','seng','rud']
str_cinst=[["Control Source", "None"], ["Main Engine Throttle Control","None"], ["Sub Engine Throttle Control","None"],["Rudder Control", "None"]]

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

    def load(self, path_aws1_log, log_time=-1):
        data,log_time=ldl.loadAWS1LogFiles(path_aws1_log, log_time)
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

        return log_time

    def getRelSogRpm(self, ts=0.0, te=sys.float_info.max):
        lstatt,tstatt = ldl.getListAndTime(par_statt, self.statt)
        terr=ldl.getErrorAtt(tstatt, lstatt)
        lstvel,tstvel = ldl.getListAndTime(par_stvel, self.stvel)
        lctrlst,tctrlst = ldl.getListAndTime(par_cstat, self.ctrlst)
        lengr,tengr = ldl.getListAndTime(par_engr, self.engr)
        return ldl.getRelSogRpm(ts, te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr)
    
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
        
        iapinst = ldl.seekAWS1LogTime(tapinst, ts)
        iuiinst = ldl.seekAWS1LogTime(tuiinst, ts)
        ictrlst = ldl.seekAWS1LogTime(tctrlst, ts)
        istpos = ldl.seekAWS1LogTime(tstpos, ts)
        istvel = ldl.seekAWS1LogTime(tstvel, ts)
        istatt = ldl.seekAWS1LogTime(tstatt, ts)
        i9dof = ldl.seekAWS1LogTime(tst9dof, ts)
        istdp = ldl.seekAWS1LogTime(tstdp, ts)
        iengr = ldl.seekAWS1LogTime(tengr, ts)
        iengd = ldl.seekAWS1LogTime(tengd, ts) 
        tstrm = self.strm['t']
        istrm = ldl.seekAWS1LogTime(tstrm, ts)
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
            vapinst = ldl.itpltAWS1DataVec(lapinst, tcur, tapinst, iapinst)
            ldl.printAWS1DataVec("apinst", par_cinst, vapinst)

            iuiinst = ldl.seekNextDataIndex(tcur, iuiinst, tuiinst)
            vuiinst = ldl.itpltAWS1DataVec(luiinst, tcur, tuiinst, iuiinst)
            ldl.printAWS1DataVec("uiinst", par_cinst, vuiinst)

            ictrlst = ldl.seekNextDataIndex(tcur, ictrlst, tctrlst)
            vctrlst = ldl.itpltAWS1DataVec(lctrlst, tcur, tctrlst, ictrlst)
            ldl.printAWS1DataVec("ctrlst", par_cstat, vctrlst)

            istpos = ldl.seekNextDataIndex(tcur, istpos, tstpos)
            vstpos = ldl.itpltAWS1DataVec(lstpos, tcur, tstpos, istpos)
            ldl.printAWS1DataVec("stpos", par_stpos, vstpos)

            istvel = ldl.seekNextDataIndex(tcur, istvel, tstvel)
            vstvel = ldl.itpltAWS1DataVec(lstvel, tcur, tstvel, istvel)
            ldl.printAWS1DataVec("stvel", par_stvel, vstvel)

            istatt = ldl.seekNextDataIndex(tcur, istatt, tstatt)
            vstatt = ldl.itpltAWS1DataVec(lstatt, tcur, tstatt, istatt)
            ldl.printAWS1DataVec("statt", par_statt, vstatt)

            i9dof = ldl.seekNextDataIndex(tcur, i9dof, tst9dof)
            vst9dof = ldl.itpltAWS1DataVec(lst9dof, tcur, tst9dof, i9dof)
            ldl.printAWS1DataVec("st9dof", par_9dof, vst9dof)

            istdp = ldl.seekNextDataIndex(tcur, istdp, tstdp)
            vstdp = ldl.itpltAWS1DataVec(lstdp, tcur, tstdp, istdp)
            ldl.printAWS1DataVec("stdp", par_stdp, vstdp)

            iengr = ldl.seekNextDataIndex(tcur, iengr, tengr)
            vengr = ldl.itpltAWS1DataVec(lengr, tcur, tengr, iengr)
            ldl.printAWS1DataVec("engr", par_engr, vengr)

            iengd = ldl.seekNextDataIndex(tcur, iengd, tengd)
            vengd = ldl.itpltAWS1DataVec(lengd, tcur, tengd, iengd)
            ldl.printAWS1DataVec("engr", par_engd, vengd)
        
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

    def plot(self, ts=0, te=sys.float_info.max, path='./'):
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
        
        iapinst = ldl.seekAWS1LogTime(tapinst, ts)
        iapinstf = ldl.seekAWS1LogTime(tapinst, te)
        iuiinst = ldl.seekAWS1LogTime(tuiinst, ts)
        iuiinstf = ldl.seekAWS1LogTime(tuiinst, te)
        ictrlst = ldl.seekAWS1LogTime(tctrlst, ts)
        ictrlstf = ldl.seekAWS1LogTime(tctrlst, te)
        istpos = ldl.seekAWS1LogTime(tstpos, ts)
        istposf = ldl.seekAWS1LogTime(tstpos, te)
        istvel = ldl.seekAWS1LogTime(tstvel, ts)
        istvelf = ldl.seekAWS1LogTime(tstvel, te)
        istatt = ldl.seekAWS1LogTime(tstatt, ts)
        istattf = ldl.seekAWS1LogTime(tstatt, te)
        i9dof = ldl.seekAWS1LogTime(tst9dof, ts)
        i9doff = ldl.seekAWS1LogTime(tst9dof, te)
        istdp = ldl.seekAWS1LogTime(tstdp, ts)
        istdpf = ldl.seekAWS1LogTime(tstdp, te)
        iengr = ldl.seekAWS1LogTime(tengr, ts)
        iengrf = ldl.seekAWS1LogTime(tengr, te)
        iengd = ldl.seekAWS1LogTime(tengd, ts) 
        iengdf = ldl.seekAWS1LogTime(tengd, te) 
        tstrm = self.strm['t']
        istrm = ldl.seekAWS1LogTime(tstrm, ts)
        istrmf = ldl.seekAWS1LogTime(tstrm, te) 
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

            ftotal = ldl.integrateAWS1Data(tengd, self.engd['frate'])
            ftotal /= 3600.0
            str="ftotal,%f,%f,%f,%f\n" % (ftotal, ftotal, ftotal, ftotal)
            statcsv.write(str)

        # plot data
        ldl.plotAWS1DataSection(path, par_cinst, str_cinst,
                                lapinst, tapinst, iapinst[0], iapinstf[1])
        ldl.plotAWS1DataSection(path, par_cinst, str_cinst,
                                luiinst, tuiinst, iuiinst[0], iuiinstf[1])
        ldl.plotAWS1DataSection(path, par_cstat, str_cstat,
                                lctrlst, tctrlst, ictrlst[0], ictrlstf[1])
        ldl.plotAWS1DataSection(path, par_stpos, str_stpos,
                                lstpos, tstpos, istpos[0], istposf[1])
        ldl.plotAWS1DataSection(path, par_stvel, str_stvel,
                                lstvel, tstvel, istvel[0], istvelf[1])
        ldl.plotAWS1DataSection(path, par_statt, str_statt,
                                lstatt, tstatt, istatt[0], istattf[1])
        ldl.plotAWS1DataSection(path, par_9dof, str_9dof,
                                lst9dof, tst9dof, i9dof[0], i9doff[1])
        ldl.plotAWS1DataSection(path, par_stdp, str_stdp,
                                lstdp, tstdp, istdp[0], istdpf[1])
        ldl.plotAWS1DataSection(path, par_engr, str_engr,
                                lengr, tengr, iengr[0], iengrf[1])
        ldl.plotAWS1DataSection(path, par_engd, str_engd,
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
        ldl.plotAWS1DataRelation(path, "meng", "rpm", str_cstat[0], str_engr[0], rx, ry)
        rx,ry=ldl.getRelSogRpm(ts,te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr)
        ldl.plotAWS1DataRelation(path, "sog", "rpm", str_stvel[1], str_engr[0], rx, ry)
        rx,ry=ldl.getRelFieldSogCog(ts,te,tstvel,lstvel,tctrlst,lctrlst, terr)
        ldl.plotAWS1DataRelation(path, "sog", "cog", str_stvel[1], str_stvel[0], rx, ry)

        rx,ry,rz=ldl.getRelSogRpmAcl(ts,te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr)
        ldl.plotAWS1DataRelation3D(path, "sog", "rpm", "dsog", str_stvel[1], str_engr[0], str_stvel[3], rx, ry, rz)

def plotAWS1MstatSogRpm(path_log, logs, path_plot):
    if not os.path.exists(path_plot):
        os.mkdir(path_plot)

    path_sogrpm = path_plot + "/sogrpm"    
    if not os.path.exists(path_sogrpm):
        os.mkdir(path_sogrpm)
    else:
        print("%s exists. Overwrite? (y/n)" % path)
        yorn=sys.stdin.readline().strip()
        if yorn != "y":
            return

    log = AWS1Log()
    rx = np.array([])
    ry = np.array([])
    for log_time in logs:
        log.load(path_log, int(log_time.decode("utf-8")))
        _rx,_ry=log.getRelSogRpm(0, sys.float_info.max)     
        rx = np.concatenate((rx,_rx), axis=0)
        ry = np.concatenate((ry,_ry), axis=0)
    ldl.plotAWS1DataRelation(path_sogrpm, par_stvel[1], par_engr[0],
                             str_stvel[1], str_engr[1], rx, ry)
   
    
if __name__ == '__main__':
    #loadAWS1LogFiles("/mnt/c/cygwin64/home/yhmtm/aws/log")
    log = AWS1Log()
    awspath="/mnt/d/aws"
    #awspath="/mnt/c/cygwin64/home/yhmtm/aws"
    log_time = ldl.selectAWS1Log(awspath+"/log")
    log_time = log.load(awspath+"/log", log_time)
    plot_dir=("/plot_%d" % log_time)
    log.plot(0,550, awspath+plot_dir)
    log.play(0,10000)
