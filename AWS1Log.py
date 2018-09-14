import sys
import os
import pdb          # for debug
import re           # for regular expression
import subprocess   # for command execution 
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf
from utils import label_map_util
import cv2
import ldAWS1Video as ldv

pdb.set_trace()
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
    return np.matmul(Rr,np.matmul(Rp,Ry))

#RrRpRy
#(RrRpRy)^t=(RpRy)^tRr^t=Ry^tRp^tRr^t
Raw=np.eye(3)
Ras=np.eye(3)
Rcs=genRmat(0.5*math.pi, 0.0, 0.5*math.pi)

Ta=np.array([2.0,0,0])
Tc=np.array([0,0,2.0])
Ts=np.array([0,0,0])
horizon=genHorizonPoints(Tc[2])
R=np.matmul(Rcs, np.matmul(Ras.transpose(),Raw))
#Camera position in the world:T=R^tTc+Ts
T=np.matmul(R.transpose(), Tc) + Ts
#sm=P(RMw-T)
#m=np.matmul(Pcam, np.matmul(R, horizon) - T)

channels=["ais_obj", "aws1_ctrl_ap1", "aws1_ctrl_stat", "aws1_ctrl_ui", "engstate", "state"]
chantypes=["ais_obj", "aws1_ctrl_inst", "aws1_ctrl_stat", "aws1_ctrl_inst", "engstate", "state"]

par_engr=['rpm','trim']
str_engr=[["Engine Rev", "RPM"], ["Engine Trim", "None"]]
par_engd=['valt','temp','frate']
str_engd=[["Alternator Output", "V"], ["Engine Temperature", "DegC"], ["Fuel Consumption", "L/h"]]
par_stpos=['lat','lon','alt']
str_stpos=[["Latitude", "Deg"], ["Longitude", "Deg"], ["Altitude", "m"]]
par_stvel=['cog','sog', 'dcog']
str_stvel=[["Course Over Ground","Deg"], ["Speed Over Ground", "kts"], ["Acceleration Over Ground", "kts/s"]]
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

def calcTimeStat(tvec):
    dtvec = diffAWS1Data(tvec)
    return calcStat(dtvec)

def printTimeStat(tvec):
    ttotal = tvec[-1] - tvec[0]
    dtmax, dtmin, dtavg, dtstd = calcTimeStat(tvec)
    print("Total time %f  Time step min: %f avg: %f max: %f std: %f" % (ttotal, dtmin, dtavg, dtmax, dtstd)) 

def calcStat(vec):
    return np.average(vec), np.max(vec), np.min(vec), np.std(vec)

def printStat(vname, vec):
    vmax, vmin, vavg, vstd = calcStat(vec)
    print("%s max: %f min: %f avg: %f std: %f" % (vname, vmax, vmin, vavg, vstd))
  
def diffAWS1Data(vec):
    vnew=np.empty(shape=(vec.shape[0]-1), dtype=vec.dtype)
    for i in range(vnew.shape[0]):
        vnew[i] = vec[i+1] - vec[i]
    return vnew

def diffAWS1DataVec(t, vec):
    dvec=np.zeros(shape=(vec.shape), dtype=vec.dtype)
    for i in range(1, dvec.shape[0]-1):
        dt0 = t[i]-t[i-1]
        dt1 = t[i+1]-t[i]
        idt = 1.0 / (dt0 + dt1)
        dvec[i] = idt * (vec[i+1] - vec[i-1])
    return dvec

def diffAWS1DataYaw(t, yaw):
    dyaw = np.zeros(shape=t.shape, dtype=float)

    for i in range(1, dyaw.shape[0]-1):
        dt0=t[i]-t[i-1]
        dt1=t[i+1]-t[i]
        idt = 1.0 / (dt0 + dt1)

        diff = yaw[i+1] - yaw[i-1]
        if diff > 180.:
            diff -= 360
        elif diff < -180: 
            diff += 360.
        dyaw[i] = idt * diff
    return dyaw

def diffAWS1DataCog(t, cog):
    dcog = np.zeros(shape=t.shape, dtype=float)

    for i in range(1, dcog.shape[0]-1):
        dt0=t[i]-t[i-1]
        dt1=t[i+1]-t[i]
        idt = 1.0 / (dt0 + dt1)
    
        diff = cog[i+1] - cog[i-1]
        if diff > 180.:
            diff -= 360.
        elif diff <= -180.:
            diff += 360.
        dcog[i] = idt * diff
    return dcog

def integrateAWS1Data(t, vec):
    tprev = t[0]
    vprev = vec[0]
    s = 0.0
    for i in range(1, len(t)):
        tcur = t[i]
        vcur = vec[i]
        dt = 0.5 * (tcur - tprev)
        s += dt * (vcur + vprev)
        tprev = tcur
        vprev = vcur
    return s

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
        data,log_time=loadAWS1LogFiles(path_aws1_log, log_time)
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

    def stat(self, ts=0.0, te=sys.float_info.max):
        lapinst = listAWS1DataSection(par_cinst, self.apinst)
        tapinst = self.apinst['t']
        iapinst = seekAWS1LogTime(tapinst, ts)
        iapinstf = seekAWS1LogTime(tapinst, te)
        luiinst = listAWS1DataSection(par_cinst, self.uiinst)
        tuiinst = self.uiinst['t']
        iuiinst = seekAWS1LogTime(tuiinst, ts)
        iuiinstf = seekAWS1LogTime(tuiinst, te)
        lctrlst = listAWS1DataSection(par_cstat, self.ctrlst)
        tctrlst = self.ctrlst['t']
        ictrlst = seekAWS1LogTime(tctrlst, ts)
        ictrlstf = seekAWS1LogTime(tctrlst, te)
        lstpos = listAWS1DataSection(par_stpos, self.stpos)
        tstpos = self.stpos['t']
        istpos = seekAWS1LogTime(tstpos, ts)
        istposf = seekAWS1LogTime(tstpos, te)
        lstvel = listAWS1DataSection(par_stvel, self.stvel)
        tstvel = self.stvel['t']
        istvel = seekAWS1LogTime(tstvel, ts)
        istvelf = seekAWS1LogTime(tstvel, te)
        lstatt = listAWS1DataSection(par_statt, self.statt)
        tstatt = self.statt['t']
        istatt = seekAWS1LogTime(tstatt, ts)
        istattf = seekAWS1LogTime(tstatt, te)
        lst9dof = listAWS1DataSection(par_9dof, self.st9dof)
        tst9dof = self.st9dof['t']
        i9dof = seekAWS1LogTime(tst9dof, ts)
        i9doff = seekAWS1LogTime(tst9dof, te)
        lstdp = listAWS1DataSection(par_stdp, self.stdp)
        tstdp = self.stdp['t']
        istdp = seekAWS1LogTime(tstdp, ts)
        istdpf = seekAWS1LogTime(tstdp, te)
        lengr = listAWS1DataSection(par_engr, self.engr)
        tengr = self.engr['t']
        iengr = seekAWS1LogTime(tengr, ts)
        iengrf = seekAWS1LogTime(tengr, te)
        lengd = listAWS1DataSection(par_engd, self.engd)
        tengd = self.engd['t']
        iengd = seekAWS1LogTime(tengd, ts) 
        iengdf = seekAWS1LogTime(tengd, te) 
        tstrm = self.strm['t']
        istrm = seekAWS1LogTime(tstrm, ts)
        istrmf = seekAWS1LogTime(tstrm, te) 
        strm = self.strm['strm']   
        ret = strm.set(cv2.CAP_PROP_POS_FRAMES, max(0,istrm[1] - 1))
        ret,frm = strm.read()
        ifrm = strm.get(cv2.CAP_PROP_POS_FRAMES)
        while ifrm != istrm[1]:
            ret,frm = strm.read()
            ifrm += 1

        print("STAT apinst")
        printTimeStat(tapinst)
        for key in par_cinst:
            printStat(key, self.apinst[key])

        print("STAT uiinst")
        printTimeStat(tuiinst)
        for key in par_cinst:
            printStat(key, self.uiinst[key])

        print("STAT ctrlstat")
        printTimeStat(tctrlst)
        for key in par_cstat:
            printStat(key, self.ctrlst[key])

        print("STAT stpos")
        printTimeStat(tstpos)
        for key in par_stpos:
            printStat(key, self.stpos[key])

        print("STAT stvel")
        printTimeStat(tstvel)
        for key in par_stvel:
            printStat(key, self.stvel[key])

        print("STAT statt")
        printTimeStat(tstatt)
        for key in par_statt:
            printStat(key, self.statt[key])

        print("STAT 9dof")
        printTimeStat(tst9dof)
        for key in par_9dof:
            printStat(key, self.st9dof[key])

        print("STAT stdp")
        printTimeStat(tstdp)
        for key in par_stdp:
            printStat(key, self.stdp[key])

        print("STAT engr")
        printTimeStat(tengr)
        for key in par_engr:
            printStat(key, self.engr[key])

        print("STAT engd")
        printTimeStat(tengd)
        for key in par_engd:
            printStat(key, self.engd[key])

        ftotal = integrateAWS1Data(tengd, self.engd['frate'])
        ftotal /= 3600.0
        print("Estimated fuel consumption: %f" % ftotal) 

        print("STAT strm")
        printTimeStat(tstrm)

    def play(self, ts, te, dt=0.1):
        # seek head for all data section
        lapinst = listAWS1DataSection(par_cinst, self.apinst)
        tapinst = self.apinst['t']
        iapinst = seekAWS1LogTime(tapinst, ts)
        luiinst = listAWS1DataSection(par_cinst, self.uiinst)
        tuiinst = self.uiinst['t']
        iuiinst = seekAWS1LogTime(tuiinst, ts)
        lctrlst = listAWS1DataSection(par_cstat, self.ctrlst)
        tctrlst = self.ctrlst['t']
        ictrlst = seekAWS1LogTime(tctrlst, ts)
        lstpos = listAWS1DataSection(par_stpos, self.stpos)
        tstpos = self.stpos['t']
        istpos = seekAWS1LogTime(tstpos, ts)
        lstvel = listAWS1DataSection(par_stvel, self.stvel)
        tstvel = self.stvel['t']
        istvel = seekAWS1LogTime(tstvel, ts)
        lstatt = listAWS1DataSection(par_statt, self.statt)
        tstatt = self.statt['t']
        istatt = seekAWS1LogTime(tstatt, ts)
        lst9dof = listAWS1DataSection(par_9dof, self.st9dof)
        tst9dof = self.st9dof['t']
        i9dof = seekAWS1LogTime(tst9dof, ts)
        lstdp = listAWS1DataSection(par_stdp, self.stdp)
        tstdp = self.stdp['t']
        istdp = seekAWS1LogTime(tstdp, ts)
        lengr = listAWS1DataSection(par_engr, self.engr)
        tengr = self.engr['t']
        iengr = seekAWS1LogTime(tengr, ts)
        lengd = listAWS1DataSection(par_engd, self.engd)
        tengd = self.engd['t']
        iengd = seekAWS1LogTime(tengd, ts) 
        tstrm = self.strm['t']
        istrm = seekAWS1LogTime(tstrm, ts)
        strm = self.strm['strm']   
        ret = strm.set(cv2.CAP_PROP_POS_FRAMES, max(0,istrm[1] - 1))
        ret,frm = strm.read()
        ifrm = strm.get(cv2.CAP_PROP_POS_FRAMES)
        while ifrm != istrm[1]:
            ret,frm = strm.read()
            ifrm += 1
        
        printTimeHead("apinst", iapinst, tapinst)
        printTimeHead("uiinst", iuiinst, tuiinst)
        printTimeHead("ctrlst", ictrlst, tctrlst)
        printTimeHead("stpos", istpos, tstpos)
        printTimeHead("stvel", istvel, tstvel)
        printTimeHead("statt", istatt, tstatt)
        printTimeHead("st9dof", i9dof, tst9dof)
        printTimeHead("stdp", istdp, tstdp)
        printTimeHead("engr", iengr, tengr)
        printTimeHead("engd", iengd, tengd)
        printTimeHead("strm", istrm, tstrm)

        tcur = ts

        # Setting up object detector
        label_map=label_map_util.load_labelmap('mscoco_label_map.pbtxt')
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index=label_map_util.create_category_index(categories)
        
        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

        with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        sess=tf.Session(config=config)
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
        bodet=True
        budist=False
        
        while tcur < te:
            print ("Time %fsec" % tcur)
            iapinst = seekNextDataIndex(tcur, iapinst, tapinst)
            vapinst = itpltAWS1DataVec(lapinst, tcur, tapinst, iapinst)
            printAWS1DataVec("apinst", par_cinst, vapinst)

            iuiinst = seekNextDataIndex(tcur, iuiinst, tuiinst)
            vuiinst = itpltAWS1DataVec(luiinst, tcur, tuiinst, iuiinst)
            printAWS1DataVec("uiinst", par_cinst, vuiinst)

            ictrlst = seekNextDataIndex(tcur, ictrlst, tctrlst)
            vctrlst = itpltAWS1DataVec(lctrlst, tcur, tctrlst, ictrlst)
            printAWS1DataVec("ctrlst", par_cstat, vctrlst)

            istpos = seekNextDataIndex(tcur, istpos, tstpos)
            vstpos = itpltAWS1DataVec(lstpos, tcur, tstpos, istpos)
            printAWS1DataVec("stpos", par_stpos, vstpos)

            istvel = seekNextDataIndex(tcur, istvel, tstvel)
            vstvel = itpltAWS1DataVec(lstvel, tcur, tstvel, istvel)
            printAWS1DataVec("stvel", par_stvel, vstvel)

            istatt = seekNextDataIndex(tcur, istatt, tstatt)
            vstatt = itpltAWS1DataVec(lstatt, tcur, tstatt, istatt)
            printAWS1DataVec("statt", par_statt, vstatt)

            i9dof = seekNextDataIndex(tcur, i9dof, tst9dof)
            vst9dof = itpltAWS1DataVec(lst9dof, tcur, tst9dof, i9dof)
            printAWS1DataVec("st9dof", par_9dof, vst9dof)

            istdp = seekNextDataIndex(tcur, istdp, tstdp)
            vstdp = itpltAWS1DataVec(lstdp, tcur, tstdp, istdp)
            printAWS1DataVec("stdp", par_stdp, vstdp)

            iengr = seekNextDataIndex(tcur, iengr, tengr)
            vengr = itpltAWS1DataVec(lengr, tcur, tengr, iengr)
            printAWS1DataVec("engr", par_engr, vengr)

            iengd = seekNextDataIndex(tcur, iengd, tengd)
            vengd = itpltAWS1DataVec(lengd, tcur, tengd, iengd)
            printAWS1DataVec("engr", par_engd, vengd)
        
            istrm = seekNextDataIndex(tcur, istrm, tstrm)
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
                    rows=frm_ud.shape[0]
                    cols=frm_ud.shape[1]
                    #inp = cv2.resize(frm_ud, (960, 540))
                    inp=frm_ud.copy()
                    inp=inp[:,:,[2,1,0]]
                    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                               feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
                
                    num_detections = int(out[0][0])
                    for i in range(num_detections):
                        classId = int(out[3][0][i])
                        if classId in category_index.keys():
                            oname=category_index[classId]['name']
                        else:
                            oname='Unknown'
                        score = float(out[1][0][i])
                        bbox=[float(v) for v in out[2][0][i]]
                    
                        if score > 0.3:
                            x = bbox[1] * cols
                            y = bbox[0] * rows
                            right = bbox[3] * cols
                            bottom = bbox[2] * rows
                            cv2.rectangle(frm_ud, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frm_ud, oname, (int(x), int(y)+20), font ,1, (0, 255, 0), 2, cv2.LINE_AA)            
                font=cv2.FONT_HERSHEY_SIMPLEX
                txt="Time %5.2f Frame %06d" % (tcur, ifrm)
                
                if(budist):                
                    txt+=" Undist"
                cv2.putText(frm_ud, txt, (0, 30), font, 1, (0,255,0), 2, cv2.LINE_AA)
                txt="RUD %03f ENG %03f REV %04f SOG %03f" % (vuiinst[3], vuiinst[1],vengr[0], vstvel[1])
                cv2.putText(frm_ud, txt, (0, 60), font, 1, (0,255,0), 2, cv2.LINE_AA)            
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
        lapinst = listAWS1DataSection(par_cinst, self.apinst)
        tapinst = self.apinst['t']
        iapinst = seekAWS1LogTime(tapinst, ts)
        iapinstf = seekAWS1LogTime(tapinst, te)
        luiinst = listAWS1DataSection(par_cinst, self.uiinst)
        tuiinst = self.uiinst['t']
        iuiinst = seekAWS1LogTime(tuiinst, ts)
        iuiinstf = seekAWS1LogTime(tuiinst, te)
        lctrlst = listAWS1DataSection(par_cstat, self.ctrlst)
        tctrlst = self.ctrlst['t']
        ictrlst = seekAWS1LogTime(tctrlst, ts)
        ictrlstf = seekAWS1LogTime(tctrlst, te)
        lstpos = listAWS1DataSection(par_stpos, self.stpos)
        tstpos = self.stpos['t']
        istpos = seekAWS1LogTime(tstpos, ts)
        istposf = seekAWS1LogTime(tstpos, te)
        lstvel = listAWS1DataSection(par_stvel, self.stvel)
        tstvel = self.stvel['t']
        istvel = seekAWS1LogTime(tstvel, ts)
        istvelf = seekAWS1LogTime(tstvel, te)
        lstatt = listAWS1DataSection(par_statt, self.statt)
        tstatt = self.statt['t']
        istatt = seekAWS1LogTime(tstatt, ts)
        istattf = seekAWS1LogTime(tstatt, te)
        lst9dof = listAWS1DataSection(par_9dof, self.st9dof)
        tst9dof = self.st9dof['t']
        i9dof = seekAWS1LogTime(tst9dof, ts)
        i9doff = seekAWS1LogTime(tst9dof, te)
        lstdp = listAWS1DataSection(par_stdp, self.stdp)
        tstdp = self.stdp['t']
        istdp = seekAWS1LogTime(tstdp, ts)
        istdpf = seekAWS1LogTime(tstdp, te)
        lengr = listAWS1DataSection(par_engr, self.engr)
        tengr = self.engr['t']
        iengr = seekAWS1LogTime(tengr, ts)
        iengrf = seekAWS1LogTime(tengr, te)
        lengd = listAWS1DataSection(par_engd, self.engd)
        tengd = self.engd['t']
        iengd = seekAWS1LogTime(tengd, ts) 
        iengdf = seekAWS1LogTime(tengd, te) 
        tstrm = self.strm['t']
        istrm = seekAWS1LogTime(tstrm, ts)
        istrmf = seekAWS1LogTime(tstrm, te) 
        strm = self.strm['strm']   
        ret = strm.set(cv2.CAP_PROP_POS_FRAMES, max(0,istrm[1] - 1))
        ret,frm = strm.read()
        ifrm = strm.get(cv2.CAP_PROP_POS_FRAMES)
        while ifrm != istrm[1]:
            ret,frm = strm.read()
            ifrm += 1

        if not os.path.exists(path):
            os.mkdir(path)
        def plotAWS1DataSection(keys, str, ldata, ts, i0, i1):
            idt=0
            for key in keys:
                plt.plot(ts[i0:i1], ldata[idt][i0:i1])
                ystr = str[idt][0] + " [" + str[idt][1] + "]"
                idt+=1
                figname=key+".png"
                plt.xlabel("Time [sec]")
                plt.ylabel(ystr)
                plt.savefig(path+"/"+figname)
                plt.clf()

        plotAWS1DataSection(par_cinst, str_cinst, lapinst, tapinst, iapinst[0], iapinstf[1])
        plotAWS1DataSection(par_cinst, str_cinst, luiinst, tuiinst, iuiinst[0], iuiinstf[1])
        plotAWS1DataSection(par_cstat, str_cstat, lctrlst, tctrlst, ictrlst[0], ictrlstf[1])
        plotAWS1DataSection(par_stpos, str_stpos, lstpos, tstpos, istpos[0], istposf[1])
        plotAWS1DataSection(par_stvel, str_stvel, lstvel, tstvel, istvel[0], istvelf[1])
        plotAWS1DataSection(par_statt, str_statt, lstatt, tstatt, istatt[0], istattf[1])
        plotAWS1DataSection(par_9dof, str_9dof, lst9dof, tst9dof, i9dof[0], i9doff[1])
        plotAWS1DataSection(par_stdp, str_stdp, lstdp, tstdp, istdp[0], istdpf[1])
        plotAWS1DataSection(par_engr, str_engr, lengr, tengr, iengr[0], iengrf[1])
        plotAWS1DataSection(par_engd, str_engd, lengd, tengd, iengd[0], iengdf[1])

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

        url = "https://www.openstreetmap.org/#map=13"
        payload={
            'mapnik_format':'png',
            'mapnik_scale': 25000,
            'minlon' : minlon,
            'maxlon' : maxlon,
            'minlat' : minlat,
            'maxlat' : maxlat,
            'format' : 'mapnik'
        }
 #       res = requests.post(url, payload)
 #       print(res.headers)
 #       mapimg = Image.open(BytesIO(res.content))

        bmap = Basemap(projection='merc',
                llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon, urcrnrlon=maxlon,
                lat_ts=0, resolution=None)
#        bmap.imshow(mapimg, origin='upper')
        x,y = bmap(lstpos[1][istpos[0]:istposf[1]], lstpos[0][istpos[0]:istposf[1]])
        bmap.plot(x,y)
        plt.savefig(path+"/"+"map.png")
        plt.clf()


        def plotAWS1DataRelation(parx, pary, strx, stry, rx, ry):
            figname=parx+pary+".png"
            plt.scatter(rx,ry)
            plt.xlabel(strx[0]+" ["+strx[1]+"]")
            plt.ylabel(stry[0]+" ["+stry[1]+"]")
            plt.savefig(path+"/"+figname)
            plt.clf()

        # meng/rpm, 100 < rud < 154
        trrud = findInRangeTimeRanges(tctrlst, lctrlst[2], 154, 100)
        trmeng = findStableTimeRanges(tctrlst, lctrlst[0], smgn=10.0, emgn=0.0, th=1.0)
        trng = intersectTimeRanges(trrud, trmeng)
        trng = intersectTimeRanges(trng, [[ts,te]])
        rx,ry = relateTimeRangeVecs(tctrlst, tengr, lctrlst[0], lengr[0], trng)
        plotAWS1DataRelation("meng", "rpm", str_cstat[0], str_engr[0], rx, ry)
        
def intersectTimeRanges(trng0, trng1):
    trng = []
    for tr0 in trng0:
        for tr1 in trng1:
            if(tr0[0] <= tr1[0]): # s0
                if(tr1[0] <= tr0[1]): # s0s1
                    if(tr0[1] >= tr1[1]): # s0s1e1s1
                        trng.append([tr1[0],tr1[1]])
                    else: # s0s1e0e1
                        trng.append([tr1[0],tr0[1]])
                else: # s0t0s1t1
                    pass
            else: # s1
                if(tr0[0] <= tr1[1]):# s1s0
                    if(tr1[1] <= tr0[1]): # s1s0e1e0
                        trng.append([tr0[0], tr1[1]])
                    else: # s1s0e0e1
                        trng.append([tr0[0], tr0[1]])
                else: # s1e1s0e0                    
                    pass
    return trng

def relateTimeRangeVecs(tx, ty, vx, vy, trng):
    rx = []
    ry = []
    for tr in trng:
        ix0s,ix0e = seekAWS1LogTime(tx, tr[0])
        ix1s,ix1e = seekAWS1LogTime(tx, tr[1])
        for ix in range(ix0e, ix1s):
            iys,iye = seekAWS1LogTime(ty, tx[ix])
            t = tx[ix]
            t0 = ty[iys]
            t1 = ty[iye]
            x = vx[ix]
            y = (vy[iye] * (t - t0) + vy[iys] * (t1 - t))  / (t1 - t0)
            rx.append(x)
            ry.append(y)
    return np.array(rx),np.array(ry)

def findStableTimeRanges(t, vec, smgn=5.0, emgn=0.0, th=1.0):
    ts=-1
    te=-1
    tranges=[]
    vmin = vec[0]
    vmax = vec[0]

    for ivec in range(1, vec.shape[0]):
        vmin = min(vec[ivec], vmin)
        vmax = max(vec[ivec], vmax)
        if vmax - vmin <= th:
            if(ts == -1):
                ts = t[ivec-1]
                te = t[ivec]
            else:
                te = t[ivec]
        else:
            if(ts >= 0):
                if (te - ts > emgn + smgn):
                    tranges.append([ts+smgn, te-emgn])
            ts = te = -1
            vmax = vmin = vec[ivec]

    return tranges

def findInRangeTimeRanges(t, vec, vmax=sys.float_info.max, vmin=-sys.float_info.min):
    ts=-1
    te=-1
    tranges = []

    for ivec in range(0, vec.shape[0]):
        if vmin <= vec[ivec] and vmax >= vec[ivec]:
            if ts == -1:
                ts = te = t[ivec]
            else:
                te = t[ivec]
        else:
            if(ts > 0):
                tranges.append([ts, te])
            ts = te = -1

    return tranges

def seekAWS1LogTime(tseq,tseek):        
    iend=tseq.shape[0]-1
    if(tseek > tseq[-1]):
        return iend, iend
    elif(tseek < tseq[0]):
        return 0, 0

    i=tseq.shape[0]//2
    imin = 0
    imax = iend

    while True:
        if (tseq[i] <= tseek): 
            if(tseq[i+1] > tseek):
                break
            else:
                imin = i
                i += (imax - i) // 2
        else:
            imax = i
            i -= max(1, (i - imin) // 2)

    return i,i+1

def printTimeHead(name, it, t):
    if it[1] == 0:
        print("%s t[<0]") 
    elif it[0] == t.shape[0] -1:
        print("%s t[>-1") 
    else:
        print("%s t[%d]=%f" % (name, it[0], t[it[0]]))

def seekNextDataIndex(tnext, it, t):
    if it[1] == 0:
        if t[0] > tnext:
            return it
        if t[0] <= tnext and t[1] > tnext:
            return [0,1]
    elif it[0] == t.shape[0] - 1:
        return it
    else: 
        if t[it[0]] <= tnext and t[it[1]] > tnext:
            return it
        elif t[it[1]] <= tnext and t[it[1]+1] > tnext:
            return [it[1],it[1]+1]
    return seekAWS1LogTime(t, tnext)

def listAWS1DataSection(keys, data):
    lst = []
    for key in keys:
        lst.append(data[key])
    return lst

def itpltAWS1DataVec(ldata, t, ts, it):
    vec = np.zeros(shape=len(ldata),dtype=float)
    idt = 0
    for data in ldata:
        d0 = ldata[idt][it[0]]
        d1 = ldata[idt][it[1]]
        t0 = ts[it[0]]
        t1 = ts[it[1]]
        vec[idt] = (d1 * (t - t0)  + d0 * (t1 - t)) / (t1 - t0)
        idt += 1
    return vec

def printAWS1DataVec(name, keys, vdata):
    strRec = name
    for idt in range(len(keys)):
        strRec += " %s:%f," % (keys[idt], vdata[idt])
    print (strRec)


def selectAWS1Log(path_aws1_log):
    command=['ls', path_aws1_log]
    files=subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
    logs = re.findall(rb"[0-9]{17}", files)
    if len(logs) == 0:
        return -1
    ilog = 0
    for log in logs:
        command=['t2str', log]
        str_log_time = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
        print(("%d:"%ilog)+log.decode('utf-8') + ":" + str_log_time.decode('utf-8'))
        ilog = ilog + 1
    print("Select log number:")
    str_log_number=sys.stdin.readline()
    log_number=int(str_log_number)
    print("log %d : %s is selected." % (log_number, logs[log_number]))
    log_time = int(logs[log_number])
    path_log= "%s/%d"%(path_aws1_log,log_time)
    print("Check directory.")

    ##### Check channel files
    if(os.path.isdir(path_log)):
        command = ["ls", path_log]
        files = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
        jrs = re.findall(rb".+.jr", files)
        found=False
        for chan in channels:
            for jr in jrs:
                if jr.decode('utf-8') == chan+".jr":
                    found=True
                    break
            if found:
                print(chan+".jr found.")
            else:
                print(chan+".jr not found.")
                return -1
    return log_time


def loadAWS1LogFiles(path_aws1_log, log_time=-1):
    path_log= "%s/%d"%(path_aws1_log,log_time)
    ##### Convert .log to .txt 
    command = ["ls", path_log]
    files = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
    logs_bin = re.findall(rb".+.log", files)
    logs_txt = re.findall(rb".+.txt", files)
    for log_bin in logs_bin:
        name_bin,ext = os.path.splitext(log_bin)
        chan_log = None
        chan_type = None
        for ichan in range(len(channels)):            
            if(re.match(channels[ichan], name_bin.decode('utf-8'))):
                chan_log = channels[ichan]
                chan_type = chantypes[ichan]
                break
        if chan_log is None:
            print("No matching channel for " + log_bin)
            return                

        found=False
        for log_txt in logs_txt:
            name_txt,ext = os.path.splitext(log_txt)
            if (name_txt == name_bin):
                found = True
                break
        if not found:
            path_log_bin = path_log+"/"+log_bin
            command = ["log2txt", chan_type, path_log_bin]
            print("Converting " + log_bin + " to text.")
            subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()

    ##### Scan channel log files
    command = ["ls", path_log]
    files = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
    logs_txt = re.findall(rb".+.txt", files)

    chans_logs = {}

    for chan in channels:
        chan_logs = []

        for log_txt in logs_txt:
            if (re.match(chan, log_txt.decode('utf-8'))):
                chan_logs.append(log_txt)

        print("log for " + chan + ":")
        print(chan_logs)
        chans_logs[chan] = chan_logs

    tstrm,strm=ldv.loadAWS1VideoStream(path_log+"/mako0.avi",path_log+"/mako0.ts")

    apinst = []
    uiinst = []
    ctrlst = []
    stpos = []
    stvel = []
    statt = []
    st9dof = []
    stdp = []
    engr = []
    engd = []

    def concatSectionData(data):
        keys = data[0].keys()
        datall = data[0]
        data.pop(0)
        while len(data) != 0:
            for key in keys: 
                datall[key] = np.concatenate(datall[key], data[0][key])
                data.pop(0)
            
        return datall

    #ctrl ap
    for log in chans_logs[channels[1]]:
        apinst.append(loadAWS1CtrlInst(path_log+"/"+log.decode('utf-8'), log_time))

    apinst = concatSectionData(apinst)

    #ctrl ui
    for log in chans_logs[channels[3]]:
        uiinst.append(loadAWS1CtrlInst(path_log+"/"+log.decode('utf-8'), log_time))

    uiinst = concatSectionData(uiinst)
    
    #ctrl stat
    for log in chans_logs[channels[2]]:
        ctrlst.append(loadAWS1CtrlStat(path_log+"/"+log.decode('utf-8'), log_time))

    ctrlst = concatSectionData(ctrlst)

    #engstate
    for log in chans_logs[channels[4]]:
        rapid,dynamic=loadAWS1Engstate(path_log+"/"+log.decode('utf-8'), log_time)
        engr.append(rapid)
        engd.append(dynamic)

    engr = concatSectionData(engr)
    engd = concatSectionData(engd)
    
    #state
    for log in chans_logs[channels[5]]:
        pos,vel,dp,att,s9dof=loadAWS1State(path_log+"/"+log.decode('utf-8'), log_time)
        stpos.append(pos)
        stvel.append(vel)
        stdp.append(dp)
        statt.append(att)
        st9dof.append(s9dof)
    stpos = concatSectionData(stpos)
    stvel = concatSectionData(stvel)
    stdp = concatSectionData(stdp)
    statt = concatSectionData(statt)
    st9dof = concatSectionData(st9dof)

    return {'apinst':apinst, 'uiinst':uiinst, 'ctrlst':ctrlst, 
        'stpos':stpos, 'stvel':stvel, 'statt':statt, 'st9dof':st9dof, 'stdp':stdp, 
        'engr':engr, 'engd':engd, 'strm':{'t':tstrm, 'strm':strm}}, log_time

def loadAWS1Engstate(fname, log_time):
    print("Analyzing " + fname)
    file = open(fname)
    header=file.readline()
    header=header.strip().replace(" ", "").split(',')
 
    # trapid: rpm, trim
    # tdyn: valt, temp, frate
    irecord=0
    itrapid=0 
    irpm=0
    itrim=0
    itdyn=0
    ivalt=0
    itemp=0
    ifrate=0

    for record in header:
        if(record == "trapid"):
            itrapid = irecord
        if(record == "rpm"):
            irpm = irecord
        if(record == "trim"):
            itrim = irecord
        if(record == "tdyn"):
            itdyn = irecord
        if(record == "valt"):
            ivalt = irecord
        if(record == "temp"):
            itemp = irecord
        if(record == "frate"):
            ifrate = irecord            
        irecord+=1

    torg = log_time
    tend = 0
    trapid_prev = 0
    trapid_cur = 0
    tdyn_prev = 0
    tdyn_cur = 0
    trapid=[]
    rpm=[]
    rpm_max = 0.0
    rpm_min = 0.0
    rpm_avg = 0.0
    trim=[]

    tdyn=[]
    valt=[]
    valt_max = 0.0
    valt_min = 0.0
    valt_avg = 0.0
    temp=[]
    temp_max = 0.0
    temp_min = 0.0
    temp_avg = 0.0
    frate=[]
    frate_max = 0.0
    frate_min = 0.0
    frate_avg = 0.0
    ntrapid=0
    dtrapid=0.0
    dtrapid_max=0.0
    dtrapid_min=sys.float_info.max
    ntdyn=0
    dtdyn=0.0
    dtdyn_max = 0.0
    dtdyn_min = sys.float_info.max
    
    while True:
        line = file.readline().strip().split(',')
        if not line or len(line) == 1:
            break

        t = int(line[0])  
        if tend >= t:
            break
        tend = t

        trapid_cur = int(line[itrapid]) - torg
        tdyn_cur = int(line[itdyn]) - torg
        if(trapid_cur > trapid_prev):
            # add new record
            dt = float(trapid_cur - trapid_prev) / 10000000.
            dtrapid_max = max(dtrapid_max, dt)
            dtrapid_min = min(dtrapid_min, dt)

            dtrapid+= dt
            trapid_prev = trapid_cur
            trapid.append(float(trapid_cur) / 10000000.)
            rpm.append(float(line[irpm]))
            trim.append(float(line[itrim]))
            ntrapid+=1

        if(tdyn_cur > tdyn_prev):
            # add new record
            dt = float(tdyn_cur - tdyn_prev) / 10000000.
            dtdyn_max = max(dtdyn_max, dt)
            dtdyn_min = min(dtdyn_min, dt)
            dtdyn+= dt
            tdyn_prev = tdyn_cur
            tdyn.append(float(tdyn_cur) / 10000000.)
            valt.append(float(line[ivalt]))
            temp.append(float(line[itemp])-273.0)
            frate.append(float(line[ifrate]))
            ntdyn+=1

    trapid = np.array(trapid)
    tdyn = np.array(tdyn)
    rpm = np.array(rpm)
    trim = np.array(trim)
    valt = np.array(valt)
    temp = np.array(temp)
    frate = np.array(frate)

    file.close()
    return {'t':trapid, 'rpm':rpm, 'trim':trim} ,{'t':tdyn, 'valt':valt, 'temp':temp, 'frate':frate}

def loadAWS1State(fname, log_time):
    print("Analyzing " + fname)
    file = open(fname)
    header = file.readline().strip().replace(" ", "").split(',')

    # tpos: lat, lon, alt
    # tatt: roll, pitch, yaw
    # t9dofc: mx,my,mz, ax,ay,az,gx,gy,gz
    # tvel: cog, sog
    # tdp: depth
    itpos = 0
    itatt = 0
    it9dofc = 0
    itvel = 0
    itdp = 0
    ilat = 0
    ilon = 0
    ialt = 0
    iroll = 0
    ipitch = 0
    iyaw = 0
    imx = 0
    imy = 0 
    imz = 0
    iax = 0
    iay = 0
    iaz = 0
    igx = 0
    igy = 0
    igz = 0
    icog = 0
    isog = 0
    idepth = 0

    irecord = 0
    for record in header:
        if(record == "tpos"):
            itpos = irecord
        if(record == "tatt"):
            itatt = irecord
        if(record == "t9dofc"):
            it9dofc = irecord
        if(record == "tvel"):
            itvel = irecord
        if(record == "tdp"):
            itdp = irecord
        if(record == "lat"):
            ilat = irecord
        if(record == "lon"):
            ilon = irecord
        if(record == "alt"):
            ialt = irecord
        if(record == "roll"):
            iroll = irecord
        if(record == "pitch"):
            ipitch = irecord
        if(record == "yaw"):
            iyaw = irecord
        if(record == "mx"):
            imx = irecord
        if(record == "my"):
            imy = irecord
        if(record == "mz"):
            imz = irecord
        if(record == "ax"):
            iax = irecord
        if(record == "ay"):
            iay = irecord
        if(record == "az"):
            iaz = irecord
        if(record == "gx"):
            igx = irecord
        if(record == "gy"):
            igy = irecord
        if(record == "gz"):
            igz = irecord
        if(record == "sog"):
            isog = irecord
        if(record == "cog"):
            icog = irecord
        if(record == "depth"):
            idepth = irecord

        irecord += 1

    torg = log_time
    tend = 0
    tpos = []
    tatt = []
    t9dofc = []
    tvel = []
    tdp = []
    tpos_prev = 0
    tatt_prev = 0
    t9dofc_prev = 0
    tvel_prev = 0
    tdp_prev = 0
    tpos_cur = 0
    tatt_cur = 0
    t9dofc_cur = 0
    tvel_cur = 0
    tdp_cur = 0
    npos = 0
    natt = 0
    n9dofc = 0
    nvel = 0
    ndp = 0

    dtpos=0
    dtpos_max=0.0
    dtpos_min=sys.float_info.max
    dtatt = 0
    dtatt_max=0.0
    dtatt_min=sys.float_info.max
    dt9dofc=0.0
    dt9dofc_max=0.0
    dt9dofc_min=sys.float_info.max
    dtvel=0
    dtvel_max=0.0
    dtvel_min=sys.float_info.max
    dtdp=0
    dtdp_max=0.0
    dtdp_min=sys.float_info.max

    lat=[]
    lon=[]
    alt=[]
    roll=[]
    pitch=[]
    yaw=[]
    mx=[]
    my=[]
    mz=[]
    ax=[]
    ay=[]
    az=[]
    gx=[]
    gy=[]
    gz=[]
    sog=[]
    cog=[]
    depth=[]

    while True:
        line = file.readline().strip().split(',')
        if not line or len(line) == 1:
            break

        t = int(line[0])
        if tend >= t:
            break
        tend = t

        tpos_cur = int(line[itpos]) - torg
        tatt_cur = int(line[itatt]) - torg
        t9dofc_cur = int(line[it9dofc]) - torg
        tvel_cur = int(line[itvel]) - torg
        tdp_cur = int(line[itdp]) - torg
        if tpos_cur > tpos_prev:
            dt = float(tpos_cur - tpos_prev) / 10000000.
            dtpos_max = max(dtpos_max, dt)
            dtpos_min = min(dtpos_min, dt)
            dtpos += dt
            tpos_prev = tpos_cur
            tpos.append(float(tpos_cur) / 10000000.)
            lat.append(float(line[ilat]))
            lon.append(float(line[ilon]))
            alt.append(float(line[ialt]))
            npos += 1
        if tatt_cur > tatt_prev:
            dt = float(tatt_cur - tatt_prev) / 10000000.
            dtatt_max = max(dtatt_max, dt)
            dtatt_min = min(dtatt_min, dt)
            dtatt += dt
            tatt_prev = tatt_cur
            tatt.append(float(tatt_cur) / 10000000.)
            roll.append(float(line[iroll]))
            pitch.append(float(line[ipitch]))
            yaw.append(float(line[iyaw]))
            natt += 1

        if t9dofc_cur > t9dofc_prev:
            dt = float(t9dofc_cur - t9dofc_prev) / 10000000.
            dt9dofc_max = max(dt9dofc_max, dt)
            dt9dofc_min = min(dt9dofc_min, dt)
            dt9dofc += dt
            t9dofc.append(float(t9dofc_cur) / 10000000.)
            mx.append(float(line[imx]))
            my.append(float(line[imy]))
            mz.append(float(line[imz]))
            ax.append(float(line[iax]))
            ay.append(float(line[iay]))
            az.append(float(line[iaz]))
            gx.append(float(line[igx]))
            gy.append(float(line[igy]))
            gz.append(float(line[igz]))
            t9dofc_prev = t9dofc_cur
            n9dofc += 1

        if tvel_cur > tvel_prev:
            dt = float(tvel_cur - tvel_prev) / 10000000.
            dtvel_max = max(dtvel_max, dt)
            dtvel_min = min(dtvel_min, dt)
            dtvel += dt
            tvel_prev = tvel_cur
            tvel.append(float(tvel_cur) / 10000000.)
            cog.append(float(line[icog]))
            sog.append(float(line[isog]))
            nvel += 1

        if tdp_cur > tdp_prev:
            dt = float(tdp_cur - tdp_prev) / 10000000.
            dtdp_max = max(dtdp_max, dt)
            dtdp_min = min(dtdp_min, dt)
            dtdp += dt
            tdp_prev = tdp_cur
            tdp.append(float(tdp_cur) / 10000000.)
            depth.append(float(line[idepth]))
            ndp += 1

    tpos = np.array(tpos)
    tatt = np.array(tatt)
    tvel = np.array(tvel)
    t9dofc = np.array(t9dofc)
    tdp = np.array(tdp)
    lat = np.array(lat)
    lon = np.array(lon)
    alt = np.array(alt)
    roll = np.array(roll)
    droll = diffAWS1DataVec(tatt, roll)
    pitch = np.array(pitch)
    dpitch = diffAWS1DataVec(tatt, pitch)
    yaw = np.array(yaw)
    dyaw = diffAWS1DataYaw(tatt, yaw)
    mx = np.array(mx)
    my = np.array(my)
    mz = np.array(mz)
    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    gx = np.array(gx)
    gy = np.array(gy)
    gz = np.array(gz)
    cog = np.array(cog)
    dcog = diffAWS1DataCog(tvel, cog)
    sog = np.array(sog)
    depth = np.array(depth)
    file.close()

    return (
        {'t':tpos, 'lat':lat, 'lon':lon, 'alt':alt},
        {'t':tvel, 'cog':cog, 'sog':sog, 'dcog':dcog}, 
        {'t':tdp, 'depth':depth},
        {'t':tatt, 'roll':roll, 'pitch':pitch, 'yaw':yaw, 'droll':droll, 'dpitch':dpitch, 'dyaw':dyaw},
        {'t':t9dofc, 'mx':mx, 'my':my, 'mz':mz, 'ax':ax, 'ay':ay, 'az':az, 'gx':gx, 'gy':gy, 'gz':gz})
        
def loadAWS1CtrlStat(fname, log_time):
    print("Analyzing " + fname)
    # t: rud, meng, seng
    file = open(fname)
    header=file.readline().strip().replace(" ", "").split(',')
    irecord = 0
    it = 0
    irud = 0
    imeng = 0
    iseng = 0

    for record in header:
        if(record == "t"):
            it = irecord
        if(record == "rud"):
            irud = irecord
        if(record == "meng"):
            imeng = irecord
        if(record == "seng"):
            iseng = irecord
        irecord+=1
    torg = log_time
    tend = 0
    tprev = 0
    tcur = 0
    dtavg = 0
    dtmin = sys.float_info.max
    dtmax = 0
    t=[]
    rud=[]
    meng=[]
    seng=[]
    n = 0

    while True:
        line = file.readline().strip().split(',')
        if not line or len(line) == 1:
            break
        ttmp = int(line[it])
        if tend >= ttmp:
            break

        tend = ttmp

        tcur = int(line[it]) - torg
        if(tcur > tprev):
            dt = float(tcur - tprev) / 10000000.
            dtmax = max(dtmax, dt)
            dtmin = min(dtmin, dt)
            dtavg += dt
            tprev = tcur
            t.append(float(tcur) / 10000000.)
            meng.append(int(line[imeng]))
            seng.append(int(line[iseng]))
            rud.append(int(line[irud]))
            n += 1
    dtavg /= float(n)

    t = np.array(t)
    meng = np.array(meng)
    seng = np.array(seng)
    rud = np.array(rud)

    return {'t':t, 'meng':meng, 'seng':seng, 'rud':rud}

def loadAWS1CtrlInst(fname, log_time):
    # t: acs, rud, meng, seng
    print("Analyzing " + fname)
    file = open(fname)
    header=file.readline().strip().replace(" ", "").split(',')
    irecord = 0
    it = 0
    iacs = 0
    irud = 0
    imeng = 0
    iseng = 0

    for record in header:
        if(record == "t"):
            it = irecord
        if(record == "acs"):
            iacs = irecord
        if(record == "rud"):
            irud = irecord
        if(record == "meng"):
            imeng = irecord
        if(record == "seng"):
            iseng = irecord
        irecord+=1
    torg = log_time
    tend = 0
    tprev = 0
    tcur = 0
    dtavg = 0
    dtmin = sys.float_info.max
    dtmax = 0
    t = []
    acs=[]
    rud=[]
    meng=[]
    seng=[]
    n = 0

    while True:
        line = file.readline().strip().split(',')
        if not line or len(line) == 1:
            break

        ttmp = int(line[it])
        if tend >= ttmp:
            break;
        tend = ttmp

        tcur = int(line[it]) - torg
        if(tcur > tprev):
            dt = float(tcur - tprev) / 10000000.
            dtmax = max(dtmax, dt)
            dtmin = min(dtmin, dt)
            dtavg += dt
            tprev = tcur
            t.append(float(tcur) / 10000000.)
            acs.append(int(line[iacs]))
            meng.append(int(line[imeng]))
            seng.append(int(line[iseng]))
            rud.append(int(line[irud]))
            n += 1
    dtavg /= float(n)
    
    t = np.array(t)
    acs = np.array(acs)
    meng = np.array(meng)
    seng = np.array(seng)
    rud = np.array(rud)

    file.close()
    return {'t':t, 'acs':acs, 'meng':meng, 'seng':seng, 'rud':rud}


if __name__ == '__main__':
    #loadAWS1LogFiles("/mnt/c/cygwin64/home/yhmtm/aws/log")
    log = AWS1Log()
    awspath="/mnt/d/aws"
    #awspath="/mnt/c/cygwin64/home/yhmtm/aws"
    log_time = selectAWS1Log(awspath+"/log")
    log_time = log.load(awspath+"/log", log_time)
    plot_dir=("/plot_%d" % log_time)
    log.stat(0,10000, awspath+plot_dir)
    log.plot(0,550, awspath+plot_dir)
    log.play(0,10000)
