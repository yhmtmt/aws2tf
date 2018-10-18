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
import ldAWS1Log as ldl

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
par_stvel=['cog','sog', 'dcog']
str_stvel=[["Course Over Ground","Deg"], ["Speed Over Ground", "kts"], ["Rate of Cource Change", "deg/s"]]
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

    def stat(self, ts=0.0, te=sys.float_info.max):
        lapinst = ldl.listAWS1DataSection(par_cinst, self.apinst)
        tapinst = self.apinst['t']
        iapinst = ldl.seekAWS1LogTime(tapinst, ts)
        iapinstf = ldl.seekAWS1LogTime(tapinst, te)
        luiinst = ldl.listAWS1DataSection(par_cinst, self.uiinst)
        tuiinst = self.uiinst['t']
        iuiinst = ldl.seekAWS1LogTime(tuiinst, ts)
        iuiinstf = ldl.seekAWS1LogTime(tuiinst, te)
        lctrlst = ldl.listAWS1DataSection(par_cstat, self.ctrlst)
        tctrlst = self.ctrlst['t']
        ictrlst = ldl.seekAWS1LogTime(tctrlst, ts)
        ictrlstf = ldl.seekAWS1LogTime(tctrlst, te)
        lstpos = ldl.listAWS1DataSection(par_stpos, self.stpos)
        tstpos = self.stpos['t']
        istpos = ldl.seekAWS1LogTime(tstpos, ts)
        istposf = ldl.seekAWS1LogTime(tstpos, te)
        lstvel = ldl.listAWS1DataSection(par_stvel, self.stvel)
        tstvel = self.stvel['t']
        istvel = ldl.seekAWS1LogTime(tstvel, ts)
        istvelf = ldl.seekAWS1LogTime(tstvel, te)
        lstatt = ldl.listAWS1DataSection(par_statt, self.statt)
        tstatt = self.statt['t']
        istatt = ldl.seekAWS1LogTime(tstatt, ts)
        istattf = ldl.seekAWS1LogTime(tstatt, te)
        lst9dof = ldl.listAWS1DataSection(par_9dof, self.st9dof)
        tst9dof = self.st9dof['t']
        i9dof = ldl.seekAWS1LogTime(tst9dof, ts)
        i9doff = ldl.seekAWS1LogTime(tst9dof, te)
        lstdp = ldl.listAWS1DataSection(par_stdp, self.stdp)
        tstdp = self.stdp['t']
        istdp = ldl.seekAWS1LogTime(tstdp, ts)
        istdpf = ldl.seekAWS1LogTime(tstdp, te)
        lengr = ldl.listAWS1DataSection(par_engr, self.engr)
        tengr = self.engr['t']
        iengr = ldl.seekAWS1LogTime(tengr, ts)
        iengrf = ldl.seekAWS1LogTime(tengr, te)
        lengd = ldl.listAWS1DataSection(par_engd, self.engd)
        tengd = self.engd['t']
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

        print("STAT apinst")
        ldl.printTimeStat(tapinst)
        for key in par_cinst:
            ldl.printStat(key, self.apinst[key])

        print("STAT uiinst")
        ldl.printTimeStat(tuiinst)
        for key in par_cinst:
            ldl.printStat(key, self.uiinst[key])

        print("STAT ctrlstat")
        ldl.printTimeStat(tctrlst)
        for key in par_cstat:
            ldl.printStat(key, self.ctrlst[key])

        print("STAT stpos")
        ldl.printTimeStat(tstpos)
        for key in par_stpos:
            ldl.printStat(key, self.stpos[key])

        print("STAT stvel")
        ldl.printTimeStat(tstvel)
        for key in par_stvel:
            ldl.printStat(key, self.stvel[key])

        print("STAT statt")
        ldl.printTimeStat(tstatt)
        for key in par_statt:
            ldl.printStat(key, self.statt[key])

        print("STAT 9dof")
        ldl.printTimeStat(tst9dof)
        for key in par_9dof:
            ldl.printStat(key, self.st9dof[key])

        print("STAT stdp")
        ldl.printTimeStat(tstdp)
        for key in par_stdp:
            ldl.printStat(key, self.stdp[key])

        print("STAT engr")
        ldl.printTimeStat(tengr)
        for key in par_engr:
            ldl.printStat(key, self.engr[key])

        print("STAT engd")
        ldl.printTimeStat(tengd)
        for key in par_engd:
            ldl.printStat(key, self.engd[key])

        ftotal = integrateAWS1Data(tengd, self.engd['frate'])
        ftotal /= 3600.0
        print("Estimated fuel consumption: %f" % ftotal) 

        print("STAT strm")
        ldl.printTimeStat(tstrm)

    def play(self, ts, te, dt=0.1):
        # seek head for all data section
        lapinst = ldl.listAWS1DataSection(par_cinst, self.apinst)
        tapinst = self.apinst['t']
        iapinst = ldl.seekAWS1LogTime(tapinst, ts)
        luiinst = ldl.listAWS1DataSection(par_cinst, self.uiinst)
        tuiinst = self.uiinst['t']
        iuiinst = ldl.seekAWS1LogTime(tuiinst, ts)
        lctrlst = ldl.listAWS1DataSection(par_cstat, self.ctrlst)
        tctrlst = self.ctrlst['t']
        ictrlst = ldl.seekAWS1LogTime(tctrlst, ts)
        lstpos = ldl.listAWS1DataSection(par_stpos, self.stpos)
        tstpos = self.stpos['t']
        istpos = ldl.seekAWS1LogTime(tstpos, ts)
        lstvel = ldl.listAWS1DataSection(par_stvel, self.stvel)
        tstvel = self.stvel['t']
        istvel = ldl.seekAWS1LogTime(tstvel, ts)
        lstatt = ldl.listAWS1DataSection(par_statt, self.statt)
        tstatt = self.statt['t']
        istatt = ldl.seekAWS1LogTime(tstatt, ts)
        lst9dof = ldl.listAWS1DataSection(par_9dof, self.st9dof)
        tst9dof = self.st9dof['t']
        i9dof = ldl.seekAWS1LogTime(tst9dof, ts)
        lstdp = ldl.listAWS1DataSection(par_stdp, self.stdp)
        tstdp = self.stdp['t']
        istdp = ldl.seekAWS1LogTime(tstdp, ts)
        lengr = ldl.listAWS1DataSection(par_engr, self.engr)
        tengr = self.engr['t']
        iengr = ldl.seekAWS1LogTime(tengr, ts)
        lengd = ldl.listAWS1DataSection(par_engd, self.engd)
        tengd = self.engd['t']
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
        lapinst = ldl.listAWS1DataSection(par_cinst, self.apinst)
        tapinst = self.apinst['t']
        iapinst = ldl.seekAWS1LogTime(tapinst, ts)
        iapinstf = ldl.seekAWS1LogTime(tapinst, te)
        luiinst = ldl.listAWS1DataSection(par_cinst, self.uiinst)
        tuiinst = self.uiinst['t']
        iuiinst = ldl.seekAWS1LogTime(tuiinst, ts)
        iuiinstf = ldl.seekAWS1LogTime(tuiinst, te)
        lctrlst = ldl.listAWS1DataSection(par_cstat, self.ctrlst)
        tctrlst = self.ctrlst['t']
        ictrlst = ldl.seekAWS1LogTime(tctrlst, ts)
        ictrlstf = ldl.seekAWS1LogTime(tctrlst, te)
        lstpos = ldl.listAWS1DataSection(par_stpos, self.stpos)
        tstpos = self.stpos['t']
        istpos = ldl.seekAWS1LogTime(tstpos, ts)
        istposf = ldl.seekAWS1LogTime(tstpos, te)
        lstvel = ldl.listAWS1DataSection(par_stvel, self.stvel)
        tstvel = self.stvel['t']
        istvel = ldl.seekAWS1LogTime(tstvel, ts)
        istvelf = ldl.seekAWS1LogTime(tstvel, te)
        lstatt = ldl.listAWS1DataSection(par_statt, self.statt)
        tstatt = self.statt['t']
        istatt = ldl.seekAWS1LogTime(tstatt, ts)
        istattf = ldl.seekAWS1LogTime(tstatt, te)
        lst9dof = ldl.listAWS1DataSection(par_9dof, self.st9dof)
        tst9dof = self.st9dof['t']
        i9dof = ldl.seekAWS1LogTime(tst9dof, ts)
        i9doff = ldl.seekAWS1LogTime(tst9dof, te)
        lstdp = ldl.listAWS1DataSection(par_stdp, self.stdp)
        tstdp = self.stdp['t']
        istdp = ldl.seekAWS1LogTime(tstdp, ts)
        istdpf = ldl.seekAWS1LogTime(tstdp, te)
        lengr = ldl.listAWS1DataSection(par_engr, self.engr)
        tengr = self.engr['t']
        iengr = ldl.seekAWS1LogTime(tengr, ts)
        iengrf = ldl.seekAWS1LogTime(tengr, te)
        lengd = ldl.listAWS1DataSection(par_engd, self.engd)
        tengd = self.engd['t']
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
        trrud = ldl.findInRangeTimeRanges(tctrlst, lctrlst[2], 154, 100)
        trmeng = ldl.findStableTimeRanges(tctrlst, lctrlst[0], smgn=10.0, emgn=0.0, th=1.0)
        trng = ldl.intersectTimeRanges(trrud, trmeng)
        trng = ldl.intersectTimeRanges(trng, [[ts,te]])
        rx,ry = ldl.relateTimeRangeVecs(tctrlst, tengr, lctrlst[0], lengr[0], trng)
        plotAWS1DataRelation("meng", "rpm", str_cstat[0], str_engr[0], rx, ry)

        # sog/rpm, -3 < dcog < 3, 100 < rud < 154, 153 < meng < 255
        trcog = ldl.findInRangeTimeRanges(tstvel, lstvel[2], 3,-3)
        trmeng = ldl.findInRangeTimeRanges(tctrlst, lctrlst[0], 255, 150)
        trsog = ldl.findStableTimeRanges(tstvel, lstvel[1], smgn=1.0, emgn=0.0, th=1.0)
        trng = ldl.intersectTimeRanges(trrud, trcog)
        trng = ldl.intersectTimeRanges(trng, trsog)
        trng = ldl.intersectTimeRanges(trng, trmeng)
        trng = ldl.intersectTimeRanges(trng, [[ts,te]])
        rx,ry = ldl.relateTimeRangeVecs(tstvel, tengr, lstvel[1], lengr[0], trng)
        plotAWS1DataRelation("sog", "rpm", str_stvel[1], str_engr[0], rx, ry)

if __name__ == '__main__':
    #loadAWS1LogFiles("/mnt/c/cygwin64/home/yhmtm/aws/log")
    log = AWS1Log()
    awspath="/mnt/d/aws"
    #awspath="/mnt/c/cygwin64/home/yhmtm/aws"
    log_time = ldl.selectAWS1Log(awspath+"/log")
    log_time = log.load(awspath+"/log", log_time)
    plot_dir=("/plot_%d" % log_time)
    log.stat(0,10000, awspath+plot_dir)
    log.plot(0,550, awspath+plot_dir)
    log.play(0,10000)
