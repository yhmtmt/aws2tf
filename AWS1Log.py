import sys
import os
import pdb          # for debug
import re           # for regular expression
import subprocess   # for command execution 
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2
import ldAWS1Video as ldv

pdb.set_trace()

par_engr=['rpm','trim']
par_engd=['valt','temp','frate']
par_stpos=['lat','lon','alt']
par_stvel=['cog','sog', 'dcog']
par_stdp=['depth']
par_statt=['roll','pitch','yaw','droll', 'dpitch', 'dyaw']
par_9dof=['mx','my','mz','ax','ay','az','gx','gy','gz']
par_cstat=['meng','seng','rud']
par_cinst=['acs','meng','seng','rud']

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
        data=loadAWS1LogFiles(path_aws1_log, log_time)
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

    def stat(self):
        print("STAT apinst")
        printTimeStat(self.apinst['t'])
        for key in par_cinst:
            printStat(key, self.apinst[key])

        print("STAT uiinst")
        printTimeStat(self.uiinst['t'])
        for key in par_cinst:
            printStat(key, self.uiinst[key])

        print("STAT ctrlstat")
        printTimeStat(self.ctrlst['t'])
        for key in par_cstat:
            printStat(key, self.ctrlst[key])

        print("STAT stpos")
        printTimeStat(self.stpos['t'])
        for key in par_stpos:
            printStat(key, self.stpos[key])

        print("STAT stvel")
        printTimeStat(self.stvel['t'])
        for key in par_stvel:
            printStat(key, self.stvel[key])

        print("STAT statt")
        printTimeStat(self.statt['t'])
        for key in par_statt:
            printStat(key, self.statt[key])

        print("STAT 9dof")
        printTimeStat(self.st9dof['t'])
        for key in par_9dof:
            printStat(key, self.st9dof[key])

        print("STAT stdp")
        printTimeStat(self.stdp['t'])
        for key in par_stdp:
            printStat(key, self.stdp[key])

        print("STAT engr")
        printTimeStat(self.engr['t'])
        for key in par_engr:
            printStat(key, self.engr[key])

        print("STAT engd")
        printTimeStat(self.engd['t'])
        for key in par_engd:
            printStat(key, self.engd[key])

        ftotal = integrateAWS1Data(self.engd['t'], self.engd['frate'])
        ftotal /= 3600.0
        print("Estimated fuel consumption: %f" % ftotal) 

        print("STAT strm")
        printTimeStat(self.strm['t'])

    def play(self, ts, te):
        # seek head for all data section
        def printTimeHead(name, it, t):
            if t[1] == 0:
                print("%s t[<0]") 
            elif t[0] == t.shape[0] -1:
                print("%s t[>-1") 
            else:
                print("%s t[%d]=%f" % (name, it[0], t[it[0]]))
        
        tapinst = self.apinst['t']
        iapinst = seekAWS1LogTime(tapinst, ts)
        tuiinst = self.uiinst['t']
        iuiinst = seekAWS1LogTime(tuiinst, ts)
        tctrlst = self.ctrlst['t']
        ictrlst = seekAWS1LogTime(tctrlst, ts)
        tstpos = self.stpos['t']
        istpos = seekAWS1LogTime(tstpos, ts)
        tstvel = self.stvel['t']
        istvel = seekAWS1LogTime(tstvel, ts)
        tstatt = self.statt['t']
        istatt = seekAWS1LogTime(tstatt, ts)
        tst9dof = self.st9dof['t']
        i9dof = seekAWS1LogTime(tst9dof, ts)
        tstdp = self.stdp['t']
        istdp = seekAWS1LogTime(tstdp, ts)
        tengr = self.engr['t']
        iengr = seekAWS1LogTime(tengr, ts)
        tengd = self.engd['t']
        iengd = seekAWS1LogTime(tengd, ts) 
        tstrm = self.strm['t']
        istrm = seekAWS1LogTime(tstrm, ts)

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

def seekAWS1LogTime(tseq,tseek):
    iend=tseq.shape[0]-1
    if(tseek > tseq[-1]):
        return iend, iend
    elif(tseek < tseq[0]):
        return 0, 0

    i=tseq.shape[0]/2
    imin = 0
    imax = iend

    while True:
        if (tseq[i] <= tseek): 
            if(tseq[i+1] > tseek):
                break
            else:
                imin = i
                i += (imax - i) / 2
        else:
            imax = i
            i -= min(1, (i - imin) / 2)

    return i,i+1


def loadAWS1LogFiles(path_aws1_log, log_time=-1): 
    ##### Select log file (in aws time)
    if (log_time == -1):
        command=['ls', path_aws1_log]    
        files=subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
        logs = re.findall("[0-9]{17}", files)
        ilog = 0
        for log in logs:
            command=['t2str', log]
            str_log_time = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
            print(("%d:"%ilog)+log + ":" + str_log_time)
            ilog = ilog + 1
        print("Select log number:")
        str_log_number=sys.stdin.readline()
        log_number=int(str_log_number)
        print("log %d : %s is selected." % (log_number, logs[log_number]))
        log_time = long(logs[log_number])
    path_log= "%s/%d"%(path_aws1_log,log_time)
    print("Check directory.")

    ##### Check channel files
    channels=["ais_obj", "aws1_ctrl_ap1", "aws1_ctrl_stat", "aws1_ctrl_ui", "engstate", "state"]
    chantypes=["ais_obj", "aws1_ctrl_inst", "aws1_ctrl_stat", "aws1_ctrl_inst", "engstate", "state"]

    if(os.path.isdir(path_log)):
        command = ["ls", path_log]
        files = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
        jrs = re.findall(".+.jr", files)
        found=False
        for chan in channels:
            for jr in jrs:
                if jr == chan+".jr":
                    found=True
                    break
            if found:
                print(chan+".jr found.")

            else:
                print(chan+".jr not found.")
                return
    
    ##### Convert .log to .txt 
    command = ["ls", path_log]
    files = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
    logs_bin = re.findall(".+.log", files)
    logs_txt = re.findall(".+.txt", files)
    for log_bin in logs_bin:
        name_bin,ext = os.path.splitext(log_bin)
        chan_log = None
        chan_type = None
        for ichan in range(len(channels)):            
            if(re.match(channels[ichan], name_bin)):
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
    logs_txt = re.findall(".+.txt", files)

    chans_logs = {}

    for chan in channels:
        chan_logs = []

        for log_txt in logs_txt:
            if (re.match(chan, log_txt)):
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
        apinst.append(loadAWS1CtrlInst(path_log+"/"+log, log_time))

    apinst = concatSectionData(apinst)

    #ctrl ui
    for log in chans_logs[channels[3]]:
        uiinst.append(loadAWS1CtrlInst(path_log+"/"+log, log_time))

    uiinst = concatSectionData(uiinst)
    
    #ctrl stat
    for log in chans_logs[channels[2]]:
        ctrlst.append(loadAWS1CtrlStat(path_log+"/"+log, log_time))

    ctrlst = concatSectionData(ctrlst)

    #engstate
    for log in chans_logs[channels[4]]:
        rapid,dynamic=loadAWS1Engstate(path_log+"/"+log, log_time)
        engr.append(rapid)
        engd.append(dynamic)

    engr = concatSectionData(engr)
    engd = concatSectionData(engd)
    
    #state
    for log in chans_logs[channels[5]]:
        pos,vel,dp,att,s9dof=loadAWS1State(path_log+"/"+log, log_time)
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
        'engr':engr, 'engd':engd, 'strm':{'t':tstrm, 'strm':strm}}

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

        t = long(line[0])  
        if tend >= t:
            break
        tend = t

        trapid_cur = long(line[itrapid]) - torg
        tdyn_cur = long(line[itdyn]) - torg
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

        t = long(line[0])
        if tend >= t:
            break
        tend = t

        tpos_cur = long(line[itpos]) - torg
        tatt_cur = long(line[itatt]) - torg
        t9dofc_cur = long(line[it9dofc]) - torg
        tvel_cur = long(line[itvel]) - torg
        tdp_cur = long(line[itdp]) - torg
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
            if natt == 57947:
                print("break")
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
            gx.append(float(line[igz]))
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
        ttmp = long(line[it])
        if tend >= ttmp:
            break

        tend = ttmp

        tcur = long(line[it]) - torg
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

        ttmp = long(line[it])
        if tend >= ttmp:
            break;
        tend = ttmp

        tcur = long(line[it]) - torg
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

#loadAWS1LogFiles("/mnt/c/cygwin64/home/yhmtm/aws/log")
log = AWS1Log()
log.load("/mnt/c/cygwin64/home/yhmtm/aws/log")
log.stat()
log.play(100,200)
