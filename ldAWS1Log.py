import sys
import os
import re
import subprocess
import random
import numpy as np
import matplotlib.pyplot as plt
import ldAWS1Video as ldv

channels=["ais_obj", "aws1_ctrl_ap1", "aws1_ctrl_stat", "aws1_ctrl_ui", "engstate", "state"]
chantypes=["ais_obj", "aws1_ctrl_inst", "aws1_ctrl_stat", "aws1_ctrl_inst", "engstate", "state"]


def calcTimeStat(tvec):
    dtvec = diffAWS1Data(tvec)
    return calcStat(dtvec)

def printTimeStat(tvec):
    ttotal = tvec[-1] - tvec[0]
    dtmax, dtmin, dtavg, dtstd = calcTimeStat(tvec)
    print("Total time %f  Time step min: %f avg: %f max: %f std: %f" % (ttotal, dtmin, dtavg, dtmax, dtstd)) 

def calcStat(vec):
    if len(vec) != 0:
        return np.average(vec), np.max(vec), np.min(vec), np.std(vec)
    else:
        return np.nan,np.nan,np.nan,np.nan


def printStat(vname, vec):
    vmax, vmin, vavg, vstd = calcStat(vec)
    print("%s max: %f min: %f avg: %f std: %f" % (vname, vmax, vmin, vavg, vstd))

def saveStat(file, vname, vec):
    vmax, vmin, vavg, vstd = calcStat(vec)
    str="%s, %f, %f, %f, %f\n" % (vname, vmax, vmin, vavg, vstd)
    file.write(str)
    
def diffAWS1Data(vec):
    ''' Calculates difference of each subsequent data. '''
    vnew=np.empty(shape=(vec.shape[0]-1), dtype=vec.dtype)
    for i in range(vnew.shape[0]):
        vnew[i] = vec[i+1] - vec[i]
    return vnew

def diffAWS1DataVec(t, vec):
    ''' Calculates time derivative of the sequence.'''
    dvec=np.zeros(shape=(vec.shape), dtype=vec.dtype)
    for i in range(1, dvec.shape[0]-1):
        dt0 = t[i]-t[i-1]
        dt1 = t[i+1]-t[i]
        idt = 1.0 / (dt0 + dt1)
        dvec[i] = idt * (vec[i+1] - vec[i-1])
    return dvec

def diffAWS1DataYaw(t, yaw):
    ''' Calculates yaw rate from the time sequence of yaw.'''
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
    ''' Calculates course changing rate from the time sequence of cog'''
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
    ''' calculates time integral of the given sequence.'''
    if len(t) == 0:
        return 0
    
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

def complementTimeRange(t, trng):
    if len(t) == 0:
        return []
    
    tprev=t[0]
    trngc=[]
    for tr in trng:
        if tprev != tr[0]:
            trngc.append([tprev, tr[0]])
        tprev = tr[1]
    if tprev != t[-1]:
        trngc.append([tprev, t[-1]])
    return trngc

def intersectTimeRanges(trng0, trng1):
    ''' calculate intersection of the two sets of time ranges'''
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
    ''' 
       set of related points is calculated from two sets of time sequence:
       (tx, vx),(ty,vy).  Because both sequence could have different time
       intervals,  the values of y are linearly interpolated. 
    '''
    
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
    '''
        find stable ranges in given time sequence. smgn and emgn are 
        the start and end margins eliminated from the result. th is the 
        amplitude allowed in the stable ranges.
    '''
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
    '''
    find time ranges the values is in the range [vmin, vmax]
    '''
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
    '''
    finds index of the time sequence tseq corresponding to the time tseek.
    the function returns two indices corresponding to before and after tseek.
    '''
    iend=tseq.shape[0]-1
    if iend < 0:
        return iend, iend
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


def seekNextDataIndex(tnext, it, t):
    '''
    find next data index given a tnext. 
    if tnext is in the range headed by next time index, 
    it[1],it[1]+1 is returned, otherwise binary search is 
    done by seekAWS1LogTime.
    '''    
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

def printTimeHead(name, it, t):
    if it[1] == 0:
        print("%s t[<0]") 
    elif it[0] == t.shape[0] -1:
        print("%s t[>-1") 
    else:
        print("%s t[%d]=%f" % (name, it[0], t[it[0]]))

def listAWS1DataSection(keys, data):
    lst = []
    for key in keys:
        lst.append(data[key])
    return lst

def itpltAWS1DataVec(ldata, t, ts, it):
    '''
    gives linear interpoloation at time t for ldata. 
    (ldata is a list of time sequence along time given as ts)
    '''
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

def loadAWS1Logs(path_log, list_file):
    file=open(list_file)
    logs=[]
    while True:
        log_time=file.readline().strip()
        if len(log_time) != 17:
            break
        
        path=path_log + "/" + log_time
        if(os.path.isdir(path)):
            logs.append(log_time.encode('utf-8'))
        else:
            print("No such log: %s" % path)
    return logs

    
def listAWS1Logs(path_aws1_log):
    command=['ls', path_aws1_log]
    files=subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
    logs = re.findall(rb"[0-9]{17}", files)
    if len(logs) == 0:
        return -1
    return logs

def printAWS1Logs(logs):
    ilog = 0
    for log in logs:
        command=['t2str', log]
        str_log_time = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
        print(("%d:"%ilog)+log.decode('utf-8') + ":" + str_log_time.decode('utf-8'))
        ilog = ilog + 1
    

def selectAWS1Log(path_aws1_log, log_number=-1):
    logs=listAWS1Logs(path_aws1_log)
    if log_number == -1:
        printAWS1Logs(logs)
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
            path_log_bin = path_log+"/"+log_bin.decode('utf-8')
            command = ["log2txt", chan_type, path_log_bin]
            print("Converting " + log_bin.decode('utf-8') + " to text.")
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
    if n != 0:
        dtavg /= float(n)
    else:
        dtavg = 0

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
    if n != 0:
        dtavg /= float(n)
    else:
        dtavg = 0
        
    t = np.array(t)
    acs = np.array(acs)
    meng = np.array(meng)
    seng = np.array(seng)
    rud = np.array(rud)

    file.close()
    return {'t':t, 'acs':acs, 'meng':meng, 'seng':seng, 'rud':rud}

def getListAndTime(par, data):
    l = listAWS1DataSection(par, data)
    t = data['t']
    return l,t

def getRelMengRpm(ts,te, tctrlst, lctrlst, tengr, lengr, terr=[[]]):
    # meng/rpm, 100 < rud < 154
    trrud = findInRangeTimeRanges(tctrlst, lctrlst[2], 154, 100)
    trmeng = findStableTimeRanges(tctrlst, lctrlst[0], smgn=10.0, emgn=0.0, th=1.0)
    trng = intersectTimeRanges(trrud, trmeng)
    trng = intersectTimeRanges(trng, [[ts,te]])
    trng = intersectTimeRanges(trng, terr)
    rx,ry = relateTimeRangeVecs(tctrlst, tengr, lctrlst[0], lengr[0], trng)
    return rx,ry

def getRelSogRpm(ts,te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr=[[]]):
    # sog/rpm, -3 < dcog < 3, 100 < rud < 154, 152 < meng < 255
    trcog = findInRangeTimeRanges(tstvel, lstvel[2], 3,-3)
    trrud = findInRangeTimeRanges(tctrlst, lctrlst[2], 154, 100)
    trmeng = findInRangeTimeRanges(tctrlst, lctrlst[0], 255, 152)
    trsog = findStableTimeRanges(tstvel, lstvel[1], smgn=10.0, emgn=10.0, th=1.0)
    trng = intersectTimeRanges(trrud, trcog)
    trng = intersectTimeRanges(trng, trsog)
    trng = intersectTimeRanges(trng, trmeng)
    trng = intersectTimeRanges(trng, [[ts,te]])
    trng = intersectTimeRanges(trng, terr)
    rx,ry = relateTimeRangeVecs(tstvel, tengr, lstvel[1], lengr[0], trng)
    return rx,ry

def getRelFieldSogCog(ts,te, tstvel, lstvel, tctrlst, lctrlst, terr=[[]]):
    # sog/cog -3 < dcog < 3, 100 < rud < 154, 102 < meng < 152
    trdcog = findInRangeTimeRanges(tstvel, lstvel[2], 3,-3)
    trrud = findInRangeTimeRanges(tctrlst, lctrlst[2], 154, 100)
    trmeng = findInRangeTimeRanges(tctrlst, lctrlst[0], 150, 102)
    trsog = findStableTimeRanges(tstvel, lstvel[1], smgn=30.0, emgn=0.0, th=1.0)
    trcog = findStableTimeRanges(tstvel, lstvel[0], smgn=30.0, emgn=0.0, th=5.0)
    trng = intersectTimeRanges(trrud, trdcog)
    trng = intersectTimeRanges(trng, trsog)
    trng = intersectTimeRanges(trng, trcog)
    trng = intersectTimeRanges(trng, trmeng)
    trng = intersectTimeRanges(trng, [[ts,te]])
    trng = intersectTimeRanges(trng, terr)
    rx,ry = relateTimeRangeVecs(tstvel, tstvel, lstvel[1], lstvel[0], trng)
    return rx,ry

def getErrorAtt(tstatt, lstatt):
    # No update found in attitude values
    trng = findStableTimeRanges(tstatt,lstatt[0],smgn=1.0, emgn=1.0, th=0.0)
    return complementTimeRange(tstatt, trng)

def plotAWS1DataSection(path, keys, str, ldata, ts, i0, i1):
    idt=0
    for key in keys:
        plt.plot(ts[i0:i1], ldata[idt][i0:i1])
        rel=np.c_[ts[i0:i1], ldata[idt][i0:i1]]
        ystr = str[idt][0] + " [" + str[idt][1] + "]"
        idt+=1
        figname=key+".png"
        plt.xlabel("Time [sec]")
        plt.ylabel(ystr)
        plt.savefig(path+"/"+figname)
        plt.clf()
        csvname=key+".csv"

        np.savetxt(path+"/"+csvname, rel, delimiter=',')
        
def plotAWS1DataRelation(path, parx, pary, strx, stry, rx, ry):
    figname=parx+pary+".png"
    plt.scatter(rx,ry)
    plt.xlabel(strx[0]+" ["+strx[1]+"]")
    plt.ylabel(stry[0]+" ["+stry[1]+"]")
    plt.savefig(path+"/"+figname)
    plt.clf()
    
    csvname=parx+pary+".csv"
    rel=np.c_[rx,ry]
    np.savetxt(path+"/"+csvname, rel, delimiter=',')
