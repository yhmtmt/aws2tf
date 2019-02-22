import sys
import os
import re
import subprocess
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ldAWS1Video as ldv
import Opt as opt
from scipy import signal

savgol_att=False
savgol_window_att=5
savgol_dim_att=2
savgol_vel=False
savgol_window_vel=5
savgol_dim_vel=2
savgol_rpm=False
savgol_window_rpm=5
savgol_dim_rpm=2

chantypes=["ais_obj", "aws1_ctrl_inst", "aws1_ctrl_stat", "aws1_ctrl_inst", "engstate", "state"]
channels=["ais_obj", "aws1_ctrl_ap1", "aws1_ctrl_stat", "aws1_ctrl_ui", "engstate", "state"]


def calcTimeStat(tvec):
    dtvec = diffData(tvec)
    return calcStat(dtvec)

def printTimeStat(tvec):
    ttotal = tvec[-1] - tvec[0]
    dtavg, dtmax, dtmin, dtstd = calcTimeStat(tvec)
    print("Total time %f  Time step min: %f avg: %f max: %f std: %f" % (ttotal, dtmin, dtavg, dtmax, dtstd)) 

def calcStat(vec):
    if len(vec) != 0:
        return np.average(vec), np.max(vec), np.min(vec), np.std(vec)
    else:
        return 0,0,0,0

def printStat(vname, vec):
    vavg, vmax, vmin, vstd = calcStat(vec)
    print("%s max: %f min: %f avg: %f std: %f" % (vname, vmax, vmin, vavg, vstd))

def saveStat(file, vname, vec):
    vavg, vmax, vmin, vstd = calcStat(vec)
    str="%s, %f, %f, %f, %f\n" % (vname, vmax, vmin, vavg, vstd)
    file.write(str)

def saveStatGiven(file, vname, vmax, vmin, vavg, vstd):
    str="%s, %f, %f, %f, %f\n" % (vname, vmax, vmin, vavg, vstd)
    file.write(str)
    
def diffData(vec):
    ''' Calculates difference of each subsequent data. '''
    vnew=np.empty(shape=(vec.shape[0]-1), dtype=vec.dtype)
    for i in range(vnew.shape[0]):
        vnew[i] = vec[i+1] - vec[i]
    return vnew

def diffDataVec(t, vec):
    ''' Calculates time derivative of the sequence.'''
    dvec=np.zeros(shape=(vec.shape), dtype=vec.dtype)
    for i in range(1, dvec.shape[0]-1):
        dt0 = t[i]-t[i-1]
        dt1 = t[i+1]-t[i]
        idt = 1.0 / (dt0 + dt1)
        dvec[i] = idt * (vec[i+1] - vec[i-1])
    return dvec

def diffDataYaw(t, yaw):
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

def diffDataCog(t, cog):
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

def integrateData(t, vec):
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

def sampleMaxDistPoints(nsmpl, vecs):
    # calculate normalization factor
    vmax=[]
    vmin=[]
    ntotal = 0
    ndim = len(vecs)
    for vec in vecs:
        if len(vec) != 0:
            vmax.append(np.max(vec))
            vmin.append(np.min(vec))
            ntotal = len(vec)

    if ntotal == 0:
        return None
    
    vfac=[]
    for ivec in range(ndim):
        vfac.append(1.0/(vmax[ivec] - vmin[ivec]))

    vfac=np.array(vfac)
    dtable=np.zeros(shape=(nsmpl,nsmpl), dtype='float64')
    samples=np.zeros(shape=(nsmpl,len(vecs)), dtype='float64')
    
    # fill first nsmpl samples
    for ivec in range(len(vecs)):
        for ismple in range(nsmpl):
            samples[ismple][ivec] = vecs[ivec][ismple]
        
    # calculate distance table
    for i in range(0, nsmpl):
        for j in range(i, nsmpl):
            if i==j:
                dtable[i][i] =  0.0
            else:
                D = 0
                for k in range(ndim):
                    d=(samples[i][k]-samples[j][k]) * vfac[k]
                    D+=abs(d)
                dtable[i][j] = dtable[j][i] = D
                
    # forward sampling loop nsmpl to ntotal
    smpl = np.zeros(shape=(ndim), dtype='float64')
    dist = np.zeros(shape=(nsmpl), dtype='float64')
    ddiff = np.zeros(shape=(nsmpl), dtype='float64')

    # Only for debug
    def tot_dist(samples,vfac, ndim):
        d=0.0
        for k in range(len(samples)):
            for j in range(len(samples)):
                for i in range(ndim):
                    d+=abs(samples[j][i]-samples[k][i])*vfac[i]
        print(("Total distance: %f" % d))
        
    for ismpl in range(nsmpl, ntotal):
        for i in range(ndim):
            smpl[i] = vecs[i][ismpl]
            
        # calculate distances from new sample
        for jsmpl in range(nsmpl):
            dist[jsmpl]=0.0
            for i in range(ndim):
                d = abs(samples[jsmpl][i] - smpl[i]) * vfac[i]
                dist[jsmpl] += d

        # calculate distance differences by replacing
        jsmpl_max=-1
        ddiff_max=0.0
        for jsmpl in range(nsmpl):
            ddiff[jsmpl]=np.sum(dist)-np.sum(dtable[jsmpl])-dist[jsmpl]
            if(ddiff_max < ddiff[jsmpl]):
                ddiff_max = ddiff[jsmpl]
                jsmpl_max = jsmpl
            
        if(ddiff_max != 0.0):
            # replace sapmle
            for i in range(ndim):
                samples[jsmpl_max][i] = smpl[i]
            #tot_dist(samples, vfac, ndim)
            #updating dtable
            for i in range(nsmpl):
                if i == jsmpl_max:
                    dtable[i][i] = 0.0
                else:
                    dtable[jsmpl_max][i] = dtable[i][jsmpl_max] = dist[i]
    return samples
    
        
def getTimeRangeVecs(t, v, trng):
    x=[]
    for tr in trng:
        ix0s,ix0e = seekLogTime(t, tr[0])
        ix1s,ix1e = seekLogTime(t, tr[1])
        for ix in range(ix0e, ix1s):           
            x.append(v[ix])
    return np.array(x)

def relateTimeRangeVecs(tx, ty, vx, vy, trng):
    ''' 
       set of related points is calculated from two sets of time sequence:
       (tx, vx),(ty,vy).  Because both sequence could have different time
       intervals,  the values of y are linearly interpolated. 
    '''
    
    rx = []
    ry = []
    for tr in trng:
        ix0s,ix0e = seekLogTime(tx, tr[0])
        ix1s,ix1e = seekLogTime(tx, tr[1])
        for ix in range(ix0e, ix1s):
            iys,iye = seekLogTime(ty, tx[ix])
            if iys==iye: # out of range for y data.
                continue
            
            t = tx[ix]
            t0 = ty[iys]
            t1 = ty[iye]
            x = vx[ix]
            y = (vy[iye] * (t - t0) + vy[iys] * (t1 - t))  / (t1 - t0)
            rx.append(x)
            ry.append(y)
    return np.array(rx),np.array(ry)


def relateTimeRangeVecs3D(tx, ty, tz, vx, vy, vz, trng):
    ''' 
       set of related points is calculated from two sets of time sequence:
       (tx, vx),(ty,vy).  Because both sequence could have different time
       intervals,  the values of y are linearly interpolated. 
    '''
    
    rx = []
    ry = []
    rz = []
    for tr in trng:
        ix0s,ix0e = seekLogTime(tx, tr[0])
        ix1s,ix1e = seekLogTime(tx, tr[1])
        for ix in range(ix0e, ix1s):
            iys,iye = seekLogTime(ty, tx[ix])
            izs,ize = seekLogTime(tz, tx[ix])
            if iys!=iye and izs!=ize: # out of range for y data. 
                t = tx[ix]
                x = vx[ix]
                
                t0 = ty[iys]
                t1 = ty[iye]
                y = (vy[iye] * (t - t0) + vy[iys] * (t1 - t)) / (t1 - t0)
                
                t0 = tz[izs]
                t1 = tz[ize]
                z = (vz[ize] * (t - t0) + vz[izs] * (t1 - t)) / (t1 - t0)
                rx.append(x)
                ry.append(y)
                rz.append(z)
                
    return np.array(rx),np.array(ry),np.array(rz)

def findStableTimeRanges(t, vec, smgn=5.0, emgn=0.0, th=1.0, len_min=0):
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
                if (te - ts > emgn + smgn + len_min):
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
            if(ts >= 0):
                tranges.append([ts, te])
            ts = te = -1

    if(ts >= 0):
        te = t[-1]
        tranges.append([ts, te])
        ts = te = -1        
    return tranges

def seekLogTime(tseq,tseek):
    '''
    finds index of the time sequence tseq corresponding to the time tseek.
    the function returns two indices corresponding to before and after tseek.
    '''
    iend=tseq.shape[0]-1
    if iend < 0:
        return iend, iend
    if(tseek >= tseq[-1]):
        return iend, iend
    elif(tseek <= tseq[0]):
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
    done by seekLogTime.
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
        elif it[1] == t.shape[0] - 1:
            return it[1],it[1]
        elif t[it[1]] <= tnext and t[it[1]+1] > tnext:
            return [it[1],it[1]+1]
    return seekLogTime(t, tnext)

def printTimeHead(name, it, t):
    if it[1] == 0:
        print("%s t[<0]") 
    elif it[0] == t.shape[0] -1:
        print("%s t[>-1") 
    else:
        print("%s t[%d]=%f" % (name, it[0], t[it[0]]))

def listDataSection(keys, data):
    lst = []
    for key in keys:
        lst.append(data[key])
    return lst

def itpltDataVec(ldata, t, ts, it, angl=[]):
    '''
    gives linear interpoloation at time t for ldata. 
    (ldata is a list of time sequence along time given as ts)
    '''
    vec = np.zeros(shape=len(ldata),dtype=float)
    idt = 0
    for data in ldata:
        if it[1] < 0:
            vec[idt] = np.nan
            continue
        
        if it[1] == it[0]:
            vec[idt] = ldata[idt][it[0]]
            idt += 1
            continue         

        d0 = ldata[idt][it[0]]
        d1 = ldata[idt][it[1]]
        is_angle = len(angl) > idt and angl[idt]
        if is_angle:
            diff = d0 - d1
            if diff < -180.0:
                d1 -= 360.0
            elif diff > 180.0:
                d0 -= 360.0        
            
        t0 = ts[it[0]]
        t1 = ts[it[1]]
        vec[idt] = (d1 * (t - t0)  + d0 * (t1 - t)) / (t1 - t0)
        if is_angle:
            if vec[idt] < 0:
                vec[idt] += 360.0
            elif vec[idt] > 360.0:
                vec[idt] -= 360.0                
        idt += 1
    return vec

def printDataVec(name, keys, vdata):
    strRec = name
    for idt in range(len(keys)):
        strRec += " %s:%f," % (keys[idt], vdata[idt])
    print (strRec)

def loadListLogs(path_log, list_file):
    ''' load list in the file list_file '''
    file=open(list_file)
    logs=[]
    while True:
        log_time=file.readline().strip()
        if len(log_time) != 17:
            break
        
        path=path_log + "/" + log_time
        if(os.path.isdir(path)):
            logs.append(log_time)
        else:
            print("No such log: %s" % path)
    return logs
    
def listLogs(path_log):
    ''' list logs in the path_log '''
    command=['ls', path_log]
    files=subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
    logs = re.findall(rb"[0-9]{17}", files)
    if len(logs) == 0:
        return -1
    strlogs=[ log.decode('utf-8') for log in logs ]
    return strlogs

def selectLog(path_aws1_log, log_number=-1):
    ''' select single log by specifying the index '''
    logs=listLogs(path_aws1_log)
    if log_number == -1:
        printListLogs(logs)
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

def convTtoStr(t):
    ''' convert AWS time to string '''
    command=['t2str', t]
    str_log_time = subprocess.Popen(command, stdout=subprocess.PIPE).stdout.read()
    return str_log_time.decode('utf-8')

def printListLogs(logs):
    ''' print list of logs with its index and human-readable time '''
    ilog = 0
    for log in logs:
        str_log_time = convTtoStr(log)
        print(("%d:"%ilog)+log + ":" + str_log_time)
        ilog = ilog + 1

def loadLog(path_aws1_log, log_time=-1):
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

    tstrm,strm=ldv.loadVideoStream(path_log+"/mako0.avi",path_log+"/mako0.ts")

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
        apinst.append(loadCtrlInst(path_log+"/"+log.decode('utf-8'), log_time))

    apinst = concatSectionData(apinst)

    #ctrl ui
    for log in chans_logs[channels[3]]:
        uiinst.append(loadCtrlInst(path_log+"/"+log.decode('utf-8'), log_time))

    uiinst = concatSectionData(uiinst)
    
    #ctrl stat
    for log in chans_logs[channels[2]]:
        ctrlst.append(loadCtrlStat(path_log+"/"+log.decode('utf-8'), log_time))

    ctrlst = concatSectionData(ctrlst)

    #engstate
    for log in chans_logs[channels[4]]:
        rapid,dynamic=loadEngstate(path_log+"/"+log.decode('utf-8'), log_time)
        engr.append(rapid)
        engd.append(dynamic)

    engr = concatSectionData(engr)
    engd = concatSectionData(engd)
    
    #state
    for log in chans_logs[channels[5]]:
        pos,vel,dp,att,s9dof=loadState(path_log+"/"+log.decode('utf-8'), log_time)
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

def loadEngstate(fname, log_time):
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
    if savgol_rpm:
        rpm = signal.savgol_filter(rpm, savgol_window_rpm, savgol_dim_rpm, mode="mirror")
    trim = np.array(trim)
    valt = np.array(valt)
    temp = np.array(temp)
    frate = np.array(frate)

    file.close()
    return {'t':trapid, 'rpm':rpm, 'trim':trim} ,{'t':tdyn, 'valt':valt, 'temp':temp, 'frate':frate}

def loadState(fname, log_time):
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
        if tend > t:
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
    if savgol_att:
        roll = signal.savgol_filter(roll, savgol_window_att, savgol_dim_att, mode="mirror")
    droll = diffDataVec(tatt, roll)
    pitch = np.array(pitch)
    if savgol_att:
        pitch = signal.savgol_filter(pitch, savgol_window_att, savgol_dim_att, mode="mirror")
    dpitch = diffDataVec(tatt, pitch)
    yaw = np.array(yaw)
    if savgol_att:
        yaw = signal.savgol_filter(yaw, savgol_window_att, savgol_dim_att, mode="mirror")
    dyaw = diffDataYaw(tatt, yaw)
    byaw = np.zeros(yaw.shape[0])
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
    if savgol_vel:
        cog = signal.savgol_filter(cog, savgol_window_vel, savgol_dim_vel, mode="nearest")
    dcog = diffDataCog(tvel, cog)
    sog = np.array(sog)
    if savgol_vel:
        sog = signal.savgol_filter(sog, savgol_window_vel, savgol_dim_vel, mode="nearest")
    dsog = diffDataVec(tvel, sog)
    dsog = dsog * (1852.0/3600.0) # converting unit kts/sec into m/sec 
    depth = np.array(depth)
    file.close()

    return (
        {'t':tpos, 'lat':lat, 'lon':lon, 'alt':alt},
        {'t':tvel, 'cog':cog, 'sog':sog, 'dcog':dcog, 'dsog':dsog}, 
        {'t':tdp, 'depth':depth},
        {'t':tatt, 'roll':roll, 'pitch':pitch, 'yaw':yaw, 'droll':droll, 'dpitch':dpitch, 'dyaw':dyaw, 'byaw': byaw},
        {'t':t9dofc, 'mx':mx, 'my':my, 'mz':mz, 'ax':ax, 'ay':ay, 'az':az, 'gx':gx, 'gy':gy, 'gz':gz})
        
def loadCtrlStat(fname, log_time):
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

def loadCtrlInst(fname, log_time):
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


def loadStatCsv(fname):
    file=open(fname)
    header=file.readline().strip().replace(" ", "").split(',')
    irec=0
    ipar=0
    imax=0
    imin=0
    iavg=0
    idev=0
    for record in header:
        if(record == "name"):
            ipar=irec
        elif(record == "max"):
            imax=irec
        elif(record == "min"):
            imin=irec
        elif(record == "avg"):
            iavg=irec
        elif(record == "dev"):
            idev=irec
        irec+=1

    stat={}
        
    while True:
        line = file.readline().strip().split(',')
        if not line or len(line) == 1:
            break
        stat[line[ipar]]={"max":float(line[imax]), "min":float(line[imin]),
                          "avg":float(line[iavg]), "dev":float(line[idev])}
    return stat

def loadStatCsvs(path_result, logs, strpars):
    valss=[]
    for log_time in logs:
        fname=path_result+"/"+log_time+"/stat.csv"
        stat=loadStatCsv(fname)
        vals=[]
        for strpar in strpars:
            name_stat=strpar.split(".")
            if len(name_stat) == 1: #<name>
                if(name_stat[0]=="t"):
                    vals.append(int(log_time))
                    continue
                elif (name_stat[0]=="tstr"):
                    vals.append(convTtoStr(log_time))
                    continue                
                elif not(name_stat[0] in stat.keys()):
                    print ("No item named %s in stat.csv" % name_stat[0]) 
                    break
                
                for rec in stat[name_stat[0]].keys():
                    vals.append(stat[name_stat[0][rec]])
            elif len(name_stat) == 2: # <name>.<stat>
                if not(name_stat[0] in stat.keys()):
                    print ("No item named %s in stat.csv" % name_stat[0])
                    break
                if not(name_stat[1] in stat[name_stat[0]].keys()):
                    print ("No stat named %s in stat.csv" % name_stat[1])
                    break
                vals.append(stat[name_stat[0]][name_stat[1]])
            else:
                break;
        valss.append(vals)
    return valss
                            
def getListAndTime(par, data):
    l = listDataSection(par, data)
    t = data['t']
    return l,t

def getRelMengRpm(ts,te, tcinst, lcinst, tengr, lengr, terr=[[]]):
    # distinguish four state
    # foward(>152), up, down
    # backword(<102), up, down    
    # meng/rpm, 100 < rud < 154
    
    meng=lcinst[1]
    dir=np.zeros(shape=meng.shape,dtype='int8')
    meng_prev=meng[0]
    dir[0]=0
    for i in range(1, meng.shape[0]):
        if (meng_prev < meng[i]):
            dir[i]=1
        elif (meng_prev > meng[i]):
            dir[i]=-1
        else:
            dir[i]=dir[i-1]
        meng_prev = meng[i]
        
    tup = findInRangeTimeRanges(tcinst, dir, 1, 1)
    tdown = findInRangeTimeRanges(tcinst, dir, -1, -1)
    tf = findInRangeTimeRanges(tcinst, meng, 255, 152)
    tb = findInRangeTimeRanges(tcinst, meng, 102, 0)
            
    trrud = findInRangeTimeRanges(tcinst, lcinst[3], 154, 100)
    trmeng = findStableTimeRanges(tcinst, lcinst[1], smgn=20.0, emgn=0.0, th=1.0)
    
    trng = intersectTimeRanges(trrud, trmeng)
    trng = intersectTimeRanges(trng, [[ts,te]])
    trng = intersectTimeRanges(trng, terr)
    trngf = intersectTimeRanges(trng, tf)
    trngb = intersectTimeRanges(trng, tb)
    trngfup = intersectTimeRanges(trngf, tup)
    trngfdown = intersectTimeRanges(trngf, tdown)
    trngbup = intersectTimeRanges(trngb, tup)
    trngbdown = intersectTimeRanges(trngb, tdown)
    
    rxfup,ryfup = relateTimeRangeVecs(tcinst, tengr, lcinst[1], lengr[0], trngfup)
    rxfdown,ryfdown = relateTimeRangeVecs(tcinst, tengr, lcinst[1], lengr[0], trngfdown)
    
    rxbup,rybup = relateTimeRangeVecs(tcinst, tengr, lcinst[1], lengr[0], trngbup)
    rxbdown,rybdown = relateTimeRangeVecs(tcinst, tengr, lcinst[1], lengr[0], trngbdown)

    return rxfup,rxfdown,rxbup,rxbdown,ryfup,ryfdown,rybup,rybdown 


def getRelSogRpm(ts,te, tstvel, lstvel, tcinst, lctrlst, tengr, lengr, terr=[[]]):
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

def getRelun(ts,te, tmdl, lmdl, terr=[[]]):
    # sog/rpm, -3deg < r < 3deg, 100 < urud < 154, 152 < meng < 255
    # stable yaw and speed, 
    rad=math.pi / 180.0
    tr = findInRangeTimeRanges(tmdl, lmdl[8], 3*rad,-3*rad)
    trud = findInRangeTimeRanges(tmdl, lmdl[1], 154, 100) 
    tu = findStableTimeRanges(tmdl, lmdl[6], smgn=10.0, emgn=10.0, th=0.5)
    
    trng = intersectTimeRanges(tr, trud)
    trng = intersectTimeRanges(trng, tu)
    trng = intersectTimeRanges(trng, [[ts,te]])
    trng = intersectTimeRanges(trng, terr)
    rx,ry = relateTimeRangeVecs(tmdl, tmdl, lmdl[6], lmdl[9], trng)
    return rx,ry

def getRelSogRpmAcl(ts, te, tstvel, lstvel, tctrlst, lctrlst, tengr, lengr, terr=[[]]):
    trcog = findInRangeTimeRanges(tstvel, lstvel[2], 3,-3)
    trrud = findInRangeTimeRanges(tctrlst, lctrlst[2], 154, 100)
    trmeng = findInRangeTimeRanges(tctrlst, lctrlst[0], 255, 152)
    trng = intersectTimeRanges(trrud, trcog)
    trng = intersectTimeRanges(trng, trmeng)
    trng = intersectTimeRanges(trng, [[ts,te]])
    trng = intersectTimeRanges(trng, terr)
    rx,ry,rz= relateTimeRangeVecs3D(tstvel, tengr, tstvel, lstvel[1], lengr[0], lstvel[3], trng)
#    rx,ry = relateTimeRangeVecs(tstvel, tengr, lstvel[1], lengr[0], trng)    
#    rx,rz = relateTimeRangeVecs(tstvel, tstvel, lstvel[1], lstvel[3], trng)
    return rx, ry, rz #sog, rpm, acl

def getRelundu(ts, te, tmdl, lmdl, terr=[[]]):
    du=diffDataVec(tmdl, lmdl[6])
    rad=math.pi / 180.0
    tr = findInRangeTimeRanges(tmdl, lmdl[8], 3*rad,-3*rad)
    trud = findInRangeTimeRanges(tmdl, lmdl[1], 154, 100) 
    
    trng = intersectTimeRanges(tr, trud)
    trng = intersectTimeRanges(trng, [[ts,te]])
    trng = intersectTimeRanges(trng, terr)
    rx,ry,rz= relateTimeRangeVecs3D(tmdl, tmdl, tmdl, lmdl[6], lmdl[9],
                                    du, trng)
#    rx,ry = relateTimeRangeVecs(tstvel, tengr, lstvel[1], lengr[0], trng)    
#    rx,rz = relateTimeRangeVecs(tstvel, tstvel, lstvel[1], lstvel[3], trng)
    return rx, ry, rz #sog, rpm, acl


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

def getStableTurn(ts, te, tstvel, lstvel, tstatt, lstatt, tctrlst, lctrlst,
                  tmstate, lmstate, terr=[[]]):
    # at least 1 circle with constant sog, dcog, rud and meng, )
    # ths = 0.5 kts/sec, thy = 2.0 deg/sec
    tcmeng = findStableTimeRanges(tmstate, lmstate[0], smgn=0.0, emgn=0.0, th=1.0)
    trun = complementTimeRange(tmstate,
                               findInRangeTimeRanges(tmstate, lmstate[2], 0.99, -0.99));
    tcrud = findStableTimeRanges(tmstate, lmstate[1], smgn=0.0, emgn=0.0, th=1.0)
    tsog = findStableTimeRanges(tstvel, lstvel[1], smgn=0.0, emgn=0.0, th=2.0)
#    tdcog = findStableTimeRanges(tstvel, lstvel[2], smgn=0.0, emgn=0.0, th=1.0)
    trng = intersectTimeRanges(tcmeng, tcrud)
    trng = intersectTimeRanges(trng, trun)
    trng = intersectTimeRanges(trng, tsog)
#    trng = intersectTimeRanges(trng, tdcog)
    
    def normAngleDiff(angle):
        if(angle > 180.0):
            angle -= 360.0
        elif(angle < -180.0):
            angle += 360.0
        return angle

    turns=[]
    for tr in trng:
        istart = seekLogTime(tstvel, tr[0])
        iend = seekLogTime(tstvel, tr[1])
        # count rotation from istart[1] to iend[0]
        cog_start = cog_prev = lstvel[0][istart[1]]
    
        cog_turn_half = cog_start + 180.0
        if(cog_turn_half > 360.0):
            cog_turn_half -= 360.0
        cog_diff_half_prev = 180.0
        cog_diff_start_prev = 0.0
        
        ncircles = 0
        is_turned_half = False
        icircle_start = istart[1]
        icircle_end = iend[0]
        for icog in range(istart[1],iend[0]):
            cog_diff = normAngleDiff(lstvel[0][icog] - cog_prev)
            cog_diff_start = normAngleDiff(lstvel[0][icog] - cog_start)
            cog_diff_half = normAngleDiff(lstvel[0][icog] - cog_turn_half)
            if is_turned_half:                
                if (abs(cog_diff_start) < 90.0) and (cog_diff_start * cog_diff_start_prev < 0):
                    is_turned_half = False
                    icircle_end = icog
                    ncircles += 1
            else:
                if (abs(cog_diff_half)<90.0) and (cog_diff_half * cog_diff_half_prev < 0):
                    is_turned_half = True                
            cog_diff_half_prev = cog_diff_half
            cog_diff_start_prev = cog_diff_start
            cog_prev = lstvel[0][icog]

        if(ncircles == 0):
            continue
        
        period = (tstvel[icircle_end] - tstvel[icircle_start]) / ncircles
        dcogavg = np.average(lstvel[0][icircle_start:icircle_end])
        sogavg = np.average(lstvel[1][icircle_start:icircle_end])
        istart = seekLogTime(tmstate, tstvel[icircle_start])
        iend = seekLogTime(tmstate, tstvel[icircle_end])
        engavg = np.average(lmstate[0][istart[1]:iend[0]])
        rudavg = np.average(lmstate[1][istart[1]:iend[0]])
        uavg = np.average(lmstate[6][istart[1]:iend[0]])
        vavg = np.average(lmstate[7][istart[1]:iend[0]])
        ravg = np.average(lmstate[8][istart[1]:iend[0]])
        psiavg = np.average(lmstate[10][istart[1]:iend[0]])
        revavg = np.average(lmstate[9][istart[1]:iend[0]])
        drift = np.average(np.arctan2(lmstate[7][istart[1]:iend[0]],
                                      lmstate[6][istart[1]:iend[0]])) * (180.0 / math.pi)
        
        # v / r = 2 pi / T ->r = v T / 2pi
        radius = 0.5 * sogavg * (1852 / 3600) * period / math.pi
        turns.append([tstvel[icircle_start],tstvel[icircle_end],
                      period, radius, drift, uavg, vavg, ravg, psiavg, revavg,
                      sogavg, rudavg, engavg])
    # returns [tstart,tend,period,radius,drift,uavg,vavg,ravg,psiavg,rev,sog,rud,meng]

    return np.array(turns)

def saveStableTurn(path, turns):
    str="tstart,tend,period,radius,drift,u,v,r,psi,rev,sog,rud,meng"
    np.savetxt(path+"/turns.csv", turns, delimiter=',', header=str, fmt="%.2f")

def loadStableTurn(path):
    turns = np.loadtxt(path+"/turns.csv", delimiter=',')
    if turns.ndim==1 and turns.shape[0]==13:
        turns=[turns]        
    return turns

def get3DoFEqSt(u, du, n):
    '''
    An equation for 4 parameters below is generated
    "s dl_xu", "s dq_xu", "s kl", "s kq"
    '''
    Xkl = -u * abs(n)
    Xkq = -n * abs(n)
    eq=[u, u*abs(u), -Xkl, -Xkq]
    res=du
    return eq,du

def get3DoFEqN(u, du, v, dv, r, dr, psi, n, xr, yr,parXY):
    '''
    An equation for 5 parameters below is generated.
      "sIz - s ma_nr","s dl_nv", "s dl_nr","s dq_nv", "s dq_nr"
    '''
    # parXY is given as the following order.
    #   ["s m xg", "s m yg", "sm - s ma_yv", "s ma_nv", "s CL", "s CD", "s CTL", "s CTQ"]

    # build a set of stable turn equations    
    vabsv = v * abs(v)
    rabsr = r * abs(r)
    nabsn = n * abs(n)

    vr = v * r
    ur = u * r
    uv = u * v
    uu = u * u
     
    # (nrx, nry) radder direction vector
    nrx = math.cos(psi)
    nry = math.sin(psi)

    # (nrxp, nryp) the vector  perpendicular to (nrx, nry)
    nrxp = -nry
    nryp = nrx

    # (vrx, vry) velocity of rudder 
    vrx = u - yr * r
    vry = v + xr * r

    # dot product  (vrx, vry)^t (nrx, nry)
    vrnr = vrx * nrx + vry * nry

    # dot product (vrx, vry)^t (nrxp, nryp)
    vrnrp = vrx * nrxp + vry * nryp

    Ncl = 0.5 * vrnrp * (-yr * v + xr * yr * r + xr * u - xr * yr * r)
    Ncd = 0.5 * abs(vrnrp) * (-yr * yr * yr * r + xr +xr * xr * r)
    Nkl = -vrnr * abs(n) * (-yr * nrx + xr * nry)
    Nkq = -nabsn * (-yr * nrx +xr * nry)
    
    if vrnr < 0:
        Ncl = -Ncl

    coeff = [ur+dv, vr-du, -uv, -ur-dv, Ncl, Ncd, Nkl, Nkq]
    
    res = -uv
    for iterm in range(len(coeff)):
        res += coeff[iterm] * parXY[iterm]
        
    eq = [-dr, v, r, vabsv, rabsr]
    
    return eq,res

def get3DoFEqXYwithStPar(u,du,v,dv,r,dr,psi,n,
                         sdl_xu, sdq_xu, sCTL, sCTQ, xr,yr):
    '''
    two equations  for 15 parameters below (s means scale 1/(m-ma_xu))
    "s m xg", "s m yg", "sm - s ma_yv", "s ma_nv", "s dl_yv", 
    "s dl_yr", "s dq_yv", "s dq_yr", "s CL", "s CD", "s CTL", "s CTQ"
    ''' 
    # build a set of stable turn equations    
    uabsu = u * abs(u)
    vabsv = v * abs(v)
    rabsr = r * abs(r)
    nabsn = n * abs(n)

    vr = v * r
    ur = u * r
    rr = r * r
    
    # (nrx, nry) radder direction vector
    nrx = math.cos(psi)
    nry = math.sin(psi)

    # (nrxp, nryp) the vector  perpendicular to (nrx, nry)
    nrxp = -nry
    nryp = nrx

    # (vrx, vry) velocity of rudder 
    vrx = u - yr * r
    vry = v + xr * r

    # dot product  (vrx, vry)^t (nrx, nry)
    vrnr = vrx * nrx + vry * nry

    # dot product (vrx, vry)^t (nrxp, nryp)
    vrnrp = vrx * nrxp + vry * nryp
    
    Xcl = 0.5 * vrnrp * (- vry)
    Xcd = 0.5 * abs(vrnrp) * (vrx)
    Xkl = -vrnr * abs(n) * nrx
    Xkq = -nabsn * nrx
    Ycl = 0.5 * vrnrp * (vrx)
    Ycd = 0.5 * abs(vrnrp) * (vry)
    Ykl = -vrnr * abs(n) * nry
    Ykq = -nabsn * nry
    
    if vrnr < 0:
        Xcl = -Xcl        
        Ycl = -Ycl
        
    eq=[[-rr, -dr, vr, r, 0, 0, 0, 0, Xcl, Xcd],
        [dr, -rr, -dv, -dr, -u, -r, -vabsv, -rabsr, Ycl, Ycd]]
    res=[-du+u*sdl_xu+uabsu*sdq_xu-sCTL*Xkl-sCTQ*Xkq,
         -ur-sCTL*Ykl-sCTQ*Ykq]
    
    return eq,res
    
    

def get3DoFEqXY(u,du,v,dv,r,dr,psi,n,xr,yr):
    '''
    two equations  for 15 parameters below (s means scale 1/(m-ma_xu))
    "s m xg", "s m yg", "s m - s ma_yv", "s ma_nv", "s dl_xu", "s dl_yv", "s dl_yr", "s dq_xu", "s dq_yv", "s dq_yr", "s CL", "s CD", "s CTL", "s CTQ"
    ''' 
    # build a set of stable turn equations    
    uabsu = u * abs(u)
    vabsv = v * abs(v)
    rabsr = r * abs(r)
    nabsn = n * abs(n)

    vr = v * r
    ur = u * r
    rr = r * r
    
    # (nrx, nry) radder direction vector
    nrx = math.cos(psi)
    nry = math.sin(psi)

    # (nrxp, nryp) the vector  perpendicular to (nrx, nry)
    nrxp = -nry
    nryp = nrx

    # (vrx, vry) velocity of rudder 
    vrx = u - yr * r
    vry = v + xr * r

    # dot product  (vrx, vry)^t (nrx, nry)
    vrnr = vrx * nrx + vry * nry

    # dot product (vrx, vry)^t (nrxp, nryp)
    vrnrp = vrx * nrxp + vry * nryp
    
    Xcl = 0.5 * vrnrp * (- vry)
    Xcd = 0.5 * abs(vrnrp) * (vrx)
    Xkl = -vrnr * abs(n) * nrx
    Xkq = -nabsn * nrx
    Ycl = 0.5 * vrnrp * (vrx)
    Ycd = 0.5 * abs(vrnrp) * (vry)
    Ykl = -vrnr * abs(n) * nry
    Ykq = -nabsn * nry
    
    if vrnr < 0:
        Xcl = -Xcl        
        Ycl = -Ycl
        
    eq=[[-vr, -rr, -dr, vr, r, -u, 0, 0, -uabsu, 0, 0, Xcl, Xcd, Xkl, Xkq],
        [dv, dr, -rr, -dv, -dr, 0, -u, -r, 0, -vabsv, -rabsr, Ycl, Ycd, Ykl, Ykq]]
    res=[-du, -ur]
    return eq,res


def eval3DoFModel(idx, u,du,v,dv,r,dr,psi,n,m,xr,yr,par):
    '''
    calculates terms of M, C, Dl, Dq, CL, CD, KL, KQ and residual 
    '''
    
    def parStr(str):
        return ("%s%d" % (str, idx))
    
    xg = par[parStr("xg")]
    yg = par[parStr("yg")]
    Xdu = par[parStr("ma_xu")]
    Ydv = par[parStr("ma_yv")]
    Ndv = Ydr = par[parStr("ma_nv")]
    Ndr = par[parStr("ma_nr")]
    Xu = par[parStr("dl_xu")]
    Yv = par[parStr("dl_yv")]
    Yr = par[parStr("dl_yr")]
    Nv = par[parStr("dl_nv")]
    Nr = par[parStr("dl_nr")]
    Xuu = par[parStr("dq_xu")]
    Yvv = par[parStr("dq_yv")]
    Yrr = par[parStr("dq_yr")]
    Nvv = par[parStr("dq_nv")]
    Nrr = par[parStr("dq_nr")]
    CL  = par[parStr("CL")]
    CD = par[parStr("CD")]
    CTL = par[parStr("CTL")]
    CTQ = par[parStr("CTQ")]    
    
    Iz = m * (xg * xg  + yg * yg)
    Izdr = Iz * dr
    
    uabsu = u * abs(u)
    vabsv = v * abs(v)
    rabsr = r * abs(r)
    nabsn = n * abs(n)

    vr = v * r
    ur = u * r
    rr = r * r
    uv = u * v
    uu = u * u
    mrr = m * rr
    muu = m * uu
    muv = m * uv
    mur = m * ur
    mvr = m * vr
    mdr = m * dr
    mdu = m * du
    mdv = m * dv
    
    # (nrx, nry) radder direction vector
    nrx = math.cos(psi)
    nry = math.sin(psi)

    # (nrxp, nryp) the vector  perpendicular to (nrx, nry)
    nrxp = -nry
    nryp = nrx

    # (vrx, vry) velocity of rudder 
    vrx = u - yr * r
    vry = v + xr * r

    # dot product  (vrx, vry)^t (nrx, nry)
    vrnr = vrx * nrx + vry * nry

    # dot product (vrx, vry)^t (nrxp, nryp)
    vrnrp = vrx * nrxp + vry * nryp

    Xcl = 0.5 * vrnrp * (- vry)
    Xcd = 0.5 * abs(vrnrp) * (vrx)
    Xkl = -vrnr * abs(n) * nrx
    Xkq = -nabsn * nrx
    Ycl = 0.5 * vrnrp * (vrx)
    Ycd = 0.5 * abs(vrnrp) * (vry)
    Ykl = -vrnr * abs(n) * nry
    Ykq = -nabsn * nry    
    Ncl = 0.5 * vrnrp * (-yr * v + xr * yr * r + xr * u - xr * yr * r)
    Ncd = 0.5 * abs(vrnrp) * (-yr * yr * yr * r + xr +xr * xr * r)
    Nkl = -vrnr * abs(n) * (-yr * nrx + xr * nry)
    Nkq = -nabsn * (-yr * nrx +xr * nry)
    
    if vrnr < 0:
        Ncl = -Ncl
        Xcl = -Xcl
        Ycl = -Ycl
    
    Mx = (m - Xdu) * du - m * yg * dr
    Cx = -m * xg * rr - (m - Ydv) * vr + Ndv * r
    Dlx = -Xu * u
    Dqx = -Xuu * uabsu
    Clx = CL * Xcl
    Cdx = CD * Xcd
    Klx = CTL * Xkl
    Kqx = CTQ * Xkq
    
    My = (m - Ydv) * dv +(m * xg - Ydr) * dr
    Cy = - m * yg * rr  + (m - Xdu) * ur
    Dly = - Yv * u - Yr * r
    Dqy = - Yvv * vabsv - Yrr * rabsr
    Cly = CL * Ycl
    Cdy = CD * Ycd
    Kly = CTL * Ykl
    Kqy = CTQ * Ykq

    Mn = - m * yg * du + (m * xg - Ndv) * dv + (Iz - Ndr) * dr
    Cn = m * xg * ur + (m - Ydv) * uv - Ndv * ur + m * yg * vr - (m - Xdu) * uv
    Dln = - Nv * v - Nr * r
    Dqn = - Nvv * vabsv - Nrr * rabsr
    Cln = CL * Ncl
    Cdn = CD * Ncd
    Kln = CTL * Nkl
    Kqn = CTQ * Nkq
    
    forces=[[Mx, Cx, Dlx, Dqx, Clx, Cdx, Klx, Kqx],
            [My, Cy, Dly, Dqy, Cly, Cdy, Kly, Kqy],
            [Mn, Cn, Dln, Dqn, Cln, Cdn, Kln, Kqn]]
    res = [np.sum(np.array(forces[0])), np.sum(np.array(forces[1])), np.sum(np.array(forces[2]))]
    return forces, res

def getStableTurnEq(u, v, r, psi, n, m, xr, yr):
    '''
    Two equations for 15 parameters below
      "xg", "yg", "ma_xu", "ma_yv", "ma_nv", 
      "dl_xu", "dl_yv", "dl_yr", "dq_xu", "dq_yv", "dq_yr", 
      "CL", "CD", "CTL", "CTQ"
    '''
    # build a set of stable turn equations    
    uabsu = u * abs(u)
    vabsv = v * abs(v)
    rabsr = r * abs(r)
    nabsn = n * abs(n)

    vr = v * r
    ur = u * r
    uv = u * v
    uu = u * u
    mrr = m * r * r
    mur = m * ur
    mvr = m * vr

    # (nrx, nry) radder direction vector
    nrx = math.cos(psi)
    nry = math.sin(psi)

    # (nrxp, nryp) the vector  perpendicular to (nrx, nry)
    nrxp = -nry
    nryp = nrx

    # (vrx, vry) velocity of rudder 
    vrx = u - yr * r
    vry = v + xr * r

    # dot product  (vrx, vry)^t (nrx, nry)
    vrnr = vrx * nrx + vry * nry

    # dot product (vrx, vry)^t (nrxp, nryp)
    vrnrp = vrx * nrxp + vry * nryp
    
    Xcl = 0.5 * vrnrp * (- v - xr * r)
    Xcd = 0.5 * vrnrp * (u - yr * r)
    Xkl = -vrnr * n * nrx
    Xkq = -nabsn * nrx
    Ycl = 0.5 * vrnrp * (u - yr * r)
    Ycd = 0.5 * vrnrp * (v + xr * r)
    Ykl = -vrnr * n * nry
    Ykq = -nabsn * nrx
    Ncl = 0.5 * vrnrp * (-yr * v + xr * yr * r + xr * u - xr * yr * r)
    Ncd = 0.5 * vrnrp * (-yr * yr * yr * r + xr +xr * xr * r)
    Nkl = -vrnr * n * (-yr * nrx + xr * nry)
    Nkq = -nabsn * (-yr * nrx +xr * nry)
    
    if vrnr < 0:
        Xcl = -Xcl        
        Ycl = -Ycl
        Ncl = -Ncl

    eq=[[-mrr, 0, 0, vr, r, -u, 0, 0, 0, 0, -uabsu, 0, 0, 0, 0,
         Xcl, Xcd, Xkl, Xkq],
        [0, -mrr, ur, 0, 0, 0, -u, -r, 0, 0, 0, -vabsv, -rabsr, 0, 0,
         Ycl, Ycd, Ykl, Ykq],
        [mur, mvr, uu, -uv, -ur, 0, 0, 0, -v, -r, 0, 0, 0, -vabsv, -rabsr,
         Ncl, Ncd, Nkl, Nkq]]
    res=[mvr,mur,m *(-uv + uu)]
    return eq,res

def getStableStraightEq(u, n):
    uabsu = u * abs(u)
    nabsn = n * abs(n)
    un = u * n
    eq = np.array([0, 0, 0, 0, 0, -u, 0, 0, 0, 0,
                   -uabsu, 0, 0, 0, 0, 0, 0, un, nabsn])
    return eq

def estimateYawBias(ts, te, tstvel, lstvel, tstatt, lstatt, terr=[[]]):
    # for sog > th_sog, stable yaw and cog
    # calculate average and standard deviation
    tsog = findInRangeTimeRanges(tstvel, lstvel[1], 100, 5)
    tyaw = findStableTimeRanges(tstatt, lstatt[2], smgn=0, emgn=0, th=3, len_min=10)
    tcog = findStableTimeRanges(tstvel, lstvel[0], smgn=0, emgn=0, th=3, len_min = 10)
    trng = intersectTimeRanges(tsog, tyaw)
    trng = intersectTimeRanges(trng, tcog)
    trng = intersectTimeRanges(trng, terr)
    betas=[]
    for tr in trng:
        ivel_start = seekLogTime(tstvel, tr[0])
        ivel_end = seekLogTime(tstvel, tr[1])
        iatt = seekLogTime(tstatt, tstvel[ivel_start[1]])
        for ivel in range(ivel_start[1], ivel_end[0]):
            iatt = seekNextDataIndex(tstvel[ivel], iatt, tstatt)
            vatt = itpltDataVec(lstatt, tstvel[ivel], tstatt, iatt)
            beta = lstvel[0][ivel] - vatt[2]
            if(beta > 180):
                beta -= 360
            elif (beta < -180):
                beta += 360
            betas.append(beta)

    if(len(betas)==0):
        return 0,0,0,0
    
    betas = np.array(betas)
    drift_avg = np.average(betas)    
    drift_dev = np.std(betas)
    drift_min = np.min(betas)
    drift_max = np.max(betas)
    
    return drift_avg, drift_max, drift_min, drift_dev
        
def getErrorAtt(tstatt, lstatt):
    # No update found in attitude values
    trng = findStableTimeRanges(tstatt,lstatt[0],smgn=1.0, emgn=1.0, th=0.0)
    return complementTimeRange(tstatt, trng)

def plotDataSection(path, name, keys, str, ldata, ts, i0, i1):
    idt=0
    for key in keys:
        plt.plot(ts[i0:i1], ldata[idt][i0:i1])
        rel=np.c_[ts[i0:i1], ldata[idt][i0:i1]]
        ystr = str[idt][0] + " [" + str[idt][1] + "]"
        idt+=1
        figname=name+key+".png"
        plt.xlabel("Time [sec]")
        plt.ylabel(ystr)
        plt.savefig(path+"/"+figname)
        plt.clf()
        csvname=name+key+".csv"

        np.savetxt(path+"/"+csvname, rel, delimiter=',')

def selectData3D(pred, x, y, z):
    l=len(x)
    rx=[]
    ry=[]
    rz=[]
    for i in range(l):
        if(pred(x[i],y[i],z[i])):
            rx.append(x[i])
            ry.append(y[i])
            rz.append(z[i])
    return np.array(rx),np.array(ry),np.array(rz)

def plotDataRelation(path, name, parx, pary, strx, stry, rx, ry, density=False):
    figname=name+parx+pary+".png"
    if density:
        plt.hist2d(rx,ry, (50,50), cmap=plt.cm.jet)
        plt.colorbar()
    else:
        plt.scatter(rx,ry)
    plt.xlabel(strx[0]+" ["+strx[1]+"]")
    plt.ylabel(stry[0]+" ["+stry[1]+"]")
    plt.savefig(path+"/"+figname)
    plt.clf()
    
    csvname=name+parx+pary+".csv"
    rel=np.c_[rx,ry]
    np.savetxt(path+"/"+csvname, rel, delimiter=',')

def plotengrev(path, streng, strrev,
               engfup,engfdown,engbup,engbdown,
               revfup,revfdown,revbup,revbdown):
    parf=None
    parb=None

    if engfup.shape[0] > 0 or engfdown.shape[0] > 0:
        figname="mengrevf.png"
        if(engfup.shape[0] > 0):
            xmin=engfup.min()
            xmax=engfup.max()
            ymin=revfup.min()
            ymax=revfup.max()
        
        if(engfdown.shape[0] > 0):
            xmin=min(xmin,engfdown.min())
            xmax=max(xmax,engfdown.max())
            ymin=min(ymin,revfdown.min())
            ymax=max(ymax,revfdown.max())

        is_fitf = False
        if(engfup.shape[0] > 0 and engfdown.shape[0] > 0):
            diru=np.full(engfup.shape, 1)
            dird=np.full(engfdown.shape, -1)
            tmp0=np.stack((engfup,diru),axis=1)
            tmp1=np.stack((engfdown,dird),axis=1)
            eng=np.concatenate((tmp0,tmp1),axis=0)
            rev=np.concatenate((revfup,revfdown), axis=0)
            ropt=opt.fitengrevf(eng, rev)
            parf=ropt.x
            print("meng-rev ahead optimized parameters:")
            print(parf)
            saveParengrevf(path, parf) 
            is_fitf = True    
    
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.scatter(engfup, revfup, c='red')
        plt.scatter(engfdown, revfdown, c='blue')
        if (is_fitf):
            x = np.array([float(i) for i in range(int(xmin),int(xmax)+1)])    
            yu = np.array([opt.funcengrevf(parf, float(i), is_up=True) for i in range(int(xmin),int(xmax)+1)])
            yd = np.array([opt.funcengrevf(parf, float(i), is_up=False) for i in range(int(xmin),int(xmax)+1)])
            plt.plot(x,yu, label="fit up", color='r', linewidth=3)
            plt.plot(x,yd, label="fit down", color='b', linewidth=3)
        
        plt.xlabel(streng[0]+" ["+streng[1]+"]")
        plt.ylabel(strrev[0]+" ["+strrev[1]+"]")
        plt.savefig(path+"/"+figname)
        plt.clf()

    if engbup.shape[0] > 0 or engbdown.shape[0] > 0:
        figname="mengrevb.png"
        xmin=ymin=sys.float_info.max
        xmax=ymax=-sys.float_info.max
        if(engbup.shape[0] > 0):
            xmin=engbup.min()
            xmax=engbup.max()
            ymin=revbup.min()
            ymax=revbup.max()
        
        if(engbdown.shape[0] > 0):
            xmin=min(xmin,engbdown.min())
            xmax=max(xmax,engbdown.max())
            ymin=min(ymin,revbdown.min())
            ymax=max(ymax,revbdown.max())

        is_fitb = False
        if (engbup.shape[0] > 0 and engbdown.shape[0] > 0):
            diru=np.full(engbup.shape, 1)
            dird=np.full(engbdown.shape, -1)
            tmp0=np.stack((engbup,diru),axis=1)
            tmp1=np.stack((engbdown,dird),axis=1)
            eng=np.concatenate((tmp0,tmp1),axis=0)
            rev=np.concatenate((revbup,revbdown), axis=0)
            ropt=opt.fitengrevb(eng, rev)
            parb=ropt.x
            print("meng-rev astern optimized parameters:")
            print(parb)
            saveParengrevb(path, parb)
            is_fitb = True
    
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.scatter(engbup, revbup, c='red')
        plt.scatter(engbdown, revbdown, c='blue')
        if (is_fitb):
            x = np.array([float(i) for i in range(int(xmin),int(xmax)+1)])    
            yu = np.array([opt.funcengrevb(parb, float(i), is_up=True) for i in range(int(xmin),int(xmax)+1)])
            yd = np.array([opt.funcengrevb(parb, float(i), is_up=False) for i in range(int(xmin),int(xmax)+1)])
            plt.plot(x,yu, label="fit up", color='r', linewidth=3)
            plt.plot(x,yd, label="fit down", color='b', linewidth=3)
    
        plt.xlabel(streng[0]+" ["+streng[1]+"]")
        plt.ylabel(strrev[0]+" ["+strrev[1]+"]")
        plt.savefig(path+"/"+figname)
        plt.clf()
    

def plotun(path, strx, stry, rx, ry):
    # first remove ry=0 points
    _rx=[]
    _ry=[]

    for i in range(len(ry)):
        if ry[i] >= 0 and rx[i] < 0:
            continue
        if ry[i] <= 0 and rx[i] > 0:
            continue
        
        _rx.append(rx[i])
        _ry.append(ry[i])
            
    rx = np.array(_rx)
    ry = np.array(_ry)
    
    res=opt.fitun(rx, ry, par0=[700.0, 500, 300.0, 1000.0])
    par = res.x
    
    xmin= 0 if len(rx)==0 else np.min(rx)
    xmax= 0 if len(ry)==0 else np.max(rx)
    
    plt.scatter(rx, ry, label="data", alpha=0.3)
    x=np.array([float(i) for i in range(int(xmin-0.5),int(xmax +0.5))])
    y=np.array([opt.funcun(par, float(i)) for i in range(int(xmin-0.5),int(xmax +0.5))])
    plt.plot(x, y, label="fit", color='r', linewidth=3)
    plt.xlabel(strx[0]+" ["+strx[1]+"]")
    plt.ylabel(stry[0]+" ["+stry[1]+"]")  
    figfile="un.png"
    plt.savefig(path+"/"+figfile)
    plt.clf()
    
    saveRelun(path, rx, ry)
    saveParun(path, par)

def saveRelun(path, u, n):
    rel=np.c_[u,n]
    csvname="un.csv"
    np.savetxt(path+"/"+csvname, rel, delimiter=',')

def loadRelun(path):
    csvname="un.csv"
    rel = np.loadtxt(path+"/"+csvname, delimiter=',')
    return rel[:,0],rel[:,1]

def load_u_v_r_phi_n(path):
    header=path+"/"+"model_state"
    fname_u=header+"u.csv"
    fname_v=header+"v.csv"
    fname_r=header+"r.csv"
    fname_psi=header+"psi.csv"
    fname_n=header+"n.csv"
    u=np.loadtxt(fname_u, delimiter=',')
    t = u[:,0]
    u = u[:,1]
    v=np.loadtxt(fname_v, delimiter=',')
    v = v[:,1]
    r=np.loadtxt(fname_r, delimiter=',')
    r = r[:,1]
    psi=np.loadtxt(fname_psi, delimiter=',')
    psi = psi[:,1]
    n=np.loadtxt(fname_n, delimiter=',')
    n=n[:,1]
    
    return t,u,v,r,psi,n
    
def saveParun(path, par):
    csvname="parun.csv"
    np.savetxt(path+"/"+csvname, par, delimiter=',')
    
def loadParun(path):
    csvname="parun.csv"
    return np.loadtxt(path+"/"+csvname, delimiter=',')

def saveParengrevf(path, par):
    csvname="parengrevf.csv"
    np.savetxt(path+"/"+csvname, par,  delimiter=',')

def saveParengrevb(path, par):                          
    csvname="parengrevb.csv"
    np.savetxt(path+"/"+csvname, par, delimiter=',')

def loadParengrevf(path, par):
    csvname="parengrevf.csv"
    return np.loadtxt(path+"/"+csvname, delimiter=',')
                          
def loadParengrevb(path, par):                          
    csvname="parengrevb.csv"
    return np.loadtxt(path+"/"+csvname, delimiter=',')    

def plotundu(path, parx, pary, parz, strx, stry, strz, rx, ry, rz):
    par = loadParun(path)
    cu = opt.cu(par)
    def isAst(x, y, z):
        return x < 0    
    def isDis(x, y, z):
        return x <= cu
    def isPln(x, y, z):
        return x > cu
    
    plotDataRelation3D(path, "all-", parx, pary, parz, strx, stry, strz, rx, ry, rz)
    rxa,rya,rza = selectData3D(isAst, rx, ry, rz)
    rxd,ryd,rzd = selectData3D(isDis, rx, ry, rz)
    rxp,ryp,rzp = selectData3D(isPln, rx, ry, rz)
    plotDataRelation3D(path, "astern-", parx, pary, parz, strx, stry, strz, rxa, rya, rza)        
    plotDataRelation3D(path, "displacement-", parx, pary, parz, strx, stry, strz, rxd, ryd, rzd)    
    plotDataRelation3D(path, "planing-", parx, pary, parz, strx, stry, strz, rxp, ryp, rzp)

    rxa=np.array([rya[i] - opt.funcun(par, rxa[i]) for i in range(rxa.shape[0])])
    plotDataRelation(path, "astern-", "n_n_e", "acl",
                             ["Difference from stable point", stry[1]],
                             strz, rxa, rza)
    
    rxd=np.array([ryd[i] - opt.funcun(par, rxd[i]) for i in range(rxd.shape[0])])
    plotDataRelation(path, "displacement-", "n_n_e", "acl",
                             ["Difference from stable point", stry[1]],
                             strz, rxd, rzd)
    rxp=np.array([ryp[i] - opt.funcun(par, rxp[i]) for i in range(rxp.shape[0])])
    plotDataRelation(path, "planing-","n_n_e", "acl",
                             ["Difference from stable point ", stry[1]],
                             strz, rxp, rzp)    
    
def plotDataRelation3D(path, name, parx, pary, parz, strx, stry, strz, rx, ry, rz):
    figname=name+parx+pary+parz+".png"
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    
    ax.scatter(rx, ry, rz)
    ax.set_xlabel(strx[0]+" ["+strx[1]+"]")
    ax.set_ylabel(stry[0]+" ["+stry[1]+"]")
    ax.set_zlabel(strz[0]+" ["+strz[1]+"]")
    plt.savefig(path+"/"+figname)
    plt.clf()
    plt.close(fig)
    csvname=name+parx+pary+parz+".csv"
    rel=np.c_[rx,ry,rz]
    np.savetxt(path+"/"+csvname, rel, delimiter=',')
    
    
