import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import subprocess
import pdb
import csv

pdb.set_trace()
def getAWS1DataAt(str_vec, fac_vec, data, idx):
    vec = [data[str_vec[ikey]][idx]*fac_vec[ikey] for ikey in range(len(str_vec))]
    return vec


def getAWS1BatchSection(str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, batch_size, input_length, predict_length, data, start=0.,end=1.):
    if(len(str_vec_in) == 0 or len(str_vec_out) == 0):
        return None, None
    len_data = len(data[str_vec_in[0]])
    istart = int(start*float(len_data))
    iend = int(end*float(len_data))
    rnum = [random.randint(istart,iend - predict_length - input_length) for i in range(batch_size)]
    xs = np.array([[getAWS1DataAt(str_vec_in, fac_vec_in, data, i) for i in range(r, r + input_length)] for r in rnum])
    ts = np.array([[getAWS1DataAt(str_vec_out, fac_vec_out, data, i) for i in range(r + input_length, r + input_length + predict_length)] for r in rnum])
    return xs, np.reshape(ts, (batch_size,-1))

def getAWS1BatchSectionSeq(str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, predict_length, data, ipos, start=0.,end=1.):
    if(len(str_vec_in) == 0 or len(str_vec_out) == 0):
        return None, None
    len_data = len(data[str_vec_in[0]])
    istart = max(ipos, int(start*float(len_data)))
    iend = int(end*float(len_data)) - predict_length - 1

    ismpl = ipos + istart
    if(ismpl >= iend):
        return None, None
        
    xs = np.array([[getAWS1DataAt(str_vec_in, fac_vec_in, data, ismpl)]])
    ts = np.array([[getAWS1DataAt(str_vec_out, fac_vec_out, data, i) for i in range(ismpl + 1, ismpl + 1 + predict_length)]])
    return xs, np.reshape(ts, (1,-1))

def loadAWS1DataList(fname, bstat=True):
    f = open(fname)
    data = f.read()
    fsizes = []
    fcsv_names = data.split('\n')

    for fcsv_name in fcsv_names:
        try:
            fcsv = open(fcsv_name)
            rcsv = csv.reader(fcsv, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

            data=[v for v in rcsv]
            fsizes.append(len(data))
        except IOError:
            fsizes.append(0)
    
    for ifile in range(len(fcsv_names)):
        if(fsizes[ifile]):
            print("File[%d] %s " % (ifile, fcsv_names[ifile]))
            stat=statAWS1Data(fcsv_names[ifile], bstat)

    return fcsv_names, fsizes


def getAWS1DataBatch(fcsv_names, str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, fsizes, batch_size, input_length, predict_length, eval=False):
    total_fsize = 0
    
    for fsize in fsizes:
        total_fsize += fsize
    
    nrnd = random.randint(0, total_fsize-1)

    total_fsize = 0
    ifile = 0
    for i in range(len(fsizes)):
        total_fsize +=fsizes[i]
        if(nrnd < total_fsize):
            ifile = i
            break

    str_vec = list(set(str_vec_in + str_vec_out))
    aws1_data = loadAWS1Data(fcsv_names[ifile],str_vec)
    
    section = random.randint(0, 1)
    if eval:
        section = 2 * section + 1
        print("Batch from %s section %d" % (fcsv_names[ifile], section))
    else:
        section = 2 * section

    xs, ts = getAWS1BatchSection(str_vec_in, fac_vec_in, str_vec_out, fac_vec_out, batch_size, input_length, predict_length, aws1_data, section * 0.25, min(1.0, (section + 1) * 0.25))

    return xs, ts
   

def statAWS1DataRec(rec):
    rmax = 0.0
    rmin = sys.float_info.max
    ravg = 0.0

    for i in range(len(rec)):
        ravg += rec[i]
        rmax = max(rmax, rec[i])
        rmin = min(rmin, rec[i])

    return rmax, rmin, ravg/float(len(rec))

def statAWS1Data(fname, bstat=True):
    str_vec=["t", "eng", "rud", "rev", "roll", "rroll", "pitch", "rpitch", "yaw", "ryaw", "cog", "rcog", "sog", "rsog"]
    data = loadAWS1Data(fname, str_vec)

    t=data["t"]
    duration=t[-1]

    dt=[t[i+1]-t[i] for i in range(len(t)-1)]
    dt_max, dt_min, dt_avg = statAWS1DataRec(dt)
    eng_max, eng_min, eng_avg = statAWS1DataRec(data["eng"])
    rud_max, rud_min, rud_avg = statAWS1DataRec(data["rud"])
    rev_max, rev_min, rev_avg = statAWS1DataRec(data["rev"])
    roll_max, roll_min, roll_avg = statAWS1DataRec(data["roll"])
    pitch_max, pitch_min, pitch_avg = statAWS1DataRec(data["pitch"])
    rpitch_max, rpitch_min, rpitch_avg = statAWS1DataRec(data["rpitch"])
    yaw_max, yaw_min, yaw_avg = statAWS1DataRec(data["yaw"])
    ryaw_max, ryaw_min, ryaw_avg = statAWS1DataRec(data["ryaw"])
    cog_max, cog_min, cog_avg = statAWS1DataRec(data["cog"])
    rcog_max, rcog_min, rcog_avg = statAWS1DataRec(data["rcog"])
    sog_max, sog_min, sog_avg = statAWS1DataRec(data["sog"])
    rsog_max, rsog_min, rsog_avg = statAWS1DataRec(data["rsog"])

    print("Total Time %f" % duration)
    print("dt max %f avg %f min %f" % (dt_max, dt_avg, dt_min))
    print("eng max %f avg %f min %f" % (eng_max, eng_avg, eng_min))
    print("rud max %f avg %f min %f" % (rud_max, rud_avg, rud_min))
    print("rev max %f avg %f min %f" % (rev_max, rev_avg, rev_min))
    print("roll max %f avg %f min %f" % (roll_max, roll_avg, roll_min))
    print("pitch max %f avg %f min %f" % (pitch_max, pitch_avg, pitch_min))
    print("rpitch max %f avg %f min %f" % (rpitch_max, rpitch_avg, rpitch_min))
    print("yaw max %f avg %f min %f" % (yaw_max, yaw_avg, yaw_min))
    print("ryaw max %f avg %f min %f" % (ryaw_max, ryaw_avg, ryaw_min))
    print("cog max %f avg %f min %f" % (cog_max, cog_avg, cog_min))
    print("rcog max %f avg %f min %f" % (rcog_max, rcog_avg, rcog_min))
    print("sog max %f avg %f min %f" % (sog_max, sog_avg, sog_min))
    print("rsog max %f avg %f min %f" % (rsog_max, rsog_avg, rsog_min))

    return {
        "duration":duration, 
        "dt_max": dt_max, "dt_min": dt_min, "dt_avg": dt_avg,
        "eng_max": eng_max, "eng_min": eng_min, "eng_avg": eng_avg,
        "rud_max":rud_max, "rud_min":rud_min, "rud_avg":rud_avg,
        "rev_max": rev_max, "rev_min": rev_min, "rev_avg": rev_avg,
        "roll_max":roll_max, "roll_min":roll_min, "roll_avg":roll_avg,
        "pitch_max":pitch_max, "pitch_min":pitch_min, "pitch_avg":pitch_avg,
        "rpitch_max":rpitch_max, "rpitch_min":rpitch_min, "rpitch_avg":rpitch_avg,
        "yaw_max":yaw_max, "yaw_min":yaw_min, "yaw_avg":yaw_avg,
        "ryaw_max":ryaw_max, "ryaw_min":ryaw_min, "ryaw_avg":ryaw_avg,
        "cog_max":cog_max, "cog_min":cog_min, "cog_avg":cog_avg,
        "rcog_max":rcog_max, "rcog_min":rcog_min, "rcog_avg":rcog_avg,
        "sog_max":sog_max, "sog_min":sog_min, "sog_avg":sog_avg,
        "rsog_max":rsog_max, "rsog_min":rsog_min, "rsog_avg":rsog_avg
    }

def loadAWS1Data(fname, str_vec, tstart=0, tend=sys.float_info.max, wavg=5):
    bstart=False
    bend=False
    t=[]
    eng=[]
    rud=[]
    rev=[]
    roll=[]
    pitch=[]
    yaw=[]
    cog=[]
    sog=[]
    rroll=[]
    rpitch=[]
    ryaw=[]
    rcog=[]
    rsog=[]
    try:
        fcsv = open(fname)
    except IOError:
        return np.array(t), np.array(eng), np.array(rud), np.array(roll), np.array(rroll), np.array(pitch), np.array(rpitch), np.array(yaw), np.array(ryaw), np.array(cog), np.array(rcog), np.array(sog), np.array(rsog)

    rcsv = csv.reader(fcsv, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    data=[v for v in rcsv]
    recname={}

    #creating dictionaly of record name
    for i in range(len(data[0])):
        recname[data[0][i]]=i    
        tinit=long(data[1][0])

    for i in range(1,len(data)):
        if(bend):
            break
    
        for j in range(len(data[i])):
            if(recname["t"]==j):
                tcur=float(long(data[i][j])-tinit)/10000000.
                if (tcur >= tstart):
                    bstart=True

                if(tcur >= tend):
                    bend=True

                if(not bstart or bend):
                    break

                t.append(tcur)            
        
            if(recname["yaw_i"]==j):
                yaw.append(float(data[i][j]))
            if(recname["roll_i"]==j):
                roll.append(float(data[i][j]))
            if(recname["pitch_i"]==j):
                pitch.append(float(data[i][j]))
            if(recname["eng_i"]==j):
                eng.append(float(data[i][j]))
            if(recname["rud_i"]==j):
                rud.append(float(data[i][j]))
            if(recname["rev_i"]==j):
                rev.append(float(data[i][j]))
            if(recname["cog_i"]==j):
                cog.append(float(data[i][j]))
            if(recname["sog_i"]==j):
                sog.append(float(data[i][j]))

    # avearging cog and sog by window +-wavg
    for i in range(wavg,len(t)-wavg-1):
        sum=0.;
        for j in range(i - wavg, i + wavg + 1):
            sum += sog[j]
        
        sog[i] = sum / float(wavg * 2 + 1);

        sum=0.;
        for j in range(i - wavg,i + wavg + 1):
            if(cog[i] - cog[j] > 180.):
                sum += cog[j]+360.;
            elif(cog[i] - cog[j] < -180.):
                sum += cog[j]-360.;
            else:
                sum += cog[j]
                
        cog[i] = sum / float(wavg * 2 + 1);
        if(cog[i] > 360.):
            cog[i] -= 360.;
        elif(cog[i] < 0.):
            cog[i] += 360.;
    
    for i in range(len(t)-1):
        dt0=t[i]-t[i-1]
        dt1=t[i+1]-t[i]
        idt = 1.0 / (dt0 + dt1)
   
        if (i < 1):
            rroll.append(0.)
            rpitch.append(0.)
            ryaw.append(0.)
            rcog.append(0.)
            rsog.append(0.)        
        else:
            rroll.append(idt * (roll[i+1]-roll[i-1]))
            rpitch.append(idt * (pitch[i+1]-pitch[i-1]))
            diff =  yaw[i+1] - yaw[i-1]
            if diff > 180.:
                diff -= 360.
            elif diff < -180:
                diff += 360.        
    
            ryaw.append(idt * (diff))
            diff = cog[i+1] - cog[i-1]
            if diff > 180.:
                diff -= 360.
            elif diff <= -180.:
                diff += 360.
            
            rcog.append(idt * (diff))
            rsog.append(idt * (sog[i+1]-sog[i-1]))
            
    rroll.append(0.)
    rpitch.append(0.)
    ryaw.append(0.)
    rcog.append(0.)
    rsog.append(0.)
    
    aws1_data_full = {'t': np.array(t), 'eng': np.array(eng), 'rud': np.array(rud), 'rev': np.array(rev), 'roll': np.array(roll), 'rroll': np.array(rroll), 'pitch': np.array(pitch), 'rpitch': np.array(rpitch), 'yaw': np.array(yaw), 'ryaw': np.array(ryaw), 'cog': np.array(cog), 'rcog': np.array(rcog), 'sog': np.array(sog), 'rsog': np.array(rsog)}
    aws1_data = {}
    for key in str_vec:
        aws1_data[key] = aws1_data_full[key]
    return aws1_data
