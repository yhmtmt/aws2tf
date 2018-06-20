import random
import numpy as np
import matplotlib.pyplot as plt
import sys
#import pdb
import csv

#pdb.set_trace()

def getAWS1BatchSection(batch_size, input_length, predict_length, eng, rud, rroll, rpitch, ryaw, rcog, sog, rsog, start=0.,end=1.):
    istart = int(start*float(len(eng)))
    iend = int(end*float(len(eng)))
    rnum = [random.randint(istart,iend - predict_length - input_length) for i in range(batch_size)]
    xs = np.array([[[eng[i],rud[i],rroll[i],rpitch[i],ryaw[i],rcog[i],sog[i],rsog[i]] for i in range(r, r + input_length)] for r in rnum])
    ts = np.array([[[rroll[i],rpitch[i],ryaw[i],rcog[i],sog[i],rsog[i]] for i in range(r + input_length, r + input_length + predict_length)] for r in rnum])
    return xs, np.reshape(ts, (batch_size,-1))

def getAWS1BatchSectionSeq(predict_length, eng, rud, rroll, rpitch, ryaw, rcog, sog, rsog, ipos, start=0.,end=1.):
    istart = max(ipos, int(start*float(len(eng))))
    iend = int(end*float(len(eng))) - predict_length - 1

    ismpl = ipos + istart
    if(ismpl >= iend):
#        return np.array([]),np.array([])
        return None, None
        
    xs = np.array([[[eng[ismpl],rud[ismpl],rroll[ismpl],rpitch[ismpl],ryaw[ismpl],rcog[ismpl],sog[ismpl],rsog[ismpl]]]])
    ts = np.array([[[rroll[i],rpitch[i],ryaw[i],rcog[i],sog[i],rsog[i]] for i in range(ismpl + 1, ismpl + 1 + predict_length)]])
    return xs, np.reshape(ts, (1,-1))

def loadAWS1DataList(fname):
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
    return fcsv_names, fsizes


def getAWS1DataBatch(fcsv_names, fsizes, batch_size, input_length, predict_length, eval=False):
    total_fsize = 0;
    
    for fsize in fsizes:
        total_fsize += fsize
    
    nrnd = random.randint(0, total_fsize-1)

    total_fsize = 0
    ifile = 0
    for i in range(len(fsizes)):
        total_fsize +=fsizes[i]
        if(nrnd < total_fsize):
            ifile = i
            break;

    t,eng,rud,roll,rroll,pitch,rpitch,yaw,ryaw,cog,rcog,sog,rsog = loadAWS1Data(fcsv_names[ifile])
    
    section = random.randint(0, 1)
    if eval:
        section = 2 * section + 1
        print("Batch from %s section %d" % (fcsv_names[ifile], section))
    else:
        section = 2 * section

    xs, ts = getAWS1BatchSection(batch_size, input_length, predict_length, eng, rud, rroll, rpitch, ryaw, rcog, sog, rsog, section * 0.25, min(1.0, (section + 1) * 0.25))

    return xs, ts
   

def loadAWS1Data(fname, tstart=0, tend=sys.float_info.max, wavg=5):
    bstart=False
    bend=False
    t=[]
    eng=[]
    rud=[]
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
    lenTimeSequence=len(data)
    lenRecord=len(data[0])


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
    
    return np.array(t), np.array(eng), np.array(rud), np.array(roll), np.array(rroll), np.array(pitch), np.array(rpitch), np.array(yaw), np.array(ryaw), np.array(cog), np.array(rcog), np.array(sog), np.array(rsog)
