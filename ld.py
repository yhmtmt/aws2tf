import random
import numpy as np
import matplotlib.pyplot as plt
import sys
#import pdb
import csv

#pdb.set_trace()
def getAWS1DataAt(str_vec, data, idx):
    vec = [data[key][idx] for key in str_vec]
    return vec


def getAWS1BatchSection(str_vec_in, str_vec_out, batch_size, input_length, predict_length, data, start=0.,end=1.):
    if(len(str_vec_in) == 0 or len(str_vec_out) == 0):
        return None, None
    len_data = len(data[str_vec_in[0]])
    istart = int(start*float(len_data))
    iend = int(end*float(len_data))
    rnum = [random.randint(istart,iend - predict_length - input_length) for i in range(batch_size)]
    xs = np.array([[getAWS1DataAt(str_vec_in, data, i) for i in range(r, r + input_length)] for r in rnum])
    ts = np.array([[getAWS1DataAt(str_vec_out, data, i) for i in range(r + input_length, r + input_length + predict_length)] for r in rnum])
    return xs, np.reshape(ts, (batch_size,-1))

def getAWS1BatchSectionSeq(str_vec_in, str_vec_out, predict_length, data, ipos, start=0.,end=1.):
    if(len(str_vec_in) == 0 or len(str_vec_out) == 0):
        return None, None
    len_data = len(data[str_vec_in[0]])
    istart = max(ipos, int(start*float(len_data)))
    iend = int(end*float(len_data)) - predict_length - 1

    ismpl = ipos + istart
    if(ismpl >= iend):
        return None, None
        
    xs = np.array([[getAWS1DataAt(str_vec_in, data, ismpl)]])
    ts = np.array([[getAWS1DataAt(str_vec_out, data, i) for i in range(ismpl + 1, ismpl + 1 + predict_length)]])
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


def getAWS1DataBatch(fcsv_names, str_vec_in, str_vec_out, fsizes, batch_size, input_length, predict_length, eval=False):
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

    str_vec = list(set(str_vec_in + str_vec_out))
    aws1_data = loadAWS1Data(fcsv_names[ifile],str_vec)
    
    section = random.randint(0, 1)
    if eval:
        section = 2 * section + 1
        print("Batch from %s section %d" % (fcsv_names[ifile], section))
    else:
        section = 2 * section

    xs, ts = getAWS1BatchSection(str_vec_in, str_vec_out, batch_size, input_length, predict_length, aws1_data, section * 0.25, min(1.0, (section + 1) * 0.25))

    return xs, ts
   

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
