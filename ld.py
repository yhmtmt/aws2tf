import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import csv

pdb.set_trace()

def loadAWS1Data(fname, tstart=0, tend=sys.float_info.max, wavg=5):
    bstart=False
    bend=False
    fcsv = open(fname)
    rcsv = csv.reader(fcsv, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    data=[v for v in rcsv]
    recname={}
    lenTimeSequence=len(data)
    lenRecord=len(data[0])

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
    
    return np.array(t), np.array(roll), np.array(rroll), np.array(pitch), np.array(rpitch), np.array(yaw), np.array(ryaw), np.array(cog), np.array(rcog), np.array(sog), np.array(rsog)

args = sys.argv

if len(args) == 1:
    sys.exit()

t,roll,rroll,pitch,rpitch,yaw,ryaw,cog,rcog,sog,rsog = loadAWS1Data(args[1])

plt.subplot(2,1,1)
plt.plot(t, cog)
plt.subplot(2,1,2)
plt.plot(t, rcog)
plt.show()

plt.subplot(2,1,1)
plt.plot(t, yaw)
plt.subplot(2,1,2)
plt.plot(t, ryaw)
plt.show()

plt.subplot(2,1,1)
plt.plot(t, sog)
plt.subplot(2,1,2)
plt.plot(t, rsog)
plt.show()

for i in range(len(t)-1):
#    print("t=%f roll=%f rroll=%f pitch=%f rpitch=%f yaw=%f ryaw=%f cog=%f rcog=%f sog=%f rsog=%f \n" % t[i], roll[i], rroll[i], pitch[i], rpitch[i], yaw[i], ryaw[i], cog[i], rcog[i], sog[i], rsog[i])
    print("t=%f " % t[i])
    print("roll=%f " % roll[i])
    print("rroll=%f " % rroll[i])
    print("pitch=%f " % pitch[i])
    print("rpitch=%f " % rpitch[i])
    print("yaw=%f " % yaw[i])
    print("ryaw=%f " % ryaw[i])
    print("cog=%f " % cog[i])
    print("rcog=%f " % rcog[i])
    print("sog=%f " % sog[i])
    print("rsog=%f " % rsog[i])
    print("\n")

    
    
