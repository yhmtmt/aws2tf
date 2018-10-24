import AWS1Log
import argparse
import sys
import os
import pdb
pdb.set_trace()

parser = argparse.ArgumentParser()

parser.add_argument("--path_log", type=str, default="./", help="Path to the AWS1's log directory")
parser.add_argument("--path_plot", type=str, default="./", help="Path plotted images to be stored.")
parser.add_argument("--ts", type=float, default=0.0, help="Sta11rt time in sec")
parser.add_argument("--te", type=float, default=sys.float_info.max, help="End time in sec")
parser.add_argument("--nlog", type=int, default=-1, help="Log number used in stat, plot, and play.")
parser.add_argument("--logs", type=str, default="", help="File of log list used in calculation through multiple logs")
parser.add_argument("--list", action="store_true", help="List logs")
parser.add_argument("--stat", action="store_true", help="Calculate Stats.")
parser.add_argument("--plot", action="store_true", help="Generate Plots.")
parser.add_argument("--play", action="store_true", help="Play log.")
args=parser.parse_args()
nlog=args.nlog
logs=args.logs
path_log=args.path_log
path_plot=args.path_plot
ts=args.ts
te=args.te

# log files to be loaded.
if args.stat or args.plot or args.play:    
    log_time = AWS1Log.ldl.selectAWS1Log(path_log, nlog)
    if log_time == -1:
        print ("No log is found, or specified.")
        exit()
    log = AWS1Log.AWS1Log()
    log_time = log.load(path_log, log_time)
    
if args.list:
    logs=AWS1Log.ldl.listAWS1Logs(path_log)
    AWS1Log.ldl.printAWS1Logs(logs)  
    
if len(logs) != 0:
    file=open(logs)
    nlog=0
    logs=[]
    while True:
        log_time=file.readline().strip()
        if len(log_time) != 17:
            break
        
        path=path_log + "/" + log_time
        if(os.path.isdir(path)):
            print("%d:%s" % (nlog, log_time))
            nlog+=1
            logs.append(log_time)
        else:
            print("No such log: %s" % path)
    
    
if args.stat:
    log.stat(ts,te)

if args.plot:
    log.plot(ts,te, path_plot)

if args.play:
    log.play(ts,te)

    

