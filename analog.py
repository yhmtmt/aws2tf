import AWS1Log
import argparse
import sys
import os
import ldAWS1Log as ldl

import pdb
pdb.set_trace()

parser = argparse.ArgumentParser()

parser.add_argument("--path_log", type=str, default="./", help="Path to the AWS1's log directory")
parser.add_argument("--path_plot", type=str, default="./", help="Path plotted images to be stored.")
parser.add_argument("--ts", type=float, default=0.0, help="Sta11rt time in sec")
parser.add_argument("--te", type=float, default=sys.float_info.max, help="End time in sec")
parser.add_argument("--nlog", type=int, default=-1, help="Log number used in plot, and play.")
parser.add_argument("--logs", type=str, default="", help="File of log list used in calculation through multiple logs")
parser.add_argument("--mstat", type=str, default="", help="Type of stats with multiple logs") 
parser.add_argument("--list", action="store_true", help="List logs")
parser.add_argument("--plot", action="store_true", help="Generate Plots.")
parser.add_argument("--play", action="store_true", help="Play log.")
parser.add_argument("--force", action="store_true", help="Force calculation")
parser.add_argument("--debug", action="store_true", help="Debug mode")

args=parser.parse_args()
nlog=args.nlog
mstat=args.mstat
logs=args.logs
path_log=args.path_log
path_plot=args.path_plot
ts=args.ts
te=args.te

# log files to be loaded.
if args.plot or args.play:    
    log_time = ldl.selectAWS1Log(path_log, nlog)
    if log_time == -1:
        print ("No log is found, or specified.")
        exit()
    log = AWS1Log.AWS1Log()
    log.load(path_log, log_time)
    
if args.list:
    logsall=ldl.listAWS1Logs(path_log)
    ldl.printAWS1Logs(logsall)  
    
if len(logs) != 0:
    logs = ldl.loadAWS1Logs(path_log, logs)
    ldl.printAWS1Logs(logs)

def plotAWS1Log(log, log_time, force=False):
    if not os.path.exists(path_plot):
        os.mkdir(path_plot)
    
    path="%s/%d" % (path_plot, log_time)
    if not os.path.exists(path) or force:        
        log.plot(ts,te, path)
    else:
        print("%s exists. Overwrite? (y/n)" % path)
        yorn=sys.stdin.readline().strip()
        if yorn == "y":
            log.plot(ts,te,path)
    
if args.plot:
    plotAWS1Log(log, log_time, args.force)
    
if args.play:
    log.play(ts,te)
    
if len(mstat) != 0:
    if mstat == "sogrpm":
        AWS1Log.plotAWS1MstatSogRpm(path_log, logs, path_plot, args.force)
    elif mstat == "plot":
        log = AWS1Log.AWS1Log()
        for log_time in logs:
            log.load(path_log, int(log_time))
            plotAWS1Log(log, int(log_time), args.force)        
    else:
        print("Unknown mult-stat %s" % mstat)
            
