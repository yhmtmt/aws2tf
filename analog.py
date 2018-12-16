import AWS1Log
import argparse
import sys
import os
import re
import ldAWS1Log as ldl

parser = argparse.ArgumentParser()

parser.add_argument("--path_log", type=str, default="./", help="Path to the AWS1's log directory")
parser.add_argument("--path_result", type=str, default="./", help="Path plotted images to be stored.")
parser.add_argument("--path_model_param", type=str, default="./model_param", help="Path model parameter is stored.") 
parser.add_argument("--ts", type=float, default=0.0, help="Start time in sec")
parser.add_argument("--te", type=float, default=sys.float_info.max, help="End time in sec")
parser.add_argument("--nlog", type=int, default=-1, help="Log selection by log number.")
parser.add_argument("--logs", type=str, default="", help="Log selecgion by file of log list")
parser.add_argument("--all", action="store_true", help="Log selection, all logs in the path_log")
parser.add_argument("--op", type=str, default="", help="Type of stats with multiple logs")
parser.add_argument("--pstat", type=str, default="", help="Print statistics")
parser.add_argument("--sel", type=str, default="", help="Select logs by condition <par>"<" or ">"<value> from logs given in --logs")
parser.add_argument("--list", action="store_true", help="List logs selected")
parser.add_argument("--force", action="store_true", help="Force process all logs")
parser.add_argument("--new", action="store_true", help="Process only new logs")
parser.add_argument("--debug", action="store_true", help="Debug mode")

args=parser.parse_args()
nlog=args.nlog
op=args.op
pstat=args.pstat
sel=args.sel
logsfile=args.logs
logs=[]
path_log=args.path_log
path_result=args.path_result
path_model_param=args.path_model_param
ts=args.ts
te=args.te

if args.debug:
    import pdb
    pdb.set_trace()

if args.all:
    logs = ldl.listLogs(path_log)
else:
    if len(logsfile) != 0: # log list is given
        logs = ldl.loadListLogs(path_log, logsfile)

    if nlog != -1: # log number is given
        log_time = ldl.selectLog(path_log, nlog)
        logs.append("%s" % log_time)

    if len(logs) == 0:
        print ("No log is specified please select.")
        log_time = ldl.selectLog(path_log)
        logs.append("%s" % log_time)

if args.list:
    ldl.printListLogs(logs)
    
def procAWS1Log(log, log_time, force=False, new=True):
    if not os.path.exists(path_result):
        os.mkdir(path_result)
    
    path="%s/%d" % (path_result, log_time)
    if not os.path.exists(path) or force:
        log.proc(ts,te, path)
    elif new:
        print("Skip processing " + path)
    else:        
        print("%s exists. Overwrite? (y/n)" % path)
        yorn=sys.stdin.readline().strip()
        if yorn == "y":
            log.proc(ts,te,path)  
    
if len(op) != 0:    
    if op == "sogrpm":
        AWS1Log.procOpSogRpm(path_log, logs, path_result, args.force)
    elif op == "proc":
        log = AWS1Log.AWS1Log()
        for log_time in logs:
            log.load_model_param(path_model_param)
            log.load(path_log, int(log_time))
            procAWS1Log(log, int(log_time), args.force, args.new)
            log.save_model_param(path_model_param)
            
    elif op == "play":
        log = AWS1Log.AWS1Log()
        for log_time in logs:
            log.load(path_log, int(log_time))
            log.play(ts, te)       
    else:
        print("Unknown Operation %s" % op)
            
if len(pstat) != 0:
    strpars=pstat.split(",")
    AWS1Log.printStat(path_log, logs, path_result, strpars)

if len(sel):
    m=sel.split("<")
    if len(m)!=2:
        m=sel.split(">")
    else:
        if m[1][0] == '=':
            cond=[m[0],'<=',m[1][1:]]
        else:
            cond=[m[0],'<',m[1]]
        
    if len(m)!=2:
        print ("Unknown operator used, or any operator is not used in " + sel)
        exit()
    else:
        if m[1][0] == '=':
            cond=[m[0],'<=',m[1][1:]]
        else:
            cond=[m[0],'>',m[1]]
        
        
    logs = AWS1Log.selectLogByCond(path_log, logs, path_result, cond)
    for log in logs:
        print (log)


