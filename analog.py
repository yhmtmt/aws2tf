import AWS1Log
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--path_log", type=str, default="./", help="Path to the AWS1's log directory")
parser.add_argument("--path_plot", type=str, default="./", help="Path plotted images to be stored.")
parser.add_argument("--log", type=int, default=-1, help="Log to be analyzed")
parser.add_argument("--ts", type=float, default=0.0, help="Start time in sec")
parser.add_argument("--te", type=float, default=sys.float_info.max, help="End time in sec")
parser.add_argument("--stat", action="store_true", help="Calculate Stats.")
parser.add_argument("--plot", action="store_true", help="Generate Plots.")
parser.add_argument("--play", action="store_true", help="Play log.")
args=parser.parse_args()

path_log=args.path_log
path_plot=args.path_plot
log_time=args.log
ts=args.ts
te=args.te

log = AWS1Log.AWS1Log()

if log_time == -1:
    log_time = AWS1Log.ldl.selectAWS1Log(path_log)

if log_time == -1:
    print ("No log is found, or specified.")
    exit()
    
log_time = log.load(path_log, log_time)

if args.stat:
    log.stat(ts,te, path_plot)

if args.plot:
    log.plot(ts,te, path_plot)

if args.play:
    log.play(ts,te)
    

