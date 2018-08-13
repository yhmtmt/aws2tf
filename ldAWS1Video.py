import numpy as np
import cv2

def loadAWS1VideoStream(fvideo, fts):
    strm = cv2.VideoCapture(fvideo)
    nfrms = strm.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = strm.get(cv2.CAP_PROP_FPS)
    width = strm.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = strm.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ts=np.fromfile(fts, dtype='int64')
    tssec=np.empty(shape=ts.shape, dtype='float')
    for i in range(ts.shape[0]):
        tssec[i] = float(ts[i] - ts[0]) * (1.0 / 10000000.0);

    nfrms=ts.shape[0]
    print("%s nfrm:%d fps: %f size: %d,%d"% (fvideo, nfrms, fps, width, height))

    return tssec,strm

'''
path="/mnt/c/cygwin64/home/yhmtm/aws/log/"
log="15297178267535846/"
fvideo=path+log+"mako0.avi"
fts=path+log+"mako0.ts"

ts,strm=loadAWS1Stream(fvideo,fts)

for ifrm in range(ts.shape[0]):
    (ret, frm) = strm.read()
    t=ts[ifrm]
    cv2.imshow('frame', frm)
    key = cv2.waitKey(10)

    if key == 27:
        cv2.destroyAllWindow()
        break

        
'''