import numpy as np
import cv2

def loadAWS1Stream(fvideo, fts):
    strm = cv2.VideoCapture(fvideo)

    nfrms = strm.get(cv2.CAP_PROP_FRAME_COUNT)
    if(nfrms == 0):
        nfrms = 0
        while True:
            (ret, frm) = strm.read()
            if not ret:
                break
            nfrms += 1            

    fps = strm.get(cv2.CAP_PROP_FPS)
    width = strm.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = strm.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("%s nfrm:%d fps: %f size: %d,%d"% (fvideo, nfrms, fps, width, height))

    ts=np.fromfile(fts, dtype='int64')
    print(ts.shape)
    print(ts)

loadAWS1Stream("/mnt/d/aws/log/15322184144532477/mako0.avi","/mnt/d/aws/log/15322184144532477/mako0.ts")


    

