import cv2
import numpy as np

def viz_flow(flow):
    # H：0°～360°, red 0°, green 120°, blue 240°
    # S：0.0～1.0
    # V：0.0(black)～1.0(white)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    # you also can set S to 255, the pic will be black
    hsv[...,2] = 255
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb

def get_flow(prvs, next):
    #get GRAY  return RGB

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    test = viz_flow(flow)
    return test


