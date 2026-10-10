import cv2, numpy as np
def green_frac(crop):
    h=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    m=(h[...,0]>=40)&(h[...,0]<=90)&(h[...,1]>=70)&(h[...,2]>=40)
    return float(m.mean())
