import numpy as np
from enum import Enum
import time
import cv2
from cv2.xfeatures2d import matchGMS

def gms(img1, img2, kp1, kp2, putative_matches, fn="none"):
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, putative_matches, withScale=False, withRotation=False, thresholdFactor=6)
    gms_Idx = [(m.queryIdx, m.trainIdx) for m in matches_gms]
    result = np.zeros(len(putative_matches))
    for i, m in enumerate(putative_matches):
        if (m.queryIdx, m.trainIdx) in gms_Idx:
            result[i] = 1
    return result