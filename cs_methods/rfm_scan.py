import numpy as np
import os
def rfm_scan(img1, img2, kp1, kp2, putative_corres, fn):
    # read txt file
    rf = open("./matlab_results/rfm_scan_result/rfm_scan_result_" + fn + '.txt', 'r')
    result = rf.readlines()
    result = [int(x.strip()) for x in result]
    goodMatches = np.array(result)
    rf.close()
    return goodMatches
