import numpy as np
import os
def vfc(img1, img2, kp1, kp2, putative_corres, fn):
    # read txt file
    rf = open("./matlab_results/vfc_result/vfc_result_" + fn + '.txt', 'r')
    result = rf.readlines()
    result = [int(x.strip()) for x in result]
    goodMatches = np.array(result)
    rf.close()
    return goodMatches