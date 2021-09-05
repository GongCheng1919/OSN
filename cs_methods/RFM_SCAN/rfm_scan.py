import numpy as np
import cv2
import os

def rfm_scan(im1, img2, kp1, kp2, putative_corres):
    left_coor = [kp1[m.queryIdx].pt for m in putative_corres]
    right_coor = [kp2[m.trainIdx].pt for m in putative_corres]
    path = './cs_methods/RFM_SCAN/'
    with open(path + 'putative_left.txt', 'w') as f:
        for coor in left_coor:
            f.write("%f %f\n" % (coor[0], coor[1]))
    
    with open(path + 'putative_right.txt', 'w') as f:
        for coor in right_coor:
            f.write("%f %f\n" % (coor[0], coor[1]))

    exe = path + "RFM_SCAN.exe"
    res = "%sRFM_SCAN_result.txt" % path
    cmd = "%s %sputative_left.txt %sputative_right.txt %s" % (exe, path, path, res)
    os.system(cmd)
    result = []
    with open(res, 'r') as f:
        lines = f.readlines()
        result = [int(i.strip()) for i in lines]
    return np.array(result)