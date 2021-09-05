import numpy as np
import os
def lpm(img1, img2, kp1, kp2, putative_corres, fn):
    left_dir = './putative_matches/' + fn + '_putative_left.txt'
    right_dir = './putative_matches/' + fn + '_putative_right.txt'
    if not os.path.exists(left_dir):
        left_coor = [kp1[m.queryIdx].pt for m in putative_corres]
        right_coor = [kp2[m.trainIdx].pt for m in putative_corres]
        f1 = open(left_dir, 'w')
        f2 = open(right_dir, 'w')
        for i in range(len(left_coor)):
            l_pt = left_coor[i]
            f1.write(str(l_pt[0]))
            f1.write('\t')
            f1.write(str(l_pt[1]))
            f1.write('\n')

            r_pt = right_coor[i]
            f2.write(str(r_pt[0]))
            f2.write('\t')
            f2.write(str(r_pt[1]))
            f2.write('\n')

        f1.close()
        f2.close()
    else:
        print("putative set data existed~ ")
    
    # read txt file
    rf = open("./matlab_results/lpm_result/lpm_result_" + fn + '.txt', 'r')
    result = rf.readlines()
    result = [int(x.strip()) for x in result]
    goodMatches = np.array(result)
    rf.close()
    return goodMatches