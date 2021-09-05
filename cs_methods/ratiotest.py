import numpy as np
def ratio_test(raw_corres):
    goodMatches=np.zeros(len(raw_corres))
    ratio=0.66
    for i in range(len(raw_corres)):
        if (raw_corres[i][0].distance < ratio * raw_corres[i][1].distance):
                goodMatches[i]=1
    return goodMatches