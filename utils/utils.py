import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("../")
from cs_methods.ratiotest import ratio_test

def DeAndC(img,detector,descriptor):
    # key points
    kps=detector.detect(img)
    # descriptors
    kps,des=descriptor.compute(img,keypoints=kps)
    return kps,des
# construction
def construct_putative_corres(img1,img2,detector,descriptor,normtype=cv2.NORM_HAMMING):
    # extraction
    kp1,de1=DeAndC(img1,detector,descriptor)
    kp2,de2=DeAndC(img2,detector,descriptor)
#     print(len(kp1),len(kp2))
    # matching
    bf=cv2.BFMatcher(normtype,False)
    raw_corres = bf.knnMatch(de1, de2, k=2)
    putative_corres=[i[0] for i in raw_corres]
    return kp1,de1,kp2,de2,np.array(raw_corres),np.array(putative_corres)
# label the true correspondences by homograph 
def label_putative_corres(kp1,kp2,putative_corres,H,pixel_th=7):
    # query kps
    query_ps=np.array([list(i.pt)+[1] for i in kp1])
    ref_ps=np.array([list(i.pt)+[1] for i in kp2])
    query2ref_indexs=[i.trainIdx for i in putative_corres]
    # identify true corres by homograph
    true_ps=np.dot(query_ps,H.T)
    true_ps=true_ps/np.expand_dims(true_ps[:,2],-1)
    # 像素误差阈值设为x pixel
    mapping_corres=ref_ps[query2ref_indexs]
    res=np.sqrt(np.power(true_ps[:,0]-mapping_corres[:,0],2)+np.power(true_ps[:,1]-mapping_corres[:,1],2))
    labels=res<pixel_th
#     print(H,sum(labels)/(len(labels)))
    return labels
def compute_prf(y_true,y_pred):
    #true positive
    TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
    #false positive
    FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
    #true negative
    TN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
    #false negative
    FN = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
    precision=TP/(TP+FP+ 1e-7)
    recall= TP / (TP + TN+ 1e-7)
#     accuracy = (TP + TN) / (TP + FP + TN + FN)
#     error_rate =  (FN + FP) / (TP + FP + TN + FN)
    F1 = 2*precision*recall/(precision+recall+ 1e-7)
    return precision,recall,F1
def metric_plot(method_name,res,style='r-o',show_auc=True):
    res=np.sort(res)
    AUC=np.sum(res)/len(res)
    label=None
    if show_auc:
        label="%s AUC: %.2f%%"%(method_name,np.mean(res)*100)
    else:
        label="%s"%method_name
    plt.plot(np.arange(1,len(res)+1)/len(res),res,style,label=label)
def give_example(img1,img2,H):
    # examples
#     img1,img2,H=imgpairs[0]
    orb = cv2.ORB_create(
        nfeatures = 10000,
        scaleFactor = 1.2,
        nlevels = 8,
        edgeThreshold = 31,
        firstLevel = 0,
        WTA_K = 2,
        patchSize = 31,
        fastThreshold = 0)
    kp1,de1,kp2,de2,raw_corres,putative_corres=construct_putative_corres(img1,img2,orb,orb,normtype=cv2.NORM_HAMMING)
    # thershold set to 12 when image size is 640x480 (refer to CODE)
    labels=label_putative_corres(kp1,kp2,putative_corres,H=H,pixel_th=12)
    print(compute_prf(labels,labels))
    putaive_img = cv2.drawMatches(img1,kp1,img2,kp2,putative_corres[labels],outImg=None,matchColor=(0,255,0),flags=2)
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.imshow(putaive_img[:,:,-1::-1])
    ratio_corres=ratio_test(raw_corres)
    precision,recall,F1=compute_prf(labels,ratio_corres)
    print(precision,recall,F1)
    ratio_img = cv2.drawMatches(img1,kp1,img2,kp2,putative_corres[np.array(ratio_corres,np.bool)],outImg=None,matchColor=(0,255,0),flags=2)
    plt.subplot(2,1,2)
    plt.imshow(ratio_img[:,:,-1::-1])
    plt.show()
def identify_reliable_corres(imgpairs, CS, detector,descriptor,normtype):
    results=[]
    for i in range(len(imgpairs)):    
        img1,img2,H=imgpairs[i][:3]
        kp1,de1,kp2,de2,raw_corres,putative_corres=construct_putative_corres(img1,img2,detector,descriptor,normtype=cv2.NORM_HAMMING)
        # thershold set to 12 when image size is 640x480 (refer to CODE)
        labels=label_putative_corres(kp1,kp2,putative_corres,H=H,pixel_th=7)
        ratio=sum(labels)/len(labels)
    #     print(compute_prf(labels,labels))
        cs_metrics={}
        strs=""
        for j in CS.keys():
            cs_m=CS[j]
            cs_metrics[j]={}
            start=time.time()
            if j=="Ratio":
                cs_corres=cs_m(raw_corres)
            else:
                cs_corres=cs_m(img1, img2, kp1, kp2, putative_corres, str(i))
            end=time.time()
            cs_metrics[j]["time"]=(end-start)*1000
            precision,recall,F1=compute_prf(labels,cs_corres)
            cs_metrics[j]["precision"]=precision
            cs_metrics[j]["recall"]=recall
            cs_metrics[j]["F1"]=F1
#             print("\t%s metrics: %.2fms, precision=%.4f, recall=%.4f, f1=%.4f"%(j,cs_metrics[j]["time"],precision,recall,F1),end="")
            strs+="%s: %.2fms "%(j,cs_metrics[j]["time"])
#         if len(imgpairs[i])>3:
#             imgpairs[i][3]=ratio
#             imgpairs[i][4]=cs_metrics
#         else:
#             imgpairs[i].append(ratio)
#             imgpairs[i].append(cs_metrics)
        results.append([ratio, cs_metrics])
        print("\rImage pair %d (%dx%d) ratio=%.4f: %s"%(i+1,img1.shape[0],img1.shape[1],ratio,strs),end="")
    return results