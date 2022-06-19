import cv2
import numpy as np
#import pylab
#from cpselect.cpselect import cpselect
import matplotlib.pyplot as plt
import os
from PIL import Image
#from scipy.spatial import KDTree
#from collections import Counter

for i in range(1,15,2):
    #-------------------first part---------------
    #read 2 images include a pair of twins
    img1="%d.jpg"%(i)
    img2="%d.jpg"%(i+1)
    original = cv2.imread(img1)  # queryImage
    compared = cv2.imread(img2)  # trainImage


    #---------------------second part-------------------------------
    #implementation of SIFT algorithm on each image in order to detect match & mismatch points
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(compared, None)

    index_params = {'algorithm': 5, 'trees': 5}
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    # for i,(m,n) in enumerate(matches):
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            # matchesMask[i] = [1, 0]
            good_points.append(m)


    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    kp1_matched = ([kp_1[m.queryIdx] for m in good_points])
    kp2_matched = ([kp_2[m.trainIdx] for m in good_points])

    kp1_miss_matched = [kp for kp in kp_1 if kp not in kp1_matched]
    kp2_miss_matched = [kp for kp in kp_2 if kp not in kp2_matched]


    # draw only miss matched or not matched keypoints location
    img1_miss_matched_kp = cv2.drawKeypoints(original, kp1_matched,original, color=(0,255,0),flags=0)
    # plt.imshow(img1_miss_matched_kp), plt.show()

    img2_miss_matched_kp = cv2.drawKeypoints(compared, kp2_matched, compared, color=(0,255,0),flags=0)
    # plt.imshow(img2_miss_matched_kp), plt.show()


    # print("Keypoints 1st image: " + str(len(kp_1)))
    # print("Keypoints 2nd Image: " + str(len(kp_2)))
    print("Good Matches %d :" %(i), len(good_points))
    percentage_match = (len(good_points) / number_keypoints) * 100
    print("matching percentage %d:" %(i)+ str('%'), percentage_match)
    print("unmatch Keypoints of %d image: " %(i) + str(len(kp1_miss_matched)))
    print("unmatch Keypoints of %d image: " %(i+1) + str(len(kp2_miss_matched)))

    # draw matched and mis matched points together
    result = cv2.drawMatches(original, kp_1, compared, kp_2, good_points, None, matchColor=(0, 255, 0),singlePointColor=(255, 0, 0))

    path = '...'
    cv2.imwrite(os.path.join(path, "result%d.jpg" % (i)), result)