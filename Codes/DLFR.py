#The original paper: A distinctive landmark-based face recognition system for identical twins by extracting novel weighted features.
# Mousavi, S., Charmi, M., & Hassanpoor, H. (2021).
#Computers & Electrical Engineering, 94, 107326.
#https://doi.org/10.1016/j.compeleceng.2021.107326

#The other related paper: Mousavi, S., Charmi, M. & Hassanpoor, H. Recognition of identical twins based on the most...
# distinctive region of the face: Human criteria and machine processing approaches.
# Multimed Tools Appl 80, 15765–15802 (2021).
# https://doi.org/10.1007/s11042-020-10360-3



import cv2
import numpy as np
import pylab
from cpselect.cpselect import cpselect
from matplotlib import pyplot as plt
import os
# from PIL import Image
from collections import Counter

twin_match_p_facecurve=[0,0,4,4,1,1,0,0,2,2,1,1,0,0,1,1,1,1,1,1,1,1,5,5,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,
                        0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,0,0,0,0,0,0,3,3,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,
                        0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,1,1,0,0,1,1,0,0,
                        1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,2,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,2,2,
                        0,0,1,1,0,0,0,0,0,0,1,1,4,4,0,0,0,0,0,0,0,0,0,0,2,2,1,1]
twin_match_p_facecurve = np.asarray(twin_match_p_facecurve, dtype=np.int32)
people_match_p_facecurve=[2,2,3,3,0,0,1,1,5,5,0,0,13,13,3,3,4,4,3,3,0,0,2,2,1,1,2,2,3,3,1,1,6,6,0,0,0,0,7,7,3,3,0,0,0,0,
                          6,6,0,0,7,7,0,0,0,0,0,0,1,1,0,0,0,0,1,11,1,0,0,1,1,7,7,0,0,1,1,4,4,1,1,0,0,2,2,2,2,5,5,3,3,1,1,
                          1,1,1,1,1,1,1,1,3,3,0,0,3,3,0,0,0,0,1,1,3,3,0,0,3,3,6,6,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,
                          0,8,8,0,0,1,1,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,2,2,0,0,2,2,0,0]

people_match_p_facecurve = np.asarray(people_match_p_facecurve, dtype=np.int32)
for i in range(181,183,2):
    class_val = []
    class_val2 = []

    #------------------------first part---------------------------
    #read 2 images include a pair of twins/nontwins
    img1="%d.jpg"%(i)
    img2="%d.jpg"%(i+1)
    original = cv2.imread(img1)  # queryImage
    compared = cv2.imread(img2)  # trainImage
    (r_org, c_org, ch_org) = original.shape
    original_W = np.zeros([r_org, c_org, ch_org], dtype=np.uint8)

    #------------------------second part----------------------------
    #implementation of Modified SIFT algorithm on each image in order to detect match & mismatch points of them
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(compared, None)

    index_params = {'algorithm': 5, 'trees': 5}
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            # matchesMask[i] = [1, 0]
            good_points.append(m)

    kp1_matched = ([kp_1[m.queryIdx] for m in good_points])
    kp2_matched = ([kp_2[m.trainIdx] for m in good_points])

    kp1_miss_matched = [kp for kp in kp_1 if kp not in kp1_matched]
    kp2_miss_matched = [kp for kp in kp_2 if kp not in kp2_matched]
    # draw only miss matched or not matched keypoints location
    img1_miss_matched_kp = cv2.drawKeypoints(original, kp1_miss_matched, None, color=(0, 255, 0), flags=0)
    plt.imshow(img1_miss_matched_kp), plt.show()

    img2_miss_matched_kp = cv2.drawKeypoints(compared, kp2_miss_matched, None, color=(0, 255, 0), flags=0)
    plt.imshow(img2_miss_matched_kp), plt.show()

    print("Keypoints 1st image: " + str(len(kp_1)))
    print("Keypoints 2nd Image: " + str(len(kp_2)))
    print("Good Matches :", len(good_points))
    percentage_match = (len(good_points) / number_keypoints) * 100
    print("matching percentage :" + str('%'), percentage_match)
    print("unmatch Keypoints of 1st image: " + str(len(kp1_miss_matched)))
    print("unmatch Keypoints of 2nd image: " + str(len(kp2_miss_matched)))

    # draw matched and mis matched points together
    match_points=len(good_points)
    result = cv2.drawMatches(original, kp_1, compared, kp_2, good_points, None, matchColor=(0, 255, 0),singlePointColor=(255, 0, 0))
    plt.imshow(result), plt.show()
    cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
    path = '....'
    cv2.imwrite(os.path.join(path, "U_match_W%d.jpg"%(i)), img1_miss_matched_kp)
    cv2.imwrite(os.path.join(path, "U_match_W%d.jpg"%(i+1)), img2_miss_matched_kp)
    cv2.imwrite(os.path.join(path, "result%d.jpg"%(i)), result)

    #-----------------------third part--------------------------------
    # affine Transformation
    cppoints = cpselect("path" % (i + 1))
    src_points = []
    dst_points = []
    (rows, cols, ch) = original.shape
    for u in cppoints:
        src_points.append([u['img1_x'], u['img1_y']])
        dst_points.append([u['img2_x'], u['img2_y']])
    src_points_np=np.asarray(src_points,dtype=np.float32)
    dst_points_np=np.asarray(dst_points,dtype=np.float32)

    affine_matrix = cv2.getAffineTransform(src_points_np, dst_points_np)
    img_output = cv2.warpAffine(original, affine_matrix, (cols, rows))
    # plt.imshow(img_output), plt.show()


    #----------------------fourth part----------------------------------
    #find location of mismatched points in transformed image
    coord_m_m_2 = []
    coord_m_m_1 = []
    for t in kp1_miss_matched:
        bb=np.array([t.pt[0],t.pt[1],1])
        coord_m_m_1.append(bb)
    for w in kp2_miss_matched:
        cc=np.array([w.pt[0],w.pt[1]])
        coord_m_m_2.append(cc)
    coord_mm1_np=np.asarray(coord_m_m_1,dtype=np.float32)
    coord_mm2_np = np.asarray(coord_m_m_2, dtype=np.float32)
    coord_mm1_np=coord_mm1_np.T

    coord_mm1_new = np.matmul(affine_matrix,coord_mm1_np )

    print("mismatch of 1st image"+str(coord_m_m_1))
    print("mismatch of 2nd image" + str(coord_m_m_2))

    #-------------------------AND------------------------------------------
    #AND for jaw and mismatches of image1
    ima1 = "jaw_%d.jpg" % (i)
    ima2 = "mismatch%d.jpg" % (i)
    imga1 = cv2.imread(ima1)
    imga2 = cv2.imread(ima2)

    # convert images to graylevel
    img1_G = cv2.cvtColor(imga1, cv2.COLOR_BGR2GRAY)
    img2_G = cv2.cvtColor(imga2, cv2.COLOR_BGR2GRAY)

    # binary image
    ret, bi_img1 = cv2.threshold(img1_G, 127, 255, cv2.THRESH_BINARY)
    ret, bi_img2 = cv2.threshold(img2_G, 127, 255, cv2.THRESH_BINARY)

    # AND operand
    img_bi_AND1 = cv2.bitwise_and(bi_img1, bi_img2)
    # cv2.imshow("Bitwise AND of Image 1 and 2", img_bi_AND)
    rowsa1, colsa1 = img_bi_AND1.shape

    #AND for jaw and mismatches of image 2
    ima3 = "jaw_%d.jpg" % (i + 1)
    ima4 = "mismatch%d.jpg" % (i + 1)
    imga3 = cv2.imread(ima3)
    imga4 = cv2.imread(ima4)

    # convert images to graylevel
    img3_G = cv2.cvtColor(imga3, cv2.COLOR_BGR2GRAY)
    img4_G = cv2.cvtColor(imga4, cv2.COLOR_BGR2GRAY)

    # binary image
    ret, bi_img3 = cv2.threshold(img3_G, 127, 255, cv2.THRESH_BINARY)
    ret, bi_img4 = cv2.threshold(img4_G, 127, 255, cv2.THRESH_BINARY)

    # AND operand
    img_bi_AND2 = cv2.bitwise_and(bi_img3, bi_img4)
    # cv2.imshow("Bitwise AND of Image 1 and 2", img_bi_AND)
    rowsa2, colsa2 = img_bi_AND2.shape

    points1 = []
    for ii in range(rowsa1):
        for jj in range(colsa1):
            k = img_bi_AND1[ii, jj]
            if k == 255:
                p = []
                p = [ii, jj]
                points1.append(p)

    points1 = np.asarray(points1, dtype=np.int32)
    points2 = []
    for qq in range(rowsa2):
        for ww in range(colsa2):
            z = img_bi_AND2[qq, ww]
            if z == 255:
                g = []
                g = [qq, ww]
                points2.append(g)
    points2 = np.asarray(points2, dtype=np.int32)

    if len(points1)==0 or len(points2)==0:
            points_AND=0
    else:
        points_AND = np.concatenate((points1, points2), axis=0)  # a list that include all mismatch points in 2 images
    #-----------------------------------fifth part-------------------------------------
    #comparison of mismatch neighbourhood with KDtree
