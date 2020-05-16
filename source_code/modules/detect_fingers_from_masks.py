import cv2
import math
import numpy as np


def filterBiggestContours(contours):
    """Filter the contours to keep the biggest ones"""

    # Detect the max number of points and the largest area
    maxIndex = 0
    maxArea = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            maxArea = area

    # Keep all contours with nb of points > maxIndex an areas > 0.5 * maxArea
    keep_indexes = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area > maxArea * 0.2):
            keep_indexes.append(i)

    return [contours[i] for i in keep_indexes]


def getMinimizedConvHull(convHull):
    # Determine the minimum separation distance:
    # We determine the width of the contour and apply an arbitrary threshold / 10
    minX, minY, maxX, maxY = convHull[0, 0][0], convHull[0,0][1], convHull[0, 0][0], convHull[0, 0][1]
    for point in convHull:
        X, Y = point[0][0], point[0][1]
        if X > maxX:
            maxX = X
        elif X < minX:
            minX = X
        if Y > maxY:
            maxY = Y
        elif Y < minY:
            minY = Y
    separation_dist = min(maxX-minX, maxY-minY)/10

    # We go through the list of points once to construct [d0, d1, ..., dn]
    distances = []
    for i in range(1, len(convHull)):
        distances.append((convHull[i, 0][0]-convHull[i-1, 0][0])
                         ** 2 + (convHull[i, 0][1]-convHull[i-1, 0][1])**2)

    minimized_indexes = []
    count = 1
    for i in range(len(distances)):
        if(distances[i] > separation_dist**2):
            minimized_indexes.append(i - int(count/2))
            count = 1
        else:
            count += 1
    if count > 0:
        minimized_indexes.append(len(distances) - int(count/2))
    return minimized_indexes


def getMinimumConvHullKMeansClustering(convHull, k_Max=9):

    # Classify the points series with a kmean clustering
    lowest_compacity = 10000000
    labelized = []
    for i in range(1, 10):
        compacity, labels, means = cv2.kmeans(data=np.float32(convHull), K=i, bestLabels=None,
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=5,
                                              flags=cv2.KMEANS_RANDOM_CENTERS)
        if(compacity < lowest_compacity):
            lowest_compacity = compacity
            labelized = labels

    # Select median point of each cluster
    center_indexes = []
    start_index = 0
    while(labelized[start_index, 0] == labelized[-1, 0]):
        start_index += 1
    count, cluster = 0, labelized[start_index, 0]
    for i in range(start_index, len(labelized)):
        if(labelized[i, 0] != labelized[i-1, 0] and count > 0):
            center_indexes.append(i - 1 - int(count/2))
            count = 1
            cluster = labelized[i, 0]
        else:
            count += 1
    if(count > 0):
        center_indexes.append(len(labelized) - 1 - int(count/2))

    return center_indexes


def calculateAngle(pt0, pt1, pt2):
    """Cosine rule"""
    a = math.sqrt((pt2[0] - pt0[0])**2 + (pt2[1] - pt0[1])**2)
    b = math.sqrt((pt1[0] - pt0[0])**2 + (pt1[1] - pt0[1])**2)
    c = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle


def detectFingersFromHullConvex(contour, hullConvex_indexes):

    fingers_detected = []
    relevant_convexity_defects = []

    convexity_defects = cv2.convexityDefects(contour, hullConvex_indexes)
    if convexity_defects is None:
        return fingers_detected, relevant_convexity_defects

    # Classic method: counting the hollows
    for defect in convexity_defects:
        s, e, f, d = defect[0]
        start = tuple(contour[s, 0])
        end = tuple(contour[e, 0])
        farthest = tuple(contour[f, 0])
        if(d > 10000) and calculateAngle(start, farthest, end) < 5*math.pi / 12:
            relevant_convexity_defects.append(farthest)
            if(start not in fingers_detected):
                fingers_detected.append(start)
            if(end not in fingers_detected):
                fingers_detected.append(end)

    # Alternative method: count fingers directly
    if len(fingers_detected) < 1:
        for k in range(1, len(convexity_defects)):
            s0, e0, f0, d0 = convexity_defects[k-1][0]
            s1, e1, f1, d1 = convexity_defects[k][0]
            far0 = tuple(contour[f0, 0])
            end0 = tuple(contour[e0, 0])
            far1 = tuple(contour[f1, 0])
            if(calculateAngle(far0, end0, far1) < math.pi / 3):
                fingers_detected.append(end0)
                relevant_convexity_defects.append(far0)
                relevant_convexity_defects.append(far1)

    return fingers_detected, relevant_convexity_defects


def findFingersFromMask(hands_mask, keepBiggestContours=False):
    LINE_WIDTH = 5 + int(hands_mask.shape[1] * 0.1)
    PT_RADIUS = LINE_WIDTH * 2

    # --- DETECT CONTOURS
    # There might be no contour when hand is not inside the frame
    contours, _ = cv2.findContours(
        hands_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(keepBiggestContours):
        contours = filterBiggestContours(contours)

    if (len(contours) == 0):
        print('No contours detected')
        return [], [], [], []

    # --- FIND HULL CONVEX
    convHull_points = []
    convHull_indexes = []
    for i in range(len(contours)):
        convHull_points.append(cv2.convexHull(contours[i], returnPoints=True))
        convHull_indexes.append(cv2.convexHull(
            contours[i], returnPoints=False))

    # We filter the convex envelopes to keep a minimum set of points forming the envelope
    filtered_convHull_points = []
    filtered_convHull_indexes = []
    for i in range(len(convHull_points)):
        filter_indexes = getMinimizedConvHull(convHull_points[i])
        filtered_convHull_points.append(convHull_points[i][filter_indexes])
        filtered_convHull_indexes.append(convHull_indexes[i][filter_indexes])

    # Find fingers and relevant convexity defects
    fingers_detects_list = []
    convexity_defects_list = []
    for i in range(len(filtered_convHull_indexes)):
        fingers_detected, relevant_convexity_defects = detectFingersFromHullConvex(
            contours[i], filtered_convHull_indexes[i])
        fingers_detects_list.append(fingers_detected)
        convexity_defects_list.append(relevant_convexity_defects)

    return fingers_detects_list, convexity_defects_list, contours, filtered_convHull_points
