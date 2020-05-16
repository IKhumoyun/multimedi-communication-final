## Imports ##
import cv2
import numpy as np

from utils import get_LAB_image, get_YCC_image


## Functions ##


#  A function to detect skin pixels on a picture
#
#  @param imageRGB : the image to analyse in RGB
#  @returns the image with only the pixels indetified as skin still visible
def skin_detection_generic(imageRGB):

    # We define the limits in LAB coordinate for our first mask and apply it
    LAB_limit_inf = (50, 133, 130)
    LAB_limit_sup = (230, 175, 255)
    mask1 = cv2.inRange(get_LAB_image(imageRGB), LAB_limit_inf, LAB_limit_sup)
    result = cv2.bitwise_and(imageRGB, imageRGB, mask=mask1)

    # We define the limits in YCrCb coordinate for our second mask and apply it
    YCC_limit_inf = (50, 135, 80)
    YCC_limit_sup = (230, 255, 255)
    mask2 = cv2.inRange(get_YCC_image(result), YCC_limit_inf, YCC_limit_sup)
    final = cv2.bitwise_and(result, result, mask=mask2)

    return final, mask2
