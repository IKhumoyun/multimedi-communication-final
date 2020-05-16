## Imports ##
import cv2
import numpy as np

from utils import get_LAB_image, get_YCC_image


## Functions ##

#  A function to enhance a mask before applying it by removing isolated points and filling the holes in a skin zone
#
#  @param mask : the mask to enhance (as a grayscale image)
#  @returns the enhanced mask
def enhance_mask(mask):
    # First we remove the very isolated pixels
    temp = remove_very_isolated_pixels(mask)

    # Then we fill the holes using the following kernel and the openCV closing function
    kernel_closing = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel_closing)

    # Finally we remove the big groups of isolated pixels
    final = remove_isolated_group_of_pixels(closing)
    return final


#  A function to remove very isolated pixels from a mask
#
#  @params mask : the mask to process
#  @returns a mask where all very isolated pixels have been removed
def remove_very_isolated_pixels(mask):

    # We apply a blur to the mask to change the color of isolated points from white to grey
    mask_clean = cv2.blur(mask, (3, 3))

    # We apply a treshold to remove all grey points from the mask
    ret, mask_clean = cv2.threshold(mask_clean, 230, 255, cv2.THRESH_BINARY)

    return mask_clean


#  A function to remove group of isolated pixels from a mask
#
#  @params mask : the mask to process
#  @returns a mask where all group of isolated pixels have been removed
def remove_isolated_group_of_pixels(mask):

    # We apply a blur to the mask to change the color of globally isolated points from white to grey
    mask_clean = cv2.blur(mask, (10, 10))

    # We apply a treshold to remove certain grey points from the mask
    ret, mask_clean = cv2.threshold(mask_clean, 170, 255, cv2.THRESH_BINARY)

    return mask_clean


#  A function to detect skin pixels on a picture, being particularly suited to the counting on fingers use case
#
#  @param imageRGB : the image to analyse in RGB
#  @returns the image with only the pixels indetified as skin still visible
def skin_detection_specific(imageRGB):

    # We define the limits in LAB coordinate for our first mask, enhance it and finally apply it
    LAB_limit_inf = (50, 135, 130)
    LAB_limit_sup = (230, 175, 255)
    mask1 = cv2.inRange(get_LAB_image(imageRGB), LAB_limit_inf, LAB_limit_sup)
    enhanced_mask1 = enhance_mask(mask1)
    result = cv2.bitwise_and(imageRGB, imageRGB, mask=enhanced_mask1)

    # We define the limits in YCrCb coordinate for our second mask, enhance it and finally apply it
    YCC_limit_inf = (50, 135, 80)
    YCC_limit_sup = (230, 255, 255)
    mask2 = cv2.inRange(get_YCC_image(result), YCC_limit_inf, YCC_limit_sup)
    enhanced_mask2 = enhance_mask(mask2)
    final = cv2.bitwise_and(result, result, mask=enhanced_mask2)

    return final, enhanced_mask2

