## Imports ##
import cv2
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


## Functions ##


#  A function to load an RGB image and display it
#
#  @param path_to_image: The path to the image we want to load
#  @return the image encoded in RGB
def load_and_show_rgb_image(path_to_image):
    img = cv2.imread(path_to_image, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return img_rgb


#  A function to load an RGB image
#
#  @param path_to_image: The path to the image we want to load
#  @return the image encoded in RGB
def load_rgb_image(path_to_image):
    img = cv2.imread(path_to_image, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


#  A function to load a gray scale image
#
#  @param path_to_image: The path to the image we want to load
#  @return the image encoded in graysacle
def load_gray_image(path_to_image):
    img = cv2.imread(path_to_image, 0)
    return img


#  A function to display an RGB image
#
#  @param image_rgb : the image encoded in RGB that we want to display
def show_color(imageRGB):
    plt.imshow(imageRGB)
    plt.show()


#  A function to display a grayscale image
#
#  @param image_rgb : the image encoded in RGB that we want to display
def show_gray(image_gray):
    plt.imshow(image_gray, cmap='gray')
    plt.show()


#  A fonction to convert an image from RGB to HSV

#  @param imageRGB : the image encoded in RGB that we want to convert
#  @returns the image encoded in HSV
def get_HSV_image(imageRGB):
    return cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)


#  A fonction to convert an image from RGB to LAB

#  @param imageRGB : the image encoded in RGB that we want to convert
#  @returns the image encoded in LAB
def get_LAB_image(imageRGB):
    return cv2.cvtColor(imageRGB, cv2.COLOR_RGB2LAB)


#  A fonction to convert an image from RGB to YCrCb

#  @param imageRGB : the image encoded in RGB that we want to convert
#  @returns the image encoded in YCrCb
def get_YCC_image(imageRGB):
    return cv2.cvtColor(imageRGB, cv2.COLOR_RGB2YCrCb)


#  A fonction to plot the repartition of the pixels of the image in the RGB color space
#
#  @param imageRGB : the image we want to analyse encoded in RGB
def plot_RGB_color_space(imageRGB):

    # We split the image in its three channels and create the figure to plot
    r, g, b = cv2.split(imageRGB)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    #  We set and scale the colors of each point to match the one of its corresponding pixel
    pixel_colors = imageRGB.reshape(
        (np.shape(imageRGB)[0]*np.shape(imageRGB)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # We plot the image, set the labels and display it
    axis.scatter(r.flatten(), g.flatten(), b.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()


#  A fonction to plot the repartition of the pixels of the image in the HSV color space
#
#  @param imageRGB : the image we want to analyse encoded in RGB
def plot_HSV_color_space(imageRGB):

    # We split the image in its three channels and create the figure to plot
    h, s, v = cv2.split(cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV))
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    #  We set and scale the colors of each point to match the one of its corresponding pixel
    pixel_colors = imageRGB.reshape(
        (np.shape(imageRGB)[0]*np.shape(imageRGB)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # We plot the image, set the labels and display it
    axis.scatter(h.flatten(), s.flatten(), v.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


#  A fonction to plot the repartition of the pixels of the image in the LAB color space
#
#  @param imageRGB : the image we want to analyse encoded in RGB
def plot_LAB_color_space(imageRGB):

    # We split the image in its three channels and create the figure to plot
    l, a, b = cv2.split(cv2.cvtColor(imageRGB, cv2.COLOR_RGB2LAB))
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    #  We set and scale the colors of each point to match the one of its corresponding pixel
    pixel_colors = imageRGB.reshape(
        (np.shape(imageRGB)[0]*np.shape(imageRGB)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # We plot the image, set the labels and display it
    axis.scatter(l.flatten(), a.flatten(), b.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Luma")
    axis.set_ylabel("a (green to red)")
    axis.set_zlabel("b (blue to yellow)")
    plt.show()


#  A fonction to plot the repartition of the pixels of the image in the YCrCb color space
#
#  @param imageRGB : the image we want to analyse encoded in RGB
def plot_YCC_color_space(imageRGB):

    # We split the image in its three channels and create the figure to plot
    y, cr, cb = cv2.split(cv2.cvtColor(imageRGB, cv2.COLOR_RGB2YCrCb))
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    #  We set and scale the colors of each point to match the one of its corresponding pixel
    pixel_colors = imageRGB.reshape(
        (np.shape(imageRGB)[0]*np.shape(imageRGB)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # We plot the image, set the labels and display it
    axis.scatter(y.flatten(), cr.flatten(), cb.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Y")
    axis.set_ylabel("Cr")
    axis.set_zlabel("Cb")
    plt.show()
