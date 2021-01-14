import numpy as np
import cv2 
import os

from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as nnf


# MSER Pre-processing Algorithm
def mser(image):
    mser = cv2.MSER_create()
    mser_image = np.copy(image)
    gray = cv2.cvtColor(mser_image, cv2.COLOR_BGR2GRAY)
    regions, _ = mser.detectRegions(gray.astype(np.uint8))
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(mser_image, hulls, 1, (0, 0, 255))

    return mser_image


# Function to reduce the image without aliasing
def reduce_image(image):
    """Reduces an image to half its shape.
    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)
    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.
    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.
    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    kernel = np.array([0.25 - 0.4 / 2.0, 0.25, 0.4, 0.25, 0.25 - 0.4 / 2.0])
    kernel_filter = np.outer(kernel, kernel)
    filtered_image = cv2.filter2D(image, -1, kernel_filter)
    reduced_filtered_image = filtered_image[::2, ::2]

    return reduced_filtered_image


# Generate Gaussian pyramid
def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.
    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.
    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.
    All images in the pyramid should floating-point with values in
    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.
    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pyramid_list = [image]
    temp = np.copy(image)
    for i in range(1, levels):
        temp = reduce_image(temp)
        pyramid_list.append(temp)

    return pyramid_list


# Generate Laplacian pyramid
def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.
    This method uses expand_image() at each level.
    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().
    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    pyramid_list = []

    for i in range(len(g_pyr)-1):
        gau_frame = g_pyr[i]
        lap_exp = expand_image(g_pyr[i + 1])

        if gau_frame.shape[0] < lap_exp.shape[0]:
            lap_exp = np.delete(lap_exp, (-1), axis=0)

        if gau_frame.shape[1] < lap_exp.shape[1]:
            lap_exp = np.delete(lap_exp, (-1), axis=1)

        pyramid_list.append(gau_frame - lap_exp)

    pyramid_list.append(g_pyr[-1])

    return pyramid_list


# Non-max Suppression
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# Digit marking function
def digit_mark(image, marker, digit, thickness=1):
    """Draws a box around the given digit location and mark it with a text.

    Args:
        image (numpy.array): image array of uint8 values.
        marker(tuple): top-left corner location of the box.
        digit(int): digit.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with a box and digit marker drawn.
    """
    x, y = marker
    img_temp = np.copy(image)
    cool_color = (0, 127, 255)

    cv2.line(img_temp, (x, y), (x + 32, y), color=cool_color, thickness=thickness)
    cv2.line(img_temp, (x, y), (x, y + 32), color=cool_color, thickness=thickness)
    cv2.line(img_temp, (x + 32, y), (x + 32, y + 32), color=cool_color, thickness=thickness)
    cv2.line(img_temp, (x, y + 32), (x + 32, y + 32), color=cool_color, thickness=thickness)

    img_temp = cv2.putText(img_temp, str(digit), (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)

    return img_temp
