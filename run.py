import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as nnf

from vgg_16 import VGG_16
from utils import mser, reduce_image, gaussian_pyramid, non_max_suppression, digit_mark

# TO_DO:

# 3) Construct pyramid of size 5 (1 0.5 0.25 0.125 0.0625) - Don't process the first one
# 4) Maybe apply mser on each of the windowed image to see if it gets any better? <Optional>
# 5) After processing everything, gather bunch of windows and use non-max suppression to determine which is correct
# Extra: Use gpu_available variable always + Get nice 5 images to demonstrate and push them


# Load the image + Create the image pyramid
image_path = str(sys.argv[1])
image_orig = cv2.imread(image_path)
image_pyramid = gaussian_pyramid(image_orig, 5)


# Check NVDIA GPU Availability
gpu_available = True if torch.cuda.is_available() else False
checkpoint_path = "pretrained_weights/vgg_16_wgts.pth" if gpu_available else "pretrained_weights/vgg_16_cpu_wgts.pth"

# Load the pre-trained VGG 16 weights on the model
model = VGG_16(11)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Create directory if not exist
save_dir = "images/"
is_directory = os.path.isdir(save_dir)
if is_directory == False:
    os.mkdir(save_dir)

        
# # Resize the image in order to make the digit in a window to be fit to 32x32 size
# # Eventually, will be replaced by Gaussian pyramid
image_resized = image_pyramid[2]
# cv2.imwrite("sample.png", image_resized)
image = np.swapaxes(image_resized, 0, 2)
image = np.swapaxes(image, 1, 2)

# Window sliding through the entire image
_, height, width = image.shape
print("height: ", height)
print("width: ", width)
windows_R, windows_G, windows_B, windows_pos = [], [], [], []
windows_num = 0
for i in range(0, height-32, 5):
    for j in range(0, width-32, 5):
        windows_R.append(image[0][i:i+32, j:j+32])
        windows_G.append(image[1][i:i+32, j:j+32])
        windows_B.append(image[2][i:i+32, j:j+32])
        windows_pos.append([j, i]) 

windows_R = np.expand_dims(np.stack(windows_R, axis=0).astype(np.float32),axis=3)
windows_G = np.expand_dims(np.stack(windows_G, axis=0).astype(np.float32),axis=3)
windows_B = np.expand_dims(np.stack(windows_B, axis=0).astype(np.float32),axis=3)
windows_pos = np.stack(windows_pos)
windows_3ch = np.concatenate((windows_R, windows_G, windows_B), axis=3)
windows_3ch_axes = np.swapaxes(windows_3ch, 1, 3)
windows_3ch_Tensor = Tensor(np.swapaxes(windows_3ch_axes, 2, 3))


#####################################################################################
# count = 0
# for image in windows_3ch:
#     path = save_dir + str(count) + '.png'
#     print(path)
#     cv2.imwrite(path, image)
#     count += 1
#####################################################################################

# Use the pre-trained model to predict the windows
output = model(windows_3ch_Tensor)
output_prob = nnf.softmax(output, dim=1)
prediction_prob, prediction = output_prob.topk(1, dim=1)
prediction_prob, prediction = prediction_prob.detach().numpy(), prediction.numpy()

# Calculate the average position of the detected digits within the image
digits = []
detected_digits = []
detected_digits_indices = []
detected_indices = []
detected_indices_TF = np.logical_and(prediction != 10, prediction_prob > 0.9)
for i in range(len(detected_indices_TF)):
    if detected_indices_TF[i]:
        detected_indices.append(i)
for detected_index in detected_indices:
    digit = prediction[detected_index].argmax()
    if digit not in digits:
        digits.append(digit)
    detected_digits.append(digit)
    detected_digits_indices.append(detected_index)
print(detected_digits)

# Marking each of the digits(s) into the original image
while len(digits) != 0:
    digit = digits.pop(0)
    x_list, y_list = [], []     
    for i in range(len(detected_digits)):
        if digit == detected_digits[i]:
            pos_idx = detected_digits_indices[i]
            x, y = windows_pos[pos_idx]             
            x_list.append(x)
            y_list.append(y)     
    x_mean = int(np.mean(x_list))
    y_mean = int(np.mean(y_list))
    image_resized = digit_mark(image_resized, (x_mean, y_mean), digit) 
    image_boxed = cv2.resize(image_resized, dsize=(0,0), fx=3, fy=3)
image_name = save_dir + 'case' + ".png"
cv2.imwrite(image_name, image_boxed)

