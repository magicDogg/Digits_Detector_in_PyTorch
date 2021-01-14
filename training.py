from scipy.io import loadmat
import matplotlib.pyplot as plt
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

from vgg_16 import VGG_16


# Load the SVHN Dataset as Digit Classes
training_set = loadmat('data/train_32x32.mat')
testing_set = loadmat('data/test_32x32.mat')

X_train = training_set['X']
X_test = testing_set['X']

X_train = np.swapaxes(X_train, 0, 3)
X_train = np.swapaxes(X_train, 1, 2)
X_train = np.swapaxes(X_train, 2, 3)

X_test = np.swapaxes(X_test, 0, 3)
X_test = np.swapaxes(X_test, 1, 2)
X_test = np.swapaxes(X_test, 2, 3)

y_train_raw = training_set['y']
y_test_raw = testing_set['y']

y_train = np.zeros((y_train_raw.shape[0], 1), dtype=int)
y_test = np.zeros((y_test_raw.shape[0], 1), dtype=int)

for i in range(y_train_raw.shape[0]):
  for j in range(10):
    y_train[i] = y_train_raw[i][0]

for i in range(y_test_raw.shape[0]):
  for j in range(10):
    y_test[i] = y_test_raw[i][0]


# Load the CIFAR-100 Dataset as Non-digit Classes
nd_training_set = loadmat('data/train.mat')
nd_testing_set = loadmat('data/test.mat')

X_train_nd = nd_training_set['data']
X_train_nd = np.reshape(X_train_nd, (X_train_nd.shape[0], 3, 32, 32))

X_test_nd = nd_testing_set['data']
X_test_nd = np.reshape(X_test_nd, (X_test_nd.shape[0], 3, 32, 32))

y_train_nd = np.zeros([X_train_nd.shape[0], 1]) + 10
y_test_nd = np.zeros([X_test_nd.shape[0], 1]) + 10

# Merge the training and testing sets to complete it
X_train = np.concatenate((X_train, X_train_nd), axis=0, out=None)
y_train = np.concatenate((y_train, y_train_nd), axis=0, out=None)

X_test = np.concatenate((X_test, X_test_nd), axis=0, out=None)
y_test = np.concatenate((y_test, y_test_nd), axis=0, out=None)

# Reformat the dataset into PyTorch dataloader format
training_dataset = TensorDataset(Tensor(X_train), Tensor(y_train.squeeze()).long())
training_dataset_loader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=8)

test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test.squeeze()).long())
test_dataset_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8)


# Create VGG_16 Handler & Load it into GPU
vgg16 = VGG_16(11)

if torch.cuda.is_available():
  print("Using CUDA GPU")
  vgg16.cuda()

# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# Train
for epoch in range(25):  # loop over the dataset multiple times
    print("Epoch #: ", (epoch+1))
    num_minibatches = 0

    running_loss = 0.0
    for i, data in enumerate(training_dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        num_minibatches += 1

        if torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # # print statistics
        # running_loss += loss.item()
        # if i % 64 == 63:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 64))
        #     running_loss = 0.0

    running_loss = loss.item() / (64 * num_minibatches)
    print("Loss: ", loss.item())
    running_loss = 0.0

print('Training Complete')


# Test the performance
class_correct, class_total = 0, 0
with torch.no_grad():
    for i, data in enumerate(test_dataset_loader, 0):
        images, labels = data
        if torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()

        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for j in range(len(c)):
          class_total += 1
          if c[j].item():
            class_correct += 1

print('Accuracy : ', (100 * class_correct / class_total))


# Save the weights
save_dir = "pretrained_weights/"
is_directory = os.path.isdir(save_dir)
if is_directory == False:
    os.mkdir(save_dir)

PATH = save_dir + "vgg_16_wgts.pth"
PATH_2 = save_dir + "vgg_16_cpu_wgts.pth"
torch.save(vgg16.state_dict(), PATH)
torch.save(vgg16.cpu().state_dict(), PATH_2)
