import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # RGB image and a mask...
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.fc1 = nn.Linear(1048576, 1024)
		self.bn4 = nn.BatchNorm1d(1024)
		self.ac1 = nn.LeakyReLU(negative_slope=0.2)
		self.fc2 = nn.Linear(1024, 1024)
		self.bn5 = nn.BatchNorm1d(1024)
		self.ac2 = nn.LeakyReLU(negative_slope=0.2)
		self.fc3 = nn.Linear(1024, 512)
		self.bn6 = nn.BatchNorm1d(512)
		self.ac3 = nn.LeakyReLU(negative_slope=0.2)

class Decoder(nn.Module):
    def __init__(self, numVertices=642):
        super(decoder, self).__init__()
		self.fc1 = nn.Linear(512, 1024)
		self.bn1 = nn.BatchNorm1d(1024)
		self.ac1 = nn.LeakyReLU(negative_slope=0.2)
		self.fc2 = nn.Linear(1024, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.ac2 = nn.LeakyReLU(negative_slope=0.2)
		self.fc3 = nn.Linear(1024, numVertices*3)
		self.bn3 = nn.BatchNorm1d(numVertices*3)
		self.ac3 = nn.LeakyReLU(negative_slope=0.2)
