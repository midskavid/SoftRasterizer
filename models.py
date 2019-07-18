import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import soft_renderer as sr

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


class Model():
    def __init__(self, faces, vertices):
        # set template mesh
        # Assuming faces, vertices are batch, num, 3
        self.template_mesh = sr.Mesh(vertices, faces)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)

    def forward(self, batch_size, center, displace):
    	# would be batchx3 batchxnumvertx3
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        # need to figure out these two transformations
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        return sr.Mesh(vertices.repeat(batch_size, 1, 1), self.faces.repeat(batch_size, 1, 1))

