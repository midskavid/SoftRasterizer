import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import soft_renderer as sr
import losses

from Backbones import get_backbone
from Layers.gbottleneck import GBottleneck
from Layers.gconv import GConv
from Layers.gpooling import GUnpooling
from Layers.gprojection import GProjection


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # RGB image and a mask...
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(262144, 1024, bias=False)
        self.bn4 = nn.BatchNorm1d(1024)
        self.ac1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)
        self.ac2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.ac3 = nn.LeakyReLU()
    
    def forward(self, x) :
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = x3.view(x3.shape[0], -1)
        x4 = self.ac1(self.bn4(self.fc1(x3)))
        x5 = self.ac2(self.bn5(self.fc2(x4)))
        x6 = self.ac3(self.bn6(self.fc3(x5)))

        return x6


class Decoder(nn.Module):
    def __init__(self, numVertices=642):
        super(Decoder, self).__init__()
        self.numVertices = numVertices
        self.fc1 = nn.Linear(512, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.ac1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.ac2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(1024, numVertices*3)
        self.bn3 = nn.BatchNorm1d(numVertices*3)
    
    def forward(self, x) :
        x1 = self.ac1(self.bn1(self.fc1(x)))
        x2 = self.ac2(self.bn2(self.fc2(x1)))
        # x3 = self.ac3(self.bn3(self.fc3(x2)))
        x3 = self.fc3(x2)
        return x3.reshape((-1,self.numVertices,3))



class Color(nn.Module) : 
    def __init__(self, numVertices=642):
        super(Color, self).__init__()
        self.numVertices = numVertices
        self.fc1 = nn.Linear(512, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.ac1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, numVertices*3)
    
    def forward(self, x) :
        x1 = self.ac1(self.bn1(self.fc1(x)))
        x2 = torch.sigmoid(self.fc2(x1)) # map colors to [0,1]
        return x2.reshape((-1,self.numVertices,3)) 



class MeshModel(nn.Module):
    def __init__(self, faces, vertices):
        super(MeshModel, self).__init__()
        # set template mesh
        # Assuming faces, vertices are batch, num, 3
        # The mesh class they have is such that they work with multiple meshes.. 
        # So the vertices and faces here would be for multiple(batchNum) meshes
        self.template_mesh = sr.Mesh(vertices, faces)
        self.vertices = self.template_mesh.vertices
        self.faces = self.template_mesh.faces
        #TODO : check if this works for multiple meshes..
        # Mesh connectivitiy is same!!!
        self.laplacian_loss = losses.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = losses.FlattenLoss(self.faces[0].cpu())


    def forward(self, displace, center, numViews, numBatch, texture):
    	# center, displace would be batchx3, batchxnumvertx3

        #vertices = self.vertices + displace + center

        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(center)
        vertices = torch.sigmoid(base + displace) * torch.sign(self.vertices)
        # need to figure out this transformation
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices.repeat(1, numViews, 1).reshape(numViews*numBatch, -1, 3), self.faces.repeat(1, numViews, 1).reshape(numViews*numBatch, -1, 3), textures=texture.repeat(1, numViews, 1).reshape(numViews*numBatch, -1, 3), texture_type='vertex'), laplacian_loss, flatten_loss



class P2MModel(nn.Module):

    def __init__(self, ellipsoid, pathResNet, mesh_pos=[0.,0.,-0.8]):
        super(P2MModel, self).__init__()

        self.hidden_dim = 192
        self.coord_dim = 3
        self.last_hidden_dim = 192
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = True

        self.nn_encoder, self.nn_decoder = get_backbone(pathResNet)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)
        ])

        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]),
            GUnpooling(ellipsoid.unpool_idx[1])
        ])

        self.projection = GProjection(mesh_pos, bound=0)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

    def forward(self, img, camK):
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_shape, img_feats, init_pts, camK)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, img_feats, x1, camK)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2, camK)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_feats)
        else:
            reconst = None

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst
        }
