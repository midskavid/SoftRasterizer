import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold


class GProjection(nn.Module):
    """
    Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bi-linear interpolation to get the corresponding feature.
    """

    def __init__(self, mesh_pos=[0.,0.,-0.8], bound=0):
        super(GProjection, self).__init__()
        self.mesh_pos = mesh_pos
        self.threshold = None
        self.bound = 0
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x

    @staticmethod
    def image_feature_shape(img):
        return np.array([img.size(-1), img.size(-2)])

    def forward(self, resolution, img_features, inputs, camK):
        half_resolution = torch.tensor((resolution - 1) / 2, device=inputs.device)
        
        camK = camK*(256./1920.) # bring to pixel space...
        # map to [-1, 1]
        # not sure why they render to negative x
        positions = inputs + torch.tensor(self.mesh_pos, device=inputs.device, dtype=torch.float)
        w = (-camK[:,0,0:1]*positions[:, :, 0] -camK[:,0,1:2]*positions[:, :, 1])/self.bound_val(positions[:, :, 2]) + camK[:,0,2:3] - half_resolution[0]
        h = camK[:,1,1:2]*(positions[:, :, 1] / self.bound_val(positions[:, :, 2])) + camK[:,1,2:3] - half_resolution[1]

        w /= half_resolution[0]
        h /= half_resolution[1]

        # clamp to [-1, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        feats = [inputs]
        for img_feature in img_features:
            feats.append(self.project(resolution, img_feature, torch.stack([w, h], dim=-1)))

        output = torch.cat(feats, 2)

        return output

    def project(self, img_shape, img_feat, sample_points):
        """
        :param img_shape: raw image shape
        :param img_feat: [batch_size x channel x h x w]
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """
        output = F.grid_sample(img_feat, sample_points.unsqueeze(1))
        output = torch.transpose(output.squeeze(2), 1, 2)

        return output
