import torch
import os
import tqdm
import imageio
import argparse
import json
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import soft_renderer as sr

current_dir = os.path.dirname(os.path.realpath(__file__))


class PoseReader():
    def __init__(self, fyuse_dir, scale_from_full_hd=0.5):
        scenemodel_path = os.path.join(fyuse_dir, "scenemodel_raw_refined.json")
        assert os.path.exists(scenemodel_path), "Scenemodel {} not found!".format(scenemodel_path)
        with open(scenemodel_path, 'r') as f:
            poses = json.load(f)
        # crops_path = os.path.join(root_dir, fyuse_id, "Crops")
        jpg_ids = set([int(x.split("/")[-1].split(".")[0]) for x in glob.glob(fyuse_dir + "Crops/*.jpg")])
        # Make a data structure that has a dictionary with distortion matrix, intrinsics and [R|t] that is looked
        # up by frame id
        self.poses = {}
        for frame_num, pose in enumerate(poses['trajectory']['measurements']):
            if frame_num not in jpg_ids:
                # Keep only poses for frames that matter
                continue
            # Only load in the poses that have corresponding frames in Crops folder!! TODO
            transforms = {}
            transforms['distortion'] = torch.Tensor(pose['anchor']['distortion'])
            # Distortion is [k1, k2, p1, p2] reference:
            # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
            intrinsics = torch.Tensor(pose['anchor']['intrinsicsVector']) * scale_from_full_hd
            # Account for the half pixel offset:
            K = [
                [intrinsics[0], intrinsics[2], intrinsics[3] - 0.5], [0, intrinsics[1], intrinsics[4] - 0.5], [0, 0, 1]
            ]

            # K would be [fx skew cx; 0 fy cy; 0 0 1]
            # intrinsicsVector contains [fx,fy,skew,cx,cy]
            transforms['K'] = torch.Tensor(K)
            transforms['Rt'] = torch.Tensor(pose['anchor']['transform']).reshape(4, 4)
            self.poses[frame_num] = transforms

    def __getitem__(self, idx):
        # Returns [R|t], K, distortion coeffs [k1,k2,p1,p2]
        # return self.poses[idx]['Rt'], self.poses[idx]['K'], self.poses[idx]['distortion']
        return self.poses[idx]

    def __len__(self):
        return len(self.poses)



class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 0.5)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())

    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices.repeat(batch_size, 1, 1), self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss


def SilhouetteLoss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims) + 1e-6
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data-dir', type=str, default='/home/mridul/Code/FyuseMesh/uxgpbaxotx/uxgpbaxotx/')
    parser.add_argument('-t', '--template-mesh', type=str, default=os.path.join(current_dir, '../data/obj/sphere/sphere_642.obj'))
    parser.add_argument('-o', '--output-dir', type=str, default=os.path.join(current_dir, '../data/results/output_deform'))
    parser.add_argument('-b', '--batch-size', type=int, default=120)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = Model(args.template_mesh).cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))


    #Read in data ./ 

    poseData = PoseReader(args.data_dir)
    numFrames = len(poseData)

    imagesGT = []
    projMatrices = []

    for ii in range(numFrames) :
        # print (poseData.poses.keys())
        idx = list(poseData.poses)[ii]
        projMat = torch.mm(poseData.poses[idx]['K'],poseData.poses[idx]['Rt'][0:3,:])
        # print (projMat.shape)
        imgGt = imageio.imread(args.data_dir+'Gt/{0:09d}'.format(idx)+'.jpg').astype('float32') / 255.0
        # swap axes ??
        imagesGT.append(imgGt)
        projMatrices.append(projMat)


    projMatrices = torch.stack(projMatrices)
    projMatrices = projMatrices.cuda()
    
    renderer = sr.SoftRenderer(image_size=256, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='projection', P=projMatrices)#, viewing_angle=12.5)

    loop = tqdm.tqdm(list(range(0, 2000)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    images_gt = torch.from_numpy(np.array(imagesGT)).cuda()
    

    for i in loop:
        mesh, laplacian_loss, flatten_loss = model(numFrames)
        images_pred = renderer.render_mesh(mesh)

        #print(images_pred.shape)
        # optimize mesh with silhouette reprojection error and 
        # geometry constraints
        #print (SilhouetteLoss(images_pred[:,3], images_gt), laplacian_loss, flatten_loss)
        loss = SilhouetteLoss(images_pred[:,3], images_gt) + 0.03 * laplacian_loss + 0.0003 * flatten_loss

        loop.set_description('Loss: %.4f'%(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            image = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            writer.append_data((255*image).astype(np.uint8))
            imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), (255*image[..., -1]).astype(np.uint8))
            # save optimized mesh
            model(1)[0].save_obj(os.path.join(args.output_dir, 'plane.obj'), save_texture=False)


if __name__ == '__main__':
    main()

