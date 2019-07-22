# CUDA_VISIBLE_DEVICES=1 python3 -O examples/DeformFyuse.py --output_dir ~/Code/SoftRas/data/results/output_deform1 --template_mesh ~/Code/SoftRas/data/obj/car/meshS124.obj --image_size 256 --template_mesh ~/Code/SoftRas/data/obj/sphere/sphere_642_s.obj
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
    def __init__(self, fyuse_dir, scale_from_full_hd=1.0):
        scenemodel_path = os.path.join(fyuse_dir, "scenemodel_raw_refined.json")
        assert os.path.exists(scenemodel_path), "Scenemodel {} not found!".format(scenemodel_path)
        with open(scenemodel_path, 'r') as f:
            poses = json.load(f)
        # crops_path = os.path.join(root_dir, fyuse_id, "Crops")
        jpg_ids = set([int(x.split("/")[-1].split(".")[0]) for x in glob.glob(fyuse_dir + "GtRedRed/*.jpg")])
        if len(jpg_ids) == 0 : 
            jpg_ids = set([int(x.split("/")[-1].split(".")[0]) for x in glob.glob(fyuse_dir + "GtRedRed/*.png")])
        # Make a data structure that has a dictionary with distortion matrix, intrinsics and [R|t] that is looked
        # up by frame id
        self.poses = {}
        for frame_num, pose in enumerate(poses['trajectory']['measurements']):
            if frame_num not in jpg_ids:
                # Keep only poses for frames that matter
                continue
            # Only load in the poses that have corresponding frames in Crops folder!! TODO
            transforms = {}

            # Distortion is [k1, k2, p1, p2, k3] reference:
            # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
            transforms['distortion'] = torch.Tensor(pose['anchor']['distortion'] + [0.])

            intrinsics = torch.Tensor(pose['anchor']['intrinsicsVector'])
            K = [
                [intrinsics[0], intrinsics[2], intrinsics[3]-0.5], [0, intrinsics[1], intrinsics[4] + 420-0.5], [0, 0, 1]
            ]


            transforms['K'] = torch.Tensor(K)
            transforms['Rt'] = torch.Tensor(pose['anchor']['transform']).reshape(4, 4).inverse()

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
        self.register_buffer('vertices', self.template_mesh.vertices)
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

        return sr.Mesh(vertices.repeat(batch_size, 1, 1), 
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss


def SilhouetteLoss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, default='/home/mridul/Code/FyuseMesh/uxgpbaxotx/uxgpbaxotx/')
    parser.add_argument('-t', '--template_mesh', type=str, default=os.path.join(current_dir, '../data/obj/car/meshS124.obj'))
    parser.add_argument('-o', '--output_dir', type=str, default=os.path.join(current_dir, '../data/results/output_deform'))
    parser.add_argument('-is', '--image_size', type=int, default=256)
    parser.add_argument('-os', '--orig_image_size', type=int, default=1920)
    args = parser.parse_args()
    torch.set_printoptions(profile="full")
    
    os.makedirs(args.output_dir, exist_ok=True)

    model = Model(args.template_mesh).cuda()

    #Read in data ./ 

    poseData = PoseReader(args.data_dir)
    numFrames = len(poseData)
    
    print ('No of frames : ',numFrames)
    imagesGT = []
    projMatrices = []
    distCoeffs = []
    flagSavedGT = False
    lPoseData = list(poseData.poses)
    for ii in range(numFrames) :
        idx = lPoseData[ii]
        projMat = torch.mm(poseData.poses[idx]['K'],poseData.poses[idx]['Rt'][0:3,:])
        try :
            imgGt = imageio.imread(args.data_dir+'GtRedRed/{0:08d}'.format(idx)+'.png').astype('float32') / 255.0
        except :
            try :
                imgGt = imageio.imread(args.data_dir+'GtRedRed/{0:09d}'.format(idx)+'.jpg').astype('float32') / 255.0
            except e:
                raise(e)
        imagesGT.append(imgGt)
        projMatrices.append(projMat)
        distCoeffs.append(poseData.poses[idx]['distortion'])


    projMatrices = torch.stack(projMatrices)
    distCoeffs = torch.stack(distCoeffs)
    if __debug__:
        print(projMatrices)
    
    projMatrices = projMatrices.cuda()
    print (projMatrices)
    renderer = sr.SoftRenderer(image_size=args.image_size, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='projection', P=projMatrices, dist_coeffs=distCoeffs, orig_size=args.orig_image_size)
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))


    loop = tqdm.tqdm(list(range(0, 2000)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    images_gt = torch.from_numpy(np.array(imagesGT)).cuda()
    

    for i in loop:
        mesh, laplacian_loss, flatten_loss = model(numFrames)
        images_pred = renderer.render_mesh(mesh)

        if __debug__:
            print(images_pred.shape)
            print (SilhouetteLoss(images_pred[:,3], images_gt), laplacian_loss, flatten_loss)
            for ii in range(numFrames) : 
                imageio.imsave(os.path.join(args.output_dir, 'Debug/pred_%05d.png'%ii), (255*images_pred[ii, 3,:,:].detach().cpu().numpy()).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'Debug/gt_%05d.png'%ii), (255*imagesGT[ii]).astype(np.uint8))

            break   

        # optimize mesh with silhouette reprojection error and geometry constraints
 
        loss = SilhouetteLoss(images_pred[:, 3], images_gt) + 0.3 * laplacian_loss + 0.0003 * flatten_loss 
        loop.set_description('Loss: %.4f'%(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not flagSavedGT : 
            globalImgGt = np.zeros((args.image_size*int(numFrames/10 + 1),args.image_size*10), dtype=np.uint8)

        
        if i % 100 == 0:
            images = images_pred.detach().cpu().numpy()
            globalImg = 255 * np.ones((args.image_size*int(numFrames/10 + 1),args.image_size*10), dtype=np.uint8)

            for ii in range(numFrames) : 
                col = int(ii % 10)
                row = int(ii / 10)
                image = images[ii].transpose((1,2,0))
                globalImg[row*args.image_size:row*args.image_size + args.image_size,col*args.image_size:col*args.image_size + args.image_size] = (255 - 255*image[...,-1]).astype(np.uint8)
                if not flagSavedGT : 
                    globalImgGt[row*args.image_size:row*args.image_size + args.image_size,col*args.image_size:col*args.image_size + args.image_size] = (128*imagesGT[ii]).astype(np.uint8)

            writer.append_data(globalImg+globalImgGt)
            imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), globalImg+globalImgGt)

            # save optimized mesh
            model(1)[0].save_obj(os.path.join(args.output_dir, 'car.obj'), save_texture=False)
            if not flagSavedGT : 
                imageio.imsave(os.path.join(args.output_dir, 'groundT_%05d.png'%i), globalImgGt)
                flagSavedGT = True

if __name__ == '__main__':
    main()

