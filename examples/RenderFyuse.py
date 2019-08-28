# CUDA_VISIBLE_DEVICES=0 python3 RenderFyuse.py
# Might fail on some GPUs with a cublas error!!! 
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
        scenemodel_path = fyuse_dir
        assert os.path.exists(scenemodel_path), "Scenemodel {} not found!".format(scenemodel_path)
        with open(scenemodel_path, 'r') as f:
            poses = json.load(f)
        # crops_path = os.path.join(root_dir, fyuse_id, "Crops")
        self.poses = {}
        ccount = 0
        for frame_num, pose in enumerate(poses['trajectory']['measurements']):
            transforms = {}

            # Distortion is [k1, k2, p1, p2, k3] reference:
            # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
            transforms['distortion'] = torch.Tensor(pose['anchor']['camera']['intrinsics'][-4:] + [0.])

            intrinsics = torch.Tensor(pose['anchor']['camera']['intrinsics'][:5])
            K = [
                [intrinsics[0], intrinsics[2], intrinsics[3]-0.5], [0, intrinsics[1], intrinsics[4] + 420-0.5], [0, 0, 1]
            ]


            transforms['K'] = torch.Tensor(K)
            transforms['Rt'] = torch.Tensor(pose['anchor']['transform']).reshape(4, 4).inverse()

            self.poses[frame_num] = transforms
            ccount += 1
    def __getitem__(self, idx):
        # Returns [R|t], K, distortion coeffs [k1,k2,p1,p2]
        # return self.poses[idx]['Rt'], self.poses[idx]['K'], self.poses[idx]['distortion']
        try:
            return self.poses[idx]
        except :
            raise StopIteration


    def __len__(self):
        return len(self.poses)


def LoadSemanticMesh(filename) :
    vertices = []
    faces = []
    f = open(filename, 'r')
    for line in f :
        if line.strip() == 'end_header' :
            break
        if 'vertex ' in line :
            numVert = int(line.strip().split(' ')[-1])
        if 'face ' in line :
            numFace = int(line.strip().split(' ')[-1])


    for ii in range(numVert) :
        line = f.readline().strip().split(' ')
        vertices.append([float(line[0]), float(line[1]), float(line[2])])
    for ii in range(numFace) :
        line = f.readline().strip().split(' ')
        faces.append([float(line[0]), float(line[1]), float(line[2])])        

    f.close()

    return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype = torch.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseInfo', type=str, default='/media/intelssd/akar/mesh_seg_dataset/Normalized/ScenemodelFiles/0c2ccss8lg_scenemodel_raw.json')
    parser.add_argument('--meshIn', type=str, default='/media/intelssd/akar/mesh_seg_dataset/SemanticMeshes/0c2ccss8lg_semantic_mesh.ply')
    parser.add_argument('--outputDir', type=str, default='../data/results/output_render')
    parser.add_argument('--imageSize', type=int, default=256)
    parser.add_argument('--origImageSize', type=int, default=1920)
    args = parser.parse_args()
    torch.set_printoptions(profile="full")
    
    os.makedirs(args.outputDir, exist_ok=True)

    #Read in data ./ 

    poseData = PoseReader(args.poseInfo)
    numFrames = len(poseData)
    
    print ('No of frames : ',numFrames)

    projMatrices = []
    
    for pose in poseData :
        projMat = torch.mm(pose['K'],pose['Rt'][0:3,:])
        projMatrices.append(projMat)
    projMatrices = torch.stack(projMatrices)
    projMatrices = projMatrices.cuda()


    print('Loading Mesh..')
    vertices, faces = LoadSemanticMesh(args.meshIn)
    
    vertices = vertices.repeat(numFrames,1).reshape(numFrames,-1,3).cuda()
    faces = faces.repeat(numFrames,1).reshape(numFrames,-1,3).cuda()
    print ('Mesh Loaded')
    renderer = sr.SoftRenderer(image_size=args.imageSize, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='projection', P=projMatrices, orig_size=args.origImageSize)

    meshOut = sr.Mesh(vertices, faces)
    imagesPred = renderer.render_mesh(meshOut)
    print ('Rendered Mesh.. Now saving Images')
    writer = imageio.get_writer(os.path.join(args.outputDir, 'render.gif'), mode='I')
    images = imagesPred.detach().cpu().numpy()

    for ii in range(numFrames) : 
        image = images[ii].transpose((1,2,0))
        img = (255 - 255*image[...,-1]).astype(np.uint8)
        imageio.imsave(os.path.join(args.outputDir, 'render%05d.png'%ii), img)
        writer.append_data(img)
    




if __name__ == '__main__':
    main()