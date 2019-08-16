import torch
import argparse
import random
import os
import models
import tqdm
import imageio
from PIL import Image
import glob
#import utils
import torch.nn as nn
import numpy as np
import torch.optim as optim
import soft_renderer as sr

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import pdb
import traceback
from torch import autograd
import json
import matplotlib.pyplot as plt

def SilhouetteLoss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()

def ParsePoses(dataRoot, fyuseId, pad=420):
    # This is for the new format!!!
    scenemodel_path = os.path.join(dataRoot, fyuseId,'scenemodel_norm.json')
    assert os.path.exists(scenemodel_path), "Scenemodel {} not found!".format(scenemodel_path)
    with open(scenemodel_path, 'r') as f:
        poses = json.load(f)
    allPoses = {}
    viewIds = set([int(x.split("/")[-1].split(".")[0]) for x in glob.glob(os.path.join(dataRoot, fyuseId, "*.jpg"))])

    for frameNum, pose in enumerate(poses['trajectory']['measurements']):
        if frameNum not in viewIds:
            continue
        transforms = {}
        try : 
            # k1, k2, p1, p2, k3
            transforms['distortion'] = np.array(pose['anchor']['camera']['intrinsics'][-4:] + [0.])

            intrinsics = np.array(pose['anchor']['camera']['intrinsics'][:5])
            K = [
                [intrinsics[0], intrinsics[2], intrinsics[3]+0.5], [0, intrinsics[1], intrinsics[4] + pad + 0.5], [0, 0, 1]
            ]

            K = np.array(K)
            Rt = np.linalg.inv(np.array(pose['anchor']['transform']).reshape(4, 4))
            transforms['P'] = np.matmul(K, Rt[0:3,:]).astype('float32')
        except : 
            transforms = {}
            # k1, k2, p1, p2, k3
            transforms['distortion'] = np.array(pose['anchor']['distortion'] + [0.])

            intrinsics = np.array(pose['anchor']['intrinsicsVector'])
            K = [
                [intrinsics[0], intrinsics[2], intrinsics[3]+0.5], [0, intrinsics[1], intrinsics[4] + pad+0.5], [0, 0, 1]
            ]

            K = np.array(K)
            Rt = np.linalg.inv(np.array(pose['anchor']['transform']).reshape(4, 4))
            transforms['P'] = np.matmul(K, Rt[0:3,:]).astype('float32')

        allPoses[frameNum] = transforms

    return allPoses




parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='/media/intelssd/mridul/PositionMaps_improved_unstabilized/test/', help='path to Dataset Root')
parser.add_argument('--fyuse', default='zrcviyu4b4', help='the FyuseId')
parser.add_argument('--experiment', default='/media/intelssd/mridul/Test/', help='Generated Output')
parser.add_argument('--model', default='Model5.pth', help='the path to saved state')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--origImageSize', type=int, default=1920, help='the height / width of the actual image')
parser.add_argument('--pad', type=int, default=420, help='The amount of padding added at top and bottom of image')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

opt = parser.parse_args()
print(opt)
torch.backends.cudnn.enabled = False
opt.experiment = os.path.join(opt.experiment,opt.fyuse)
os.system('mkdir -p {0}'.format(opt.experiment))


stateDict = torch.load(opt.model) 

encoderInit = nn.DataParallel(models.Encoder(), device_ids=[0])
decoderInit = nn.DataParallel(models.Decoder(numVertices=642+1), device_ids=[0])

encoderInit.load_state_dict(stateDict['stateDictEncoder'])
decoderInit.load_state_dict(stateDict['stateDictDecoder'])

encoderInit = encoderInit.cuda()
decoderInit = decoderInit.cuda()

# opEncoderInit = stateDict['optimizerEncoder'] 
# opDecoderInit = stateDict['optimizerDecoder'] 



encoderInit.eval()
decoderInit.eval()

# assuming that data root is the fyuse to be evaluated..
# strategy would be to parse all the images predict meshes and calculate IOU per view!!!

imgInput = []

for file in glob.glob(os.path.join(opt.dataRoot,opt.fyuse,'*.jpg')) :
    Im = Image.open(file)
    im = np.asarray(Im, dtype=np.float32)

    im = (im - 127.5) / 127.5
    if len(im.shape) == 2:
        im = im[:, np.newaxis]
    im = np.transpose(im, [2, 0, 1])
    imgInput.append(im)

imgInput = torch.tensor(np.stack(imgInput)).cuda()

currBatchSize = imgInput.shape[0]

imgMasks = []
namesF = []
for file in glob.glob(os.path.join(opt.dataRoot,opt.fyuse,'*.png')) :
    Im = imageio.imread(file)
    im = np.asarray(Im).astype('float32')/255.0
    im = im[np.newaxis]        
    imgMasks.append(im)
    namesF.append(int(file.split('/')[-1].replace('.png','')))

imgMasks = torch.tensor(np.stack(imgMasks)).cuda()



features = encoderInit(imgInput)
outPos = decoderInit(features)

# Load Projection Matrices...

dataProjectionMat = ParsePoses(opt.dataRoot, opt.fyuse,)
projViews = []

for ii in range(currBatchSize): 
    projViews.append(dataProjectionMat[namesF[ii]]['P'])
    
projViews = torch.tensor(np.stack(projViews)).cuda()


# Load Mesh...
vertices = []
faces = []
f = open(os.path.join(opt.dataRoot,'template_mesh.obj'),'r')
for line in f:
    words = line.split()
    if len(words) == 0:
        continue
    if words[0] == 'v':
        vertices.append([float(v) for v in words[1:4]])

    if words[0] == 'f':
        v0 = int(words[1])
        v1 = int(words[2])
        v2 = int(words[3])
        faces.append([v0, v1, v2])
vertices = torch.tensor(np.vstack(vertices).astype(np.float32)).cuda()
faces = torch.tensor(np.vstack(faces).astype(np.int32) - 1).cuda() ##### ASSUMING START FROM 1
f.close()

faces = faces.repeat(currBatchSize,1,1)
vertices = vertices.repeat(currBatchSize,1,1)
print(vertices.shape)
meshM = models.MeshModel(faces, vertices).cuda()
outCols = torch.tensor([[1,0,0]]*642,dtype=torch.float32).repeat(currBatchSize,1,1).cuda()
meshDeformed, _, _ = meshM.forward(outPos[:,:-1,:], torch.zeros_like(outPos[:,-1:,:]).cuda(), 1, currBatchSize, outCols)
meshDeformed.save_objs(opt.experiment)
renderer = sr.SoftRenderer(opt.imageSize, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='projection', P=projViews, orig_size=opt.origImageSize)
imagesPred = renderer.render_mesh(meshDeformed)

lossSS = SilhouetteLoss(imagesPred[:, 3, :, :], imgMasks[:,0,:,:])
print (lossSS.item())
ss = []
for ii in range(currBatchSize) :
    lossSS = SilhouetteLoss(imagesPred[ii:ii+1, 3, :, :], imgMasks[ii:ii+1,0,:,:])
    ss.append(lossSS.item())

plt.plot(range(len(ss)), ss, 'ro')
plt.xlabel("Frame")
plt.ylabel("IOU Loss")

plt.savefig(os.path.join(opt.experiment, opt.fyuse+'sil.png'),dpi=300)


# Save generated Result!
images = imagesPred.detach().cpu().numpy()
imagesGt = imgMasks.detach().cpu().numpy() 


numFrames = currBatchSize
globalImg = 255 * np.ones((opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.uint8)
globalImgGt = np.zeros((opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.uint8)
# globalColViews = np.zeros((3,opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.float32)
# globalColViewsGt = np.zeros((3,opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.float32)
for i in range(numFrames) : 
    col = int(i % 5)
    row = int(i / 5)
    image = images[i].transpose((1,2,0))
    globalImg[row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = (255 - 255*image[...,-1]).astype(np.uint8)
    globalImgGt[row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = (127.5*imagesGt[i]).astype(np.uint8)
    # globalColViewsGt[:,row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = colImagesGt[i]
    # globalColViews[:,row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = images[i,0:3,:,:]

imageio.imsave(os.path.join(opt.experiment, opt.fyuse+'_deform_%05d.png'%ii), globalImg+globalImgGt)
imageio.imsave(os.path.join(opt.experiment, opt.fyuse+'_groundT_%05d.png'%ii), globalImgGt)
# imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_groundTCol_%05d.jpg'%ii), globalColViewsGt.astype(np.uint8).transpose(1,2,0))
# imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_DeformCol_%05d.jpg'%ii), (255*globalColViews).astype(np.uint8).transpose(1,2,0))