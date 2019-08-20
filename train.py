# For multiGPU export the GPUs as the code inside just pushes everything to GPU 0
# CUDA_VISIBLE_DEVICES=1,2 python3 train.py --cuda --deviceIds 0 1
import torch
import argparse
import random
import os
import models
import losses
import tqdm
import imageio
#import utils
import DataLoader
import DebugHelper
import torch.nn as nn
import numpy as np
import torch.optim as optim
import soft_renderer as sr
import Ellipsoid
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import pdb
import traceback
from torch import autograd


class GuruMeditation (autograd.detect_anomaly):  
    def __init__(self):
        super(GuruMeditation, self).__init__()

    def __enter__(self):
        super(GuruMeditation, self).__enter__()
        return self  

    def __exit__(self, type, value, trace):
        super(GuruMeditation, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            Halt(str(value))

def Halt(msg):
    print (msg)
    pdb.set_trace()


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='/media/intelssd/akar/mesh_seg_dataset/', help='path to Dataset Root')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
parser.add_argument('--resNet', default='Pretrained/resnet50-19c8e357.pth', help='the path to ResNet50')
parser.add_argument('--ellipsoid', default='Ellipsoid/info_ellipsoidN.dat', help='the path to Ellipsoid')
parser.add_argument('--meshPos', type=int, nargs='+', default=[0., 0., 0], help='mesh Pos')
parser.add_argument('--fyuses', default='fyuse_ids.txt', help='the path to fyuseIds')
parser.add_argument('--scale', type=float, default=1.0, help='learning rate scaling')
parser.add_argument('--loadPath', default=None, help='the path for model')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=200, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--numViews', type=int, default=15, help='views for training')
parser.add_argument('--validationSplit', type=float, default=0.1, help='data used for validation')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--origImageSize', type=int, default=1920, help='the height / width of the actual image')
parser.add_argument('--pad', type=int, default=420, help='The amount of padding added at top and bottom of image')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
# The training weight
parser.add_argument('--lamS', type=float, default=1.0, help='weight Silhouette')
parser.add_argument('--lamL', type=float, default=0.5, help='weight Laplacian')
parser.add_argument('--lamM', type=float, default=0.033, help='weight move loss')
parser.add_argument('--lamE', type=float, default=0.1, help='weight Edge loss')

opt = parser.parse_args()
print(opt)
torch.backends.cudnn.enabled = False

opt.gpuId = opt.deviceIds[0]

if opt.experiment is None:
    opt.experiment = 'CheckMeshGen'
os.system('mkdir {0}'.format(opt.experiment))
#Clean Directory
os.system('rm -r {0}/*'.format(opt.experiment))
os.system('mkdir {0}/tmp'.format(opt.experiment))
os.system('cp *.py %s' % opt.experiment )

lamS = opt.lamS
lamL = opt.lamL
lamM = opt.lamM
lamE = opt.lamE

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#################################
# initialize models
ellipsoid = Ellipsoid.Ellipsoid(opt.meshPos, opt.ellipsoid)
modelPix2Mesh = torch.nn.DataParallel(models.P2MModel(ellipsoid, opt.resNet, opt.meshPos), device_ids=opt.deviceIds)


####################################
# Initial Optimizer
scale = opt.scale
opModelPix2Mesh = optim.Adam(modelPix2Mesh.parameters(), lr=1e-3 * scale, betas=(0.5, 0.999))

#####################################
# Load Model
if opt.load is not None : 
    stateDict = torch.load(opt.loadPath) 
    modelPix2Mesh.load_state_dict(stateDict['stateDictPix2Mesh'])
    opModelPix2Mesh.load_state_dict(stateDict['optimizerPix2Mesh'])

####################################
if opt.cuda :
    modelPix2Mesh.cuda(opt.gpuId)

####################################
# Data Loaders..
fyuseDataset = DataLoader.BatchLoader(opt.dataRoot, opt.fyuses, opt.batchSize, imSize=opt.imageSize, numViews=opt.numViews, padding=opt.pad, debugDir=opt.experiment)
datasetSize = len(fyuseDataset)
indices = list(range(datasetSize))
np.random.shuffle(indices)
split = int(np.floor(opt.validationSplit * datasetSize))

trainIndices, valIndices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
trainSampler = torch.utils.data.SubsetRandomSampler(trainIndices)
validSampler = torch.utils.data.SubsetRandomSampler(valIndices)

# TODO : check the significance of shuffle here...
trainLoader = torch.utils.data.DataLoader(fyuseDataset, batch_size=opt.batchSize, sampler=trainSampler, num_workers = 4, shuffle = False, drop_last=True) # sometimes the last batch is 1 size and batchnorm 1d fails...
validationLoader = torch.utils.data.DataLoader(fyuseDataset, batch_size=1, sampler=validSampler, num_workers = 4, shuffle = False)
dataLoaders = {"train": trainLoader, "val": validationLoader}
dataLengths = {"train": len(trainLoader), "val": len(validationLoader)}
######################################


jj = 0
writer = SummaryWriter(log_dir=opt.experiment)
torch.set_printoptions(profile="full")

criterionP2M = losses.P2MLoss(ellipsoid).cuda()


with GuruMeditation() as gr :
    for epoch in range(opt.nepoch):
        print('Epoch {}/{}'.format(epoch, opt.nepoch - 1))
        print ('===============================')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val'] : #['val', 'train'] ['train', 'val']
            if phase == 'train':
                modelPix2Mesh.train(True)  # Set model to training mode
            else:
                modelPix2Mesh.train(False)  # Set model to evaluate mode

            createMesh = True
            runningLoss = 0.0
            loop = tqdm.tqdm(list(range(dataLengths[phase])), ascii=True)

            # Iterate over data.
            for ii, dataBatch in zip(loop,dataLoaders[phase]):
                # Dataloader would return me Projection matrices and the input images and ground truth images. 
                # The manner in which dataloader creates batch, I will have to reshape them to have batch = batch*numViews
                # For mesh vertices and faces, I will get the correct format batchxnumVertx3.
                # This will be changed to the appropriate format with the forward of the dataloader.
                currBatchSize = len(dataBatch['ImgInput'])

                fyuseId = dataBatch['fyuseId']
                imgInput = dataBatch['ImgInput'].cuda(opt.gpuId)
                imgInputMsk = dataBatch['ImgInputMsk'].cuda(opt.gpuId)
                imgViews = dataBatch['ImgViews'].reshape(currBatchSize*opt.numViews,opt.imageSize,opt.imageSize).cuda(opt.gpuId)
                projViews = dataBatch['ProjViews'].reshape(currBatchSize*opt.numViews,3,4).cuda(opt.gpuId)
                distViews = dataBatch['DistViews'].reshape(currBatchSize*opt.numViews,5)
                colImgViews = dataBatch['ColImgViews'].reshape(currBatchSize*opt.numViews,3,opt.imageSize,opt.imageSize)
                imgInputK = dataBatch['ImgInputK'].cuda()
                imgInputRt = dataBatch['ImgInputRt'].cuda()

                #imgMaskedInput = torch.cat([imgInput,imgInputMsk], dim=1)
                out = modelPix2Mesh(imgInput, imgInputK) # should take in camera intrinsics!!!!                
                edgeLoss, lapLoss, moveLoss = criterionP2M(out)

                outCoordinates = out["pred_coord"]
                # write a submodule to re-orient the mesh vertices to the rest state!!!!
                for ijk in range(3):
                    outCoordinates[ijk] = torch.cat([outCoordinates[ijk], torch.ones_like(outCoordinates[ijk][:, :, None, 0])], dim=-1)
                    outCoordinates[ijk] = torch.bmm(outCoordinates[ijk], torch.inverse(imgInputRt).transpose(2,1)[:,:,0:3])

                renderer = sr.SoftRenderer(image_size=opt.imageSize, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='projection', P=projViews, orig_size=opt.origImageSize)
                SS = 0.
                for ijk in range(3) : 
                    meshOut = sr.Mesh(outCoordinates[ijk].repeat(1, opt.numViews, 1).reshape(opt.numViews*currBatchSize, -1, 3), ellipsoid.faces[ijk].repeat(opt.numViews*currBatchSize, 1, 1).type(torch.IntTensor).cuda(opt.gpuId))
                    imagesPred = renderer.render_mesh(meshOut)
                    SS += losses.SilhouetteLoss(imagesPred[:, 3,:,:], imgViews)
                
                
                loss = lamS*SS + \
                       lamL*lapLoss + \
                       lamM*moveLoss +\
                       lamE*edgeLoss
                
                # Train net..
                opModelPix2Mesh.zero_grad()

                if phase == 'train':                
                    loss.backward()
                    DebugHelper.PlotGradFlow(modelPix2Mesh.named_parameters(), os.path.join(opt.experiment,'tmp'), epoch, 'Encoder')
                    opModelPix2Mesh.step()

                if jj % 10 == 0 and phase == 'train':
                    images = imagesPred.detach().cpu().numpy()
                    imagesGt = imgViews.detach().cpu().numpy() 
                    colImagesGt = colImgViews.detach().cpu().numpy()

                    numFrames = 20 # Save only 20 frames..
                    globalImg = 255 * np.ones((opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.uint8)
                    globalImgGt = np.zeros((opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.uint8)
                    globalColViews = np.zeros((3,opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.float32)
                    globalColViewsGt = np.zeros((3,opt.imageSize*int(numFrames/5 + 1),opt.imageSize*5), dtype=np.float32)
                    for i in range(numFrames) : 
                        col = int(i % 5)
                        row = int(i / 5)
                        image = images[i].transpose((1,2,0))
                        globalImg[row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = (255 - 255*image[...,-1]).astype(np.uint8)
                        globalImgGt[row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = (127.5*imagesGt[i]).astype(np.uint8)
                        globalColViewsGt[:,row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = colImagesGt[i]
                        globalColViews[:,row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = images[i,0:3,:,:]

                    imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_deform_%05d.png'%ii), globalImg+globalImgGt)
                    imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_groundT_%05d.png'%ii), globalImgGt)
                    imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_groundTCol_%05d.jpg'%ii), globalColViewsGt.astype(np.uint8).transpose(1,2,0))
                    imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_DeformCol_%05d.jpg'%ii), (255*globalColViews).astype(np.uint8).transpose(1,2,0))
                    # save to tensorboard!!
                    writer.add_image("Deformed and Ground Truth", globalImg+globalImgGt, global_step=jj, dataformats='HW')
                writer.add_scalar(tag=phase, scalar_value=loss.item(), global_step=jj)

                if phase == 'val' and (jj % 10 == 0): 
                    # Running val in batchsize 1..
                    sr.Mesh(outCoordinates[ijk][0], ellipsoid.faces[ijk].type(torch.IntTensor).cuda(opt.gpuId)).save_obj(os.path.join(opt.experiment, fyuseId[0]+'_val_car.obj'), save_texture=False)
                elif phase == 'train' and (jj % 10 == 0):
                    sr.Mesh(outCoordinates[ijk][0], ellipsoid.faces[ijk].type(torch.IntTensor).cuda(opt.gpuId)).save_obj(os.path.join(opt.experiment, fyuseId[0]+'_train_car.obj'), save_texture=False)
                runningLoss += loss
                loop.set_description('Loss: %.4f'%(loss.item()))
                #print (runningLoss/(ii+1.))
                jj += 1

            epochLoss = runningLoss / dataLengths[phase]
            print('{} Loss: {:.4f}'.format(phase, epochLoss))

        if epoch % 5 == 0 : 
            # Save model ..
            state = {
                'epoch' : epoch,
                'stateDictPix2Mesh' : modelPix2Mesh.state_dict(),
                'optimizerPix2Mesh' : opModelPix2Mesh.state_dict(),
            }
            torch.save(state, os.path.join(opt.experiment, 'Model%d.pth'%(epoch%6)))

        print ('===============================\n\n')
