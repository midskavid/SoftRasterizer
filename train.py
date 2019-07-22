import torch
import argparse
import random
import os
import models
import losses
#import utils
import DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils

from torch.autograd import Variable
#from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='/media/intelssd/akar/mesh_seg_dataset/', help='path to Dataset Root')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=100, help='the number of epochs for training')
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
parser.add_argument('--lamC', type=float, default=1.0, help='weight Color')
parser.add_argument('--lamL', type=float, default=1.0, help='weight Laplacian')
parser.add_argument('--lamP', type=float, default=1.0, help='weight Pixel loss')


opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.experiment is None:
    opt.experiment = 'CheckMeshGen'
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

lamS = opt.lamS
lamC = opt.lamC
lamL = opt.lamL
lamP = opt.lamP

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#################################
# initialize tensors
imInputBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize))
imInputMaskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize))
# need a variable size placeholder to handle variable number of views per fyuse...


# initialize models
encoderInit = nn.DataParallel(models.Encoder(), device_ids = opt.deviceIds)
decoderInit = nn.DataParallel(models.Decoder(numVertices=642+1), device_ids = opt.deviceIds) # Center to be predicted too


##############  ######################
# Send things into GPU
if opt.cuda:
    imInputBatch = imInputBatch.cuda(opt.gpuId)
    imInputMaskBatch = imInputMaskBatch.cuda(opt.gpuId)

    encoderInit = encoderInit.cuda(opt.gpuId)
    decoderInit = decoderInit.cuda(opt.gpuId)
####################################


####################################
# Initial Optimizer
scale = 1.0
opEncoderInit = optim.Adam(encoderInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )
opDecoderInit = optim.Adam(decoderInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )
#####################################


####################################
# Data Loaders..
fyuseDataset = DataLoader.BatchLoader(opt.dataRoot, imSize=opt.imageSize, numViews=opt.numViews, padding=opt.pad)
datasetSize = len(fyuseDataset)
indices = list(range(datasetSize))
np.random.shuffle(indices)
split = int(np.floor(opt.validationSplit * datasetSize))

trainIndices, valIndices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
trainSampler = torch.utils.data.SubsetRandomSampler(trainIndices)
validSampler = torch.utils.data.SubsetRandomSampler(valIndices)

# TODO : check the significance of shuffle here...
trainLoader = torch.utils.data.DataLoader(fyuseDataset, batch_size=opt.batchSize, sampler=trainSampler, num_workers = 8, shuffle = False)
validationLoader = torch.utils.data.DataLoader(fyuseDataset, batch_size=opt.batchSize, sampler=validSampler, num_workers = 8, shuffle = False)
dataLoaders = {"train": trainLoader, "val": validationLoader}
dataLengths = {"train": len(trainLoader), "val": len(validationLoader)}
######################################


######################################


for epoch in range(opt.nepoch):
    print('Epoch {}/{}'.format(epoch, opt.nepoch - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            encoderInit.train(True)  # Set model to training mode
            decoderInit.train(True)
        else:
            encoderInit.train(False)  # Set model to evaluate mode
            decoderInit.train(False)

        running_loss = 0.0

        # Iterate over data.
        for ii, dataBatch in enumerate(dataLoaders[phase]):
            # Dataloader would return me Projection matrices and the input images and ground truth images. 
            # The manner in which dataloader creates batch, I will have to reshape them to have batch = batch*numViews
            # For mesh vertices and faces, I will get the correct format batchxnumVertx3.
            # This will be changed to the appropriate format with the forward of the dataloader.
            imgInput = dataBatch['ImgInput'].cuda()
            imgInputMsk = dataBatch['ImgInputMsk'].cuda()
            imgViews = dataBatch['ImgViews'].cuda()
            projViews = dataBatch['ProjViews'].cuda()
            distViews = dataBatch['DistViews'].cuda()
            templateVertex = dataBatch['TemplVertex'].cuda()
            templateFaces =  dataBatch['TemplFaces'].cuda()

            imgMaskedInput = torch.cat([imgInput,imgInputMsk], dim=1)
            features = encoderInit(imgMaskedInput)
            outPos = decoderInit(features)
            #print (outPos.shape)
            meshM = MeshModel(templateFaces, templateVertex)
            # TODO : calculate lap and flat loss here..
            meshDeformed = meshM.forward(outPos[:,:-1,:], outPos[:,-1,:], opt.numViews)
            renderer = sr.SoftRenderer(image_size=opt.imageSize, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='projection', P=projViews, dist_coeffs=distViews, orig_size=opt.origImageSize)
            imagesPred = renderer.render_mesh(meshDeformed)
            
            




