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
import torch.nn as nn
import numpy as np
import torch.optim as optim
import soft_renderer as sr

from torch.autograd import Variable

#from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='/media/intelssd/akar/mesh_seg_dataset/', help='path to Dataset Root')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
parser.add_argument('--fyuses', default='fyuse_ids.txt', help='the path to fyuseIds')
parser.add_argument('--scale', type=float, default=1.0, help='learning rate scaling')
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
parser.add_argument('--lamC', type=float, default=1.0, help='weight Color')
parser.add_argument('--lamL', type=float, default=1.0, help='weight Laplacian')
parser.add_argument('--lamP', type=float, default=1.0, help='weight Pixel loss')
parser.add_argument('--lamF', type=float, default=0.003, help='weight Flatten loss')

opt = parser.parse_args()
print(opt)
#torch.backends.cudnn.enabled = False

opt.gpuId = opt.deviceIds[0]

if opt.experiment is None:
    opt.experiment = 'CheckMeshGen'
os.system('mkdir {0}'.format(opt.experiment))
#Clean Directory
os.system('rm {0}/*'.format(opt.experiment))
os.system('cp *.py %s' % opt.experiment )

lamS = opt.lamS
lamC = opt.lamC
lamL = opt.lamL
lamP = opt.lamP
lamF = opt.lamF

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
encoderInit = nn.DataParallel(models.Encoder(), device_ids=opt.deviceIds)
decoderInit = nn.DataParallel(models.Decoder(numVertices=642+1), device_ids=opt.deviceIds) # Center to be predicted too


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
scale = opt.scale
opEncoderInit = optim.Adam(encoderInit.parameters(), lr=1e-2 * scale, betas=(0.5, 0.999) )
opDecoderInit = optim.Adam(decoderInit.parameters(), lr=1e-2 * scale, betas=(0.5, 0.999) )
#####################################


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


######################################
jj = 0

for epoch in range(opt.nepoch):
    print('Epoch {}/{}'.format(epoch, opt.nepoch - 1))
    print ('===============================')

    # Each epoch has a training and validation phase
    for phase in ['train', 'val'] : #['val', 'train'] ['train', 'val']
        if phase == 'train':
            encoderInit.train(True)  # Set model to training mode
            decoderInit.train(True)
        else:
            encoderInit.train(False)  # Set model to evaluate mode
            decoderInit.train(False)

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
            imgViews = dataBatch['ImgViews'].cuda(opt.gpuId)
            projViews = dataBatch['ProjViews'].reshape(currBatchSize*opt.numViews,3,4).cuda(opt.gpuId)
            distViews = dataBatch['DistViews'].reshape(currBatchSize*opt.numViews,5).cuda(opt.gpuId)
            templateVertex = dataBatch['TemplVertex'].cuda(opt.gpuId)
            templateFaces =  dataBatch['TemplFaces'].cuda(opt.gpuId)

            imgMaskedInput = torch.cat([imgInput,imgInputMsk], dim=1)
            features = encoderInit(imgMaskedInput)
            outPos = decoderInit(features)
            #print (outPos.shape)
            meshM = models.MeshModel(templateFaces, templateVertex).cuda(opt.gpuId)
            # TODO : calculate lap and flat loss here..
            meshDeformed, lapLoss, fltLoss = meshM.forward(outPos[:,:-1,:], torch.zeros_like(outPos[:,-1:,:]).cuda(), opt.numViews, currBatchSize)
            renderer = sr.SoftRenderer(image_size=opt.imageSize, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='projection', P=projViews, orig_size=opt.origImageSize)
            imagesPred = renderer.render_mesh(meshDeformed)
            SS = losses.SilhouetteLoss(imagesPred[:, 3], imgViews.reshape(currBatchSize*opt.numViews,opt.imageSize,opt.imageSize))
            loss = lamS*SS + \
                   lamL*lapLoss + \
                   lamF*fltLoss
            
            # Train net..
            opEncoderInit.zero_grad()
            opDecoderInit.zero_grad()


            if jj % 10 == 0 and phase == 'train':
                images = imagesPred.detach().cpu().numpy()
                imagesGt = imgViews.detach().cpu().numpy().reshape(opt.batchSize*opt.numViews,opt.imageSize,opt.imageSize)
                numFrames = 20 # Save only 20 frames..
                globalImg = 255 * np.ones((opt.imageSize*int(numFrames/10 + 1),opt.imageSize*10), dtype=np.uint8)
                globalImgGt = np.zeros((opt.imageSize*int(numFrames/10 + 1),opt.imageSize*10), dtype=np.uint8)
                for i in range(numFrames) : 
                    col = int(i % 10)
                    row = int(i / 10)
                    image = images[i].transpose((1,2,0))
                    globalImg[row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = (255 - 255*image[...,-1]).astype(np.uint8)
                    globalImgGt[row*opt.imageSize:row*opt.imageSize + opt.imageSize,col*opt.imageSize:col*opt.imageSize + opt.imageSize] = (127.5*imagesGt[i]).astype(np.uint8)

                
                imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_deform_%05d.png'%ii), globalImg+globalImgGt)

                # save optimized mesh
                imageio.imsave(os.path.join(opt.experiment, fyuseId[0]+'_groundT_%05d.png'%ii), globalImgGt)


            if phase == 'train':                
                loss.backward()
                opDecoderInit.step()
                opEncoderInit.step()
            if phase == 'val' and jj % 10: 
                # Running val in batchsize 1..
                meshM.forward(outPos[:,:-1,:], outPos[:,-1:,:], 1, 1)[0].save_obj(os.path.join(opt.experiment, fyuseId[0]+'_car.obj'), save_texture=False)
                
            runningLoss += loss
            loop.set_description('Loss: %.4f'%(loss.item()))
            #print (runningLoss/(ii+1.))
            jj += 1

        epochLoss = runningLoss / dataLengths[phase]
        print('{} Loss: {:.4f}'.format(phase, epochLoss))
    print ('===============================\n\n')
