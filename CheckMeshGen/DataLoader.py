import glob
import random
import json
import os
import numpy as np
import imageio

from PIL import Image
from torch.utils.data import Dataset
from skimage.util import img_as_float

class BatchLoader(Dataset):
    def __init__(self, dataRoot, batchSize, imSize = 256, isRandom=True, padding=420, numViews=20, debugDir='CheckMeshGen', rseed = None):
        self.dataRoot = dataRoot
        self.imSize = imSize
        self.numViews = numViews
        self.batchSize = batchSize 
        self.numViews = numViews
        f = open(os.path.join(self.dataRoot,'fyuse_ids.txt'),'r')
        self.dataFyuseList = [ids.strip() for ids in f]
        f.close()
        random.shuffle(self.dataFyuseList) # Permute..

        # Now train eval split...

        self.dataViewMaskNames = {}
        self.dataProjectionMat = {}
        for idx in self.dataFyuseList :
            self.dataViewMaskNames[idx] = glob.glob(os.path.join(self.dataRoot, 'Normalized/Depths/',idx,'depth*.png'))
            self.dataProjectionMat[idx] = self.ParsePoses(self.dataRoot, idx, self.dataViewMaskNames[idx], padding)


        # Load template mesh here..
        self.templateVertex, self.templateFaces = self.LoadObjMesh(os.path.join(self.dataRoot,'template_mesh.obj'))
        # print (self.dataFyuseList)
        # print (self.dataViewMaskNames)
        # print (self.dataProjectionMat)
        self.SaveDebugBatch(debugDir)    


    def __len__(self):
        return len(self.dataFyuseList)


    def __getitem__(self, ind):

        # Load Image.. pick one randomly..
        fyuseId = self.dataFyuseList[ind]
        try :
            indexes = random.sample(range(len(self.dataViewMaskNames[fyuseId])), self.numViews + 1)
        except : 
            print (fyuseId)
            raise
        # Load input image from JPEG images.. pad image accordingly : 
        frame = self.dataViewMaskNames[fyuseId][indexes[0]].split('/')[-1].replace('depth','').replace('.png','.jpg')
        imgInput = self.LoadImage(os.path.join(self.dataRoot, 'JPEGImages', fyuseId+'_'+frame))
        # read the corresponding mask...
        imgInputMsk = self.LoadMaskFromDepth(self.dataViewMaskNames[fyuseId][indexes[0]])

        # Now view info.. image and projection matrix : 
        imgViews = []
        projViews = []
        distViews = []
        lPoseData = list(self.dataProjectionMat[fyuseId])
        for ii in range(self.numViews): 
            #frame = int(self.dataViewMaskNames[fyuseId][indexes[0]].split('/')[-1].replace('depth','').replace('.png',''))
            imgViews.append(self.LoadMaskFromDepth(self.dataViewMaskNames[fyuseId][indexes[ii+1]]))
            # print (fyuseId)
            # print (lPoseData[indexes[ii]])
            # print (self.dataProjectionMat[fyuseId], indexes[ii+1])
            projViews.append(self.dataProjectionMat[fyuseId][lPoseData[indexes[ii+1]]]['P'])
            distViews.append(self.dataProjectionMat[fyuseId][lPoseData[indexes[ii+1]]]['distortion'])

        imgViews = np.stack(imgViews)
        projViews = np.stack(projViews)
        distViews = np.stack(distViews)

        batchDict = {'ImgInput': imgInput,
                     'ImgInputMsk': imgInputMsk,
                     'ImgViews': imgViews,
                     'ProjViews': projViews,
                     'DistViews': distViews,
                     'TemplVertex': self.templateVertex,
                     'TemplFaces': self.templateFaces
                    }

        return batchDict


    def LoadImage(self, imName, isGama = False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            return im

        im = Image.open(imName)
        imSq = Image.new('RGB', (960,960), (0,0,0)) ####Hardcoding
        imSq.paste(im, (0,210)) ####Hardcoding
        imSq = self.ImResize(imSq)
        im = np.asarray(imSq, dtype=np.float32)

        #### Image being fed to the network has to be normalized but the different views should not be normalized...
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        return im

    def LoadMaskFromDepth(self, imName) :
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            return im

        # im = Image.open(imName)
        # imSq = Image.new('RGB', (1920,1920), (0,0,0)) ####Hardcoding
        # imSq.paste(im, (0,420)) ####Hardcoding
        # imSq = self.ImResize(imSq)
        # im = np.asarray(imSq)
        # im = img_as_float(im)
        # im2 = np.zeros((self.imSize,self.imSize))
        # im2[np.where(im < 1.)] = 0.

        # PIL does not have a 16bit read.. therefore all this conversion between numpy and PIL
        # TODO : easier way to use OpenCV
        try : 
            Im = imageio.imread(imName)
            Im2 = np.ones((1080,1920))
            Im1 = img_as_float(Im)
            Im2[np.where(Im1 < 1.)] = 0.
            Im2 = (255-255*Im2).astype(np.uint8)
            im = Image.fromarray(Im2, 'L')
            imSq = Image.new('L', (1920,1920), (0)) ####Hardcoding
            imSq.paste(im, (0,420)) ####Hardcoding
            imSq = imSq.resize((self.imSize, self.imSize), Image.ANTIALIAS)
            im = np.asarray(imSq).astype('float32')/255.0
            im = im[np.newaxis]        
        except :
            print (imName)
            raise
        return im


    def ImResize(self, im):
        w0, h0 = im.size
        assert( (w0 == h0) )
        im = im.resize((self.imSize, self.imSize), Image.ANTIALIAS)
        return im

    def ParsePoses(self, dataRoot, fyuseId, views, pad=420):
        # This is for the new format!!!
        scenemodel_path = os.path.join(dataRoot, 'Normalized/ScenemodelFiles', fyuseId+'_scenemodel_raw.json')
        assert os.path.exists(scenemodel_path), "Scenemodel {} not found!".format(scenemodel_path)
        with open(scenemodel_path, 'r') as f:
            poses = json.load(f)
        allPoses = {}
        viewIds = set([int(view.split('/')[-1].replace('depth','').split('.')[0]) for view in views])
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
                    [intrinsics[0], intrinsics[2], intrinsics[3]-0.5], [0, intrinsics[1], intrinsics[4] + pad - 0.5], [0, 0, 1]
                ]

                K = np.array(K)
                Rt = np.linalg.inv(np.array(pose['anchor']['transform']).reshape(4, 4))
                transforms['P'] = np.matmul(K, Rt[0:3,:]).astype('float32')

            allPoses[frameNum] = transforms
        return allPoses

    def LoadObjMesh(self, filename):
        vertices = []
        faces = []
        f = open(filename,'r')
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
        vertices = np.vstack(vertices).astype(np.float32)
        faces = np.vstack(faces).astype(np.int32) - 1 ##### ASSUMING START FROM 1
        f.close()
        
        return vertices, faces

    def SaveDebugBatch(self, dataDir):
        # TODO : implement this
        dataBatch = self[2]

        # First save all images... #Then project the mesh and save dots.. That will give a pretty good idea..
        imgInput = dataBatch['ImgInput']
        imgInputMsk = dataBatch['ImgInputMsk']
        imgViews = dataBatch['ImgViews']
        projViews = dataBatch['ProjViews']
        distViews = dataBatch['DistViews']
        templateVertex = dataBatch['TemplVertex']
        templateFaces =  dataBatch['TemplFaces']

        fyuseId = self.dataFyuseList[2]
        numFrames = len(imgViews)
        imageio.imsave(os.path.join(dataDir,fyuseId+'inputImg.jpg'), (imgInput*127.5+127.5).transpose(1,2,0).astype('uint8'))
        imageio.imsave(os.path.join(dataDir,fyuseId+'inputImgMask.jpg'), (255*imgInputMsk[0,:,:]).astype('uint8'))
        globalImg = 255 * np.ones((self.imSize*int(numFrames/5 + 1),self.imSize*5), dtype=np.uint8)

        for ii in range(numFrames) : 
            col = int(ii % 5)
            row = int(ii / 5)
            print (imgViews[ii][0])
            globalImg[row*self.imSize:row*self.imSize + self.imSize,col*self.imSize:col*self.imSize + self.imSize] = (255*imgViews[ii][0]).astype(np.uint8)

        imageio.imsave(os.path.join(dataDir, fyuseId+'Views.png'), globalImg)        
        print ("DONE!")
        pass