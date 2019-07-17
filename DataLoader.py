import glob
import random
import json
import os
import numpy as np

from torch.utils.data import Dataset


class BatchLoader(Dataset):
    def __init__(self, dataRoot, imSize = 256, isRandom=True, pad=420, rseed = None):
        self.dataRoot = dataRoot
        self.imSize = imSize

        f = open(os.path.join(dataRoot,'fyuse_ids.txt','r'))
        dataFyuseList = [ids.strip() for ids in f]
        f.close()
        random.shuffle(dataFyuseList) # Permute..

        # Now train eval split...

        self.dataViewMaskNames = {}
        self.dataProjectionMat = {}
        for idx in dataFyuseList : 
            self.dataViewMaskNames[idx] = glob.glob(os.path.join(dataroot, 'Normalized/Depths/',idx,'/depth*.png'))
            self.dataProjectionMat[idx] = ParsePoses(dataRoot, idx, self.dataViewMaskNames[idx], pad)


        # Load Mesh here??

            


    def __len__(self):
        return len(self.dataFyuseList)


    def __getitem__(self, ind):
        


        batchDict = {'albedo': albedo,
                     'normal': normal,
                     'rough': rough,
                     'depth': depth,
                     'seg': seg,
                     'imP': imP,
                     'imE':  imE,
                     'imEbg': imEbg,
                     'SH': SH,
                     'name': name,
                     'albedoName': self.albedoList[self.perm[ind] ],
                     'realImage': imReal,
                     'realImageMask': segReal}



        return batchDict


    def loadImage(self, imName, isGama = False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            return im

        im = Image.open(imName)
        im = self.imResize(im)
        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        return im

    def imResize(self, im):
        w0, h0 = im.size
        assert( (w0 == h0) )
        im = im.resize((self.imSize, self.imSize), Image.ANTIALIAS)
        return im

    def loadNpy(self, name):
        data = np.load(name)
        return data

    def ParsePoses(self, dataRoot, fyuseId, views, pad=420):
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
            # k1, k2, p1, p2, k3
            transforms['distortion'] = np.array(pose['anchor']['distortion'] + [0.])

            intrinsics = np.array(pose['anchor']['intrinsicsVector'])
            K = [
                [intrinsics[0], intrinsics[2], intrinsics[3]-0.5], [0, intrinsics[1], intrinsics[4] + pad - 0.5], [0, 0, 1]
            ]

            K = np.array(K)
            Rt = np.linalg.inv(np.array(pose['anchor']['transform']).reshape(4, 4))
            transforms['P'] = np.matmul(K, Rt[0:3,:])
            allPoses[frame_num] = transforms
        return allPoses

