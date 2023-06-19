from enum import EnumMeta
import os 
import glob
from pickle import TRUE
import numpy as np
import cv2
from torch.utils.data import Dataset,DataLoader
import scipy.io as sio
import random
import torch
import torch.nn.functional as F


def list_file_path(root_path,fortrain=True,shineness=[0]):
    modeldirs = []
    testmodeldirs = []
    mea = []
    im = []
    de =[]
    test_mea = []
    test_im = []
    test_de =[]
    root_path = [root_path]
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))      # file_path/0/XXX
            
            for modeldir in modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))[:1]#[:11]   # file_path/0/XXX/shinexxx_rot xxx
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocal*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    de.append(path)
            for modeldir in modeldir_all[250:]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))#[:1]   # file_path/0/XXX/shinexxx_rot xxx
                testmodeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocal*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    test_de.append(path)
    
    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': testmodeldirs}
    # import ipdb
    # ipdb.set_trace()
    if fortrain:
        return train_sample
    else:
        return test_sample

def list_file_path_bike(root_path,fortrain=TRUE,shineness=[0]):
    modeldirs = []
    test_modeldirs = []
    mea = []
    im = []
    de =[]
    test_mea = []
    test_im = []
    test_de =[]
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))      # file_path/0/XXX
            for modeldir in modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))   # file_path/0/XXX/shinexxx_rot xxx
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    de.append(path)
                    
            for modeldir in modeldir_all[250:]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))#[:1]   # file_path/0/XXX/shinexxx_rot xxx
                test_modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    test_de.append(path)
                    
    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': test_modeldirs}
    # import ipdb
    # ipdb.set_trace()
    if fortrain:
        return train_sample
    else:
        return test_sample
    
    
def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)


class NLOSPoissonNoise:

    def __init__(self, background=[0.05, 0.5]):
        self.rate = background

    def __call__(self, x):
        if isinstance(self.rate, (int, float)):
            rate = self.rate
        elif isinstance(self.rate, (list, tuple)):
            rate = random.random() * (self.rate[1] - self.rate[0]) 
            rate += self.rate[0]
        poisson = torch.distributions.Poisson(rate)
        # shot noise + background noise
        x = torch.poisson(x) + poisson.sample(x.shape).cuda()
        return x

    def __repr__(self):
        return 'Introduce shot noise and background noise to raw histograms'


class NLOSRandomScale:

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, x):
        if isinstance(self.scale, (int, float)):
            x *= self.scale
        elif isinstance(self.scale, (list, tuple)):
            scale = random.random() * (self.scale[1] - self.scale[0]) 
            scale += self.scale[0]
            x *= scale
        return x

    def __repr__(self):
        return 'Randomly scale raw histograms'

class NLOSCompose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        repr_str = ''
        for t in self.transforms:
            repr_str += t.__repr__() + '\n'
        return repr_str


def get_transform(scale=1, background=0):
    transform = [NLOSRandomScale(scale)]
    if background != 0:
        transform += [NLOSPoissonNoise(background)]
    transform = NLOSCompose(transform)
    return transform


class LFEDataset(Dataset):

    def __init__(
        self, 
        root,               # dataset root directory
        shineness,              # data split ('train', 'val')
        for_train=True,
        ds=1,               # temporal down-sampling factor
        clip=512,           # time range of histograms
        size=256,           # measurement size (unit: px)
        scale=1,            # scaling factor (float or float tuple)
        background=0,       # background noise rate (float or float tuple)
        target_size=256,    # target image size (unit: px)
        target_noise=0,     # standard deviation of target image noise
        color='gray',        # color channel(s) of target image
    ):
        super(LFEDataset, self).__init__()
        self.root = root
        self.ds = ds
        self.clip = clip
        self.size = size
        self.transform = get_transform(scale, background)
        self.target_size = target_size
        self.target_noise = target_noise
        assert color in ('rgb', 'gray', 'r', 'g', 'b'), \
            'invalid color: {:s}'.format(color)
        self.data_list = list_file_path_bike(root,for_train,shineness)
        self.color = color

        
    def _load_meas(self, idx):
        path = self.data_list['Mea'][idx][0]
        #check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr','mp4')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)        # imgray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) imgray = imgray.reshape(600, 256, 256);
                #x = x.reshape(-1, x.shape[1], x.shape[1], 3)     # 600 256 256 3 
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.reshape(-1, x.shape[1], x.shape[1], 1)  
                x = x.transpose(3, 0, 1, 2)  # 1 600 256 256 
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened() 
                ims = []
                # Read until video is completed
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
                        ims.append(imgray)
                    else:
                        break
                # When everything done, release the video capture object
                cap.release()
                data = np.array(ims, dtype=np.float32) / 255.0       
                x = data[None]    # 1 600 256  256 
                x = x[:,:,::2,:] + x[:,:,1::2,:]
                x = x[:,:,:,::2] + x[:,:,:,1::2]
                
            x = x[:, :self.clip]                                # (1/3, t, h, w)
            # temporal down-sampling
            if self.ds > 1:
                c, t, h, w = x.shape
                assert t % self.ds == 0
                x = x.reshape(c, t // self.ds, self.ds, h, w)
                x = x.sum(axis=2)
            # clip temporal range
        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))
        
        x = torch.from_numpy(x.astype(np.float32))              # (1/3, t, h, w)
        # spatial sub-sampling
        # assert h == w
        # if h != self.size:
        #     x = F.interpolate(x, size=(self.size,) * 2, mode='nearest')
        return x

    def _load_image(self, idx):
        path = self.data_list['img'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.astype(np.float32)
                if ext == 'hdr':
                    x = x.reshape( x.shape[0], x.shape[1], 1)   #  256 256 1
                    # print(x.max(),x.min(),x.mean())    
                    x = cv2.resize(x,(self.target_size,self.target_size))
                    x = x / np.max(x)  # 原来是都是这个
                    # x = x.clip(0,1)
                    x = x.reshape( x.shape[0], x.shape[1], 1)   #  128 128 1
                else:
                    x = x / 255
        except:
            raise ValueError('image loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))             
        x = x.permute(2, 0, 1)                               # ( 1/3, h, w)       #  1 256 256 
        if self.target_noise > 0:
            x += torch.randn_like(x) * self.target_noise
            x = torch.clamp(x, min=0)
        return x

    def _load_depth(self, idx):
        path = self.data_list['dep'][idx][0]
        #check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)        #   meters  fe 0.597m
                x = x[..., 0]
                x = x.clip(0,1)
            x = cv2.resize(x, (self.target_size,self.target_size))          # (h, w, v)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))

        x = x.reshape(-1, x.shape[0], x.shape[1])
        x = torch.from_numpy(x.astype(np.float32))              # (1, h, w)
        return x
    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas = self._load_meas(idx) #self.transform()  #   the hyperparameter [0.05 2] has err
        images = self._load_image(idx)
        depths = self._load_depth(idx)
        sample = {'M_nos': meas, 'dep_gt': depths, 'img_gt': images}
        return sample



class NLOSDataset(Dataset):
    def __init__(
        self, 
        root,               # dataset root directory
        split=True,              # data split ('train', 'val')
        ds=1,               # temporal down-sampling factor
        clip=512,           # time range of histograms
        size=256,           # measurement size (unit: px)
        d_s=1,            # scaling factor (float or float tuple)
        background=0,       # background noise rate (float or float tuple)
        target_size=256,    # target image size (unit: px)
        target_noise=0,     # standard deviation of target image noise
        color='gray',        # color channel(s) of target image
    ):
        super(NLOSDataset, self).__init__()

        self.root = root
        self.ds = ds
        self.clip = clip
        self.size = size
        self.d_spatial = d_s
        # self.transform = get_transform(scale, background)
        self.target_size = target_size
        self.target_noise = target_noise

        assert color in ('rgb', 'gray', 'r', 'g', 'b'), \
            'invalid color: {:s}'.format(color)
        self.color = color
        self.split = split
        self.data_list = list_file_path(self.root,self.split)
        print(self.split,len(self.data_list['Mea']))


    def _load_meas(self, idx):
        path = self.data_list['Mea'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'mp4')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x = x.reshape(-1, x.shape[1], x.shape[1], 3)
                x = x.transpose(3, 0, 1, 2)
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened() 
                ims = []
                # Read until video is completed
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ims.append(frame)
                    else:
                        break
                # When everything done, release the video capture object
                cap.release()
                x = np.array(ims, dtype=np.float32) / 255.0  
                # print(x.shape)
                x = x.transpose(3,0,1,2)   # 3 600 256 256     
            if x.ndim == 3:
                x = x[None]
                
            # spatial down-sampling
            if self.d_spatial >1 :
                x = x[:,:,::2,:] + x[:,:,1::2,:]
                x = x[:,:,:,::2] + x[:,:,:,1::2]
                
            # temporal down-sampling
            if self.ds > 1:
                c, t, h, w = x.shape
                assert t % self.ds == 0
                x = x.reshape(c, t // self.ds, self.ds, h, w)
                x = x.sum(axis=2)

            # clip temporal range
            x = x[:, :self.clip]                                # (1/3, t, h, w)
            
            if x.shape[0] == 3:
                if self.color == 'gray':
                    x = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
                elif self.color == 'r': x = x[0:1]
                elif self.color == 'g': x = x[1:2]
                elif self.color == 'b': x = x[2:3]
            else:
                if self.color == 'rgb':
                    x = np.tile(x, [3, 1, 1, 1])
        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))
        
        x = torch.from_numpy(x.astype(np.float32))              # (1/3, t, h, w)
        # spatial sub-sampling
        # t, h, w = x.shape[1:]
        # assert h == w
        # if h != self.size:
        #     x = F.interpolate(x, size=(self.size,) * 2, mode='nearest')
        return x

    def _load_image(self, idx):
        path = self.data_list['img'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['img']     # 26 256 256 3
                x = x.transpose(1,2,0,3).reshape(x.shape[1],x.shape[2],-1)
                x = cv2.resize(x, (self.target_size, self.target_size))          # (h, w, v*3)
                # x = x / np.max(x,(0,1))                                          ## test on syn dataset need this
                x = x.clip(0,1)                                        
                x = x.reshape(x.shape[0], x.shape[1], 26, 3)    # (h, w, v, 3)
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                if ext == 'hdr':
                    x = x.reshape(-1, x.shape[1], x.shape[1], 3)
                    x = x.transpose(1, 2, 0, 3)
                    x = x.reshape(*x.shape[:2], -1)
                else:
                    x = x / 255

            if self.color == 'gray':
                x = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
            elif self.color == 'r': x = x[..., 0:1]
            elif self.color == 'g': x = x[..., 1:2]
            elif self.color == 'b': x = x[..., 2:3]
        except:
            raise ValueError('image loading failed: {:s}'.format(path))

        # ## return one view
        # x = x.astype(np.float32)[:,:,0,:]      # (h, w, v, 3)
        # x = torch.from_numpy(x)              # (h, w, 1/3)
        # x = x.permute(2, 0, 1)                               # ( 1/3, h, w)

        ### return all view
        x = torch.from_numpy(x.astype(np.float32))              # (h, w, v, 3)
        x = x.permute(2, 3, 0, 1)                               # (v, 1/3, h, w)

        if self.target_noise > 0:
            x += torch.randn_like(x) * self.target_noise
            x = torch.clamp(x, min=0)
        return x#[0]

    def _load_depth(self, idx):
        path = self.data_list['dep'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['depth']   # 26 256 256 3
                x = x.transpose(1,2,0,3).reshape(x.shape[1],x.shape[2],-1)
                x = cv2.resize(x, (self.target_size, self.target_size))          # (h, w, v*3)
                x = x.clip(0,1)
                x = x.reshape(x.shape[0], x.shape[1], 26, 3)    # (h, w, v, 3)
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = x[..., 0]
                x = x.reshape(-1, x.shape[1], x.shape[1])
                x = x.transpose(1, 2, 0)
            # x = cv2.resize(x, (self.target_size,) * 2)          # (h, w, v)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))              # (h, w, v, 3)
        x = x.permute(2, 3, 0, 1)                               # (v, 1, h, w)
        x = torch.mean(x,dim=1,keepdim=True)
        return x#[0]
    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas = self._load_meas(idx) #self.transform(self._load_meas(idx))
        images = self._load_image(idx)
        depths = self._load_depth(idx)
        sample = {'M_nos': meas, 'dep_gt': depths, 'img_gt': images}
        return sample



if __name__ == '__main__': 
    root_path = ['/data2/yueli/dataset/LFE_dataset/bike']
    shineness = [0]
    #list_file_path(root_path,shineness)
    train_set = LFEDataset(root_path,shineness,True,1,512,256,1,[0.05,2],256,0,'gray')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16) # drop_last would also influence the performance
    
    # folder_path = ['/data2/yueli/dataset/LFE_dataset/NLOS_EF_allviews_scale_0.75processed']
    # train_data = NLOSDataset(folder_path,True,1,512,256,1,[0.05,0.03],256,0.000,'gray')
    # train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=16) # drop_last would also influence the performance
    
    for index,data in enumerate(train_loader):
        print(index)
        
        mea = data['Mea']
    