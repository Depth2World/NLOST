from cv2 import transform
from math import sqrt, pow, log2
from operator import pos
import numpy as np
from numpy.core.shape_base import block
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.modules.activation import Hardswish
from torch.nn.modules.container import T
from torch.nn.modules.normalization import CrossMapLRN2d
from util import LFEDataset


from .nlost_modules import *
from .modules import *
from .utils_pytorch import phasor_1_10
from .utils_pytorch import fk_1_10
from .utils_pytorch import lct_1_10

    

class NLOST(nn.Module):
    r"""The TSTIntemodelNBlk
    The main network architecture.
    Input:
        ch_in (int): the channel size of input raw measuremnts, default: 1
    Output: 3D volume and 2D depth map, with resolution of (B,1,H,W)
    """
    def __init__(self, ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.01,target_size=256):
        super().__init__()
        self.sa = target_size // spatial
        assert self.sa in [1,2]
        self.in_chans = ch_in
        self.coders = num_coders
        self.spatial = spatial
        self.tlen = tlen
        self.bin_len = bin_len
        if self.sa==2:
            ts_stride = [2,2,2]    ## convlutional stride across the temporal and spatial dimensions
        else:
            ts_stride = [2,1,1]
        # feature extraction
        self.sig_expand = Transient_TDown_2(ch_in,2,ts_stride)
        self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len*2,dnum=self.in_chans*4)
        channels_m = 8
        self.msfeat = MsFeat_2(self.in_chans*4, channels_m) #B,32,128,H,W
        self.sig_feat = Transient_TDown_3(32,32)
        
        # downsample and fusion => patch embedding
        self.dsfusion = DsFusion(channels_m * 4, channels_m * 4)    #B,16,64,H,W
        
        # spatial downsample
        self.spds = SPDS(channels_m * 4,ds=2)
        self.posenc_l = PosiEncCNN(channels_m * 4)
        self.posenc_g = PosiEncCNN(channels_m * 4)
        # local encoders
        self.loc_encds = modelClone(WindowEncoderSep(dim=channels_m * 4,input_resolution=[self.spatial,self.spatial,16],num_heads=4,window_size=self.spatial//2), self.coders)
        # global encoders
        self.glb_encds = modelClone(GlobalEncoderSep(dim=channels_m * 4,input_resolution=[self.spatial//2,self.spatial//2,16],num_heads=4), self.coders)
        # local-global integration
        self.locglb_inte = LocGlbInteNBlks_LCGC_l1d2(channels_m * 4, [self.spatial,self.spatial,16],8, self.coders)
        # reconstruction blocks
        self.inte_rec = NLOSInteRec_2(self.in_chans * 4,channels_m)        #  64
        self.project = VisbleNet()

        self.inten_refine = Rendering(nf0=(3 * 1 + 1) * 2, out_channels=1,factor=self.sa)
        self.dep_refine = Rendering(nf0=(3 * 1 + 1) * 2, out_channels=1,factor=self.sa,isdep=True)


    def noise(self, data):
        gau = 0.05 + 0.03 * torch.randn_like(data) + data
        poi = 0.03 * torch.randn_like(data) * gau + gau
        return poi
    
    
    def normalize(self, data_bxcxdxhxw):   #  min max scaling
        b, c, d, h, w = data_bxcxdxhxw.shape
        data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
        data_min = data_bxcxk.min(2, keepdim=True)[0]
        data_zmean = data_bxcxk - data_min
        # most are 0
        data_max = data_zmean.max(2, keepdim=True)[0]
        data_norm = data_zmean / (data_max + 1e-15)
        return data_norm.view(b, c, d, h, w)


    def forward(self, inputs):
        # if self.training:
        #     inputs = self.noise(inputs)
        # else:
        #     inputs = inputs
        inputs = self.normalize(inputs)             # b 1 512 h w 
        sigal_expand = self.sig_expand(inputs)  # b 4 256 128 128 
        vlo = self.tra2vlo(sigal_expand,[0,0,0],[self.tlen,self.tlen,self.tlen])  # b 4 256 128 128   
        vlo = nn.ReLU()(vlo)
        vlo = self.normalize(vlo)    

        msfeat = self.msfeat(vlo)    
        sig_feat = self.sig_feat(msfeat) 
        patem_l = self.dsfusion(sig_feat) 
        patem_s = self.spds(patem_l)   

        # local information extraction 
        posenc_l = self.posenc_l(patem_l)    
        for loc_encd in self.loc_encds:
            winenc = loc_encd(posenc_l)   
            posenc_l = winenc 
        # global information extraction
        posenc_g = self.posenc_g(patem_s)    
        for glb_encd in self.glb_encds:
            glbenc = glb_encd(posenc_g)   
            posenc_g = glbenc
        # local global integration
        inte_loc, inte_glb = self.locglb_inte(winenc, glbenc)    
        # reconstruction
        out_vlo = self.inte_rec(inte_loc, inte_glb, msfeat)   
        
        # print(out_vlo.shape)
        # zdim = out_vlo.shape[2]
        # zdimnew = zdim * 100 // 128
        # out_vlo = out_vlo[:, :, :zdimnew]
        # out_vlo = nn.ReLU()(out_vlo)
        # out_vlo = self.normalize(out_vlo)
        
        raw = self.project(out_vlo)
        refine_intensity = self.inten_refine(raw) 
        refine_dep = self.dep_refine(raw) 

        refine_intensity = refine_intensity * 2 - 1 
        refine_dep = refine_dep * 2 - 1 

        return out_vlo, refine_intensity, refine_dep
    
