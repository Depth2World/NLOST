

import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
# sys.path.append('../utils')
# from helper import definePsf, resamplingOperator, \
# filterLaplacian
from .helper import definePsf, resamplingOperator,filterLaplacian


class lct(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 method='lct', material='diffuse',dnum=1):
        super(lct, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        #############################################################
        self.method = method
        self.material = material
        
        self.parpareparam()
        self.register_buffer("gridz_1xMx1x1_todev", self.gridz_1xMx1x1, persistent=False)
        self.register_buffer("datapad_Dx2Tx2Hx2W", torch.zeros((dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32), persistent=False)
        self.register_buffer("mtx_MxM_todev", self.mtx_MxM, persistent=False)
        self.register_buffer("mtxi_MxM_todev", self.mtxi_MxM, persistent=False)
        self.register_buffer("invpsf_real_todev", self.invpsf_real, persistent=False)
        self.register_buffer("invpsf_imag_todev", self.invpsf_imag, persistent=False)
        
        if self.method == 'bp':
            self.register_buffer("lapw_todev", self.lapw, persistent=False)
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        # maybe learnable?
        self.snr = 1e-1
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        # 0-1
        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        
        if self.method == 'lct':
            invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        elif self.method == 'bp':
            invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
        #############################################################
        if self.method == 'bp':
            lapw_kxkxk = filterLaplacian()
            k = lapw_kxkxk.shape[0]
            self.pad = nn.ReplicationPad3d(2)
            self.lapw = torch.from_numpy(lapw_kxkxk).reshape(1, 1, k, k, k)
        
    
    
    def forward(self, feture_bxdxtxhxw, tbes=[0,0,0], tens=[512,512,512]):
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        dev = feture_bxdxtxhxw.device
        
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, self.crop, hnum, wnum)
        
        gridz_1xMx1x1 = self.gridz_1xMx1x1_todev
        if self.material == 'diffuse':
            data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 4)
        elif self.material == 'specular':
            data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 2)
        
        ###############################################################
        # datapad_BDx2Tx2Hx2W = torch.zeros((bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_Dx2Tx2Hx2W = self.datapad_Dx2Tx2Hx2W
        # create new variable
        datapad_BDx2Tx2Hx2W = datapad_Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)

        left = self.mtx_MxM_todev
        right = data_BDxTxHxW.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        
        ####################################################################################
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        # datafre = torch.rfft(datapad_BDx2Tx2Hx2W, 3, onesided=False)

        datafre_temp = torch.fft.fftn(datapad_BDx2Tx2Hx2W, dim = (-3,-2,-1))
        datafre_real = datafre_temp.real #datafre[:, :, :, :, 0]
        datafre_imag = datafre_temp.imag #datafre[:, :, :, :, 1]
        # datafre_real = datafre[:, :, :, :, 0]
        # datafre_imag = datafre[:, :, :, :, 1]
        
        re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
        re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
        # refre = torch.stack([re_real, re_imag], dim=4)
        # re = torch.ifft(refre, 3)
        
        refre = torch.complex(re_real,re_imag)
        re_temp = torch.fft.ifftn(refre,dim=(-3,-2,-1))
        re = torch.stack((re_temp.real, re_temp.imag), -1)

        volumn_BDxTxHxW = re[:, :temprol_grid, :sptial_grid, :sptial_grid, 0]
        
        #########################################################################
        left = self.mtxi_MxM_todev
        right = volumn_BDxTxHxW.reshape(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        
        if self.method == 'bp':
            volumn_BDx1xTxHxW = volumn_BDxTxHxW.unsqueeze(1)
            lapw = self.lapw_todev
            volumn_BDx1xTxHxW = self.pad(volumn_BDx1xTxHxW)
            volumn_BDx1xTxHxW = F.conv3d(volumn_BDx1xTxHxW, lapw)
            volumn_BDxTxHxW = volumn_BDx1xTxHxW.squeeze(1)
            # seems border  is bad
            # if self.crop == 512:
            if True:
                volumn_BDxTxHxW[:, :1] = 0
                # volumn_BDxTxHxW[:, -10:] = 0
            # volumn_BDxTxHxW = F.relu(volumn_BDxTxHxW, inplace=False)
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        
        return volumn_BxDxTxHxW


if __name__ == '__main__':
    

    
    import os
    import cv2
    import numpy as np
    
    syn = False
    if syn:
        path = '/data/liyue/FK/LFE_render/NLOS_EF_allviews_scale_0.75processed/0/ff1fb5a9dd255b059036436a751dc6c2/shine_0.0000-rot_8.9286_-62.6221_5.4496-shift_0.1621_0.3305_-0.1085/video-confocal-gray-full.mp4'
        #'/data/liyue/FK/LFE_render/video-confocal-gray-full.mp4'
        cap = cv2.VideoCapture(path)
        # assert cap.isOpened() 
        ims = []
        i = 0
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
        # x = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]

        ims = ims[:512]
        rect_data_txhxw = np.array(ims, dtype=np.float32) / 255.0
        print(rect_data_txhxw.shape)
        rect_data_hxwxt = rect_data_txhxw.transpose(1,2,0)
        sptial_grid = 256
        crop = 512
        bin_len = 32e-12 * 3e8
        tbe = 0
        tlen = crop
    else:
        from scipy.io import loadmat
        data = loadmat('/data/liyue/FK/fk_lfe/datasets/statue0.mat')
        rect_data_hxwxt = data['measlr']
        sptial_grid = 256
        crop = 512
        bin_len = 32e-12 * 3e8  # 0.01
        tbe = 0
        tlen = crop

    Down_num = 1
    temp_down = False
    for k in range(Down_num):
        rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]
        rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]
        sptial_grid = sptial_grid // 2
        if temp_down:
            rect_data_hxwxt = rect_data_hxwxt[:, :, ::2] + rect_data_hxwxt[:, :, 1::2]
            crop = crop // 2
            bin_len = bin_len * 2 
        
    rect_data_dxhxwxt = np.expand_dims(rect_data_hxwxt, axis=0)
    rect_data_bxdxhxwxt = np.expand_dims(rect_data_dxhxwxt, axis=0)
    bnum = 1
    dnum = 1
    rect_data_bxdxhxwxt = np.tile(rect_data_bxdxhxwxt, [bnum, dnum, 1, 1, 1])
    rect_data_bxdxhxwxt = torch.from_numpy(rect_data_bxdxhxwxt).cuda()
    print(rect_data_bxdxhxwxt.shape)
    
    # import ipdb
    # ipdb.set_trace()
    #####################################################################
    # lctlayer = phasor(spatial=sptial_grid, crop=crop, bin_len=bin_len, sampling_coeff=2.0, cycles=5).cuda()
    lctlayer = lct(spatial=sptial_grid, crop=crop, bin_len=bin_len,
                   method='bp').cuda()
    # lctlayer = nn.DataParallel(lctlayer)
    
    if temp_down:
        tbe = tbe // (2 ** Down_num)
        tlen = tlen // (2 ** Down_num)
    
    for i in range(1):
        print(i)
        # import ipdb
        # ipdb.set_trace()
        re = lctlayer(rect_data_bxdxhxwxt[:, :, :, :, tbe:tbe + tlen].permute(0, 1, 4, 2, 3), \
                      [tbe, tbe, tbe], [tbe + tlen, tbe + tlen, tbe + tlen])
    
    volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
    zdim = volumn_MxNxN.shape[0] * 100 // 128
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    
    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
    print(front_view.shape)
    cv2.imwrite("re.png", (front_view / np.max(front_view))*255)
    # cv2.imshow("gt", imgt)
    # cv2.waitKey
    
    volumn_ZxYxX = volumn_MxNxN
    volumn_ZxYxX = volumn_ZxYxX / np.max(volumn_ZxYxX)
    # for i, frame in enumerate(volumn_ZxYxX):
    #     print(i)
        # cv2.imshow("re1", frame)
        # cv2.imshow("re2", frame / np.max(frame))
        # cv2.waitKey(0)
    

