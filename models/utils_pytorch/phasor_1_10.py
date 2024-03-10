

from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import  numpy as np
import sys
from .helper import definePsf, resamplingOperator, waveconvparam, waveconv


class phasor(nn.Module):
    
    def __init__(self, spatial=128, crop=256, \
                 bin_len=0.02, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5,dnum=2):
        super(phasor, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
        self.register_buffer("virtual_cos_sin_wave_inv_2x1xk_todev", self.virtual_cos_sin_wave_inv_2x1xk, persistent=False)
        datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32)
        self.register_buffer("datapad_2Dx2Tx2Hx2W", datapad_2Dx2Tx2Hx2W, persistent=False)
        self.register_buffer("mtx_MxM_todev", self.mtx_MxM, persistent=False)
        self.register_buffer("mtxi_MxM_todev", self.mtxi_MxM, persistent=False)
        self.register_buffer("invpsf_real_todev", self.invpsf_real, persistent=False)
        self.register_buffer("invpsf_imag_todev", self.invpsf_imag, persistent=False)
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
        
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        
        virtual_cos_wave_k, virtual_sin_wave_k = \
        waveconvparam(bin_resolution, virtual_wavelength, cycles)
        
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0)
        
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        # lct
        # invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        # bp
        invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
        
    def forward(self, feture_bxdxtxhxw, tbes=[0,0,0], tens=[256,256,256]):
        
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
        tnum = self.crop
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum)
        
        ############################################################
        # Step 1: convolve measurement volume with virtual wave
        
        data_BDxHxWxT = data_BDxTxHxW.permute(0, 2, 3, 1)
        data_BDHWx1xT = data_BDxHxWxT.reshape(-1, 1, tnum)
        knum = self.virtual_cos_sin_wave_inv_2x1xk.shape[2]
        phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev, padding=knum // 2)
        if knum % 2 == 0:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T[:, :, 1:]
        else:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T
        data_BDxHxWx2xT = data_BDHWx2xT.reshape(bnum * dnum, hnum, wnum, 2, tnum)
        data_2xBDxTxHxW = data_BDxHxWx2xT.permute(3, 0, 4, 1, 2)
        data_2BDxTxHxW = data_2xBDxTxHxW.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        # datapad_2BDx2Tx2Hx2W = torch.zeros((2 * bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        # create new variable
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        # actually, because it is all zero so it is ok
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W
        
        left = self.mtx_MxM_todev
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        ###########################################################3
        # Step 3: convolve with backprojection kernel
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        
        # datafre = torch.rfft(datapad_2BDx2Tx2Hx2W, 3, onesided=False)   # pytorch 1.6 
        datafre_temp = torch.fft.fftn(datapad_2BDx2Tx2Hx2W, dim = (-3,-2,-1))
        # datafre = torch.stack((datafre_temp.real, datafre_temp.imag), -1)

        datafre_real = datafre_temp.real #datafre[:, :, :, :, 0]
        datafre_imag = datafre_temp.imag #datafre[:, :, :, :, 1]
        
        re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
        re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
        # refre = torch.stack([re_real, re_imag], dim=4)  # pytorch 1.6 
        
        # re = torch.ifft(refre, 3)   # pytorch 1.6 
        refre = torch.complex(re_real,re_imag)
        re_temp = torch.fft.ifftn(refre,dim=(-3,-2,-1))
        re = torch.stack((re_temp.real, re_temp.imag), -1)
       
        volumn_2BDxTxHxWx2 = re[:, :temprol_grid, :sptial_grid, :sptial_grid, :]
        
        ########################################################################
        # Step 4: compute phasor field magnitude and inverse LCT
        cos_real = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 0]
        cos_imag = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 1]
        sin_real = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 0]
        sin_imag = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 1]
        sum_real = cos_real ** 2 - cos_imag ** 2 + sin_real ** 2 - sin_imag ** 2
        sum_image = 2 * cos_real * cos_imag + 2 * sin_real * sin_imag
        tmp = (torch.sqrt(sum_real ** 2 + sum_image ** 2) + sum_real) / 2
        # numerical issue

        tmp = F.relu(tmp, inplace=False)
        sqrt_sum_real = torch.sqrt(tmp)
        #####################################################################
        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        ########################################################################
        # do we force to be > 0?
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        return volumn_BxDxTxHxW


if __name__ == '__main__':
    
    import os
    import cv2
    import numpy as np
    
    syn = True
    if syn:
        path = '/data2/yueli/dataset/LFE_dataset/NLOS_bike_allviews_processed/0/2d655fc4ecb2df2a747c19778aa6cc0/shine_0.0000-rot_-6.2611_94.6657_3.2003-shift_-0.3453_0.2756_-0.1789/video-confocal-gray-full.mp4'
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

    Down_num = 2
    temp_down = False
    for k in range(Down_num):
        rect_data_hxwxt = rect_data_hxwxt[::2, ::2, :]
        # rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]
        # rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]
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
    lctlayer = phasor(spatial=sptial_grid, crop=crop, bin_len=bin_len, sampling_coeff=2.0, cycles=5,dnum=1).cuda()
    lctlayer = nn.DataParallel(lctlayer)
    
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
    cv2.imwrite("re_rsd.png", (front_view / np.max(front_view))*255)
    # cv2.imshow("gt", imgt)
    # cv2.waitKey
    
    volumn_ZxYxX = volumn_MxNxN
    volumn_ZxYxX = volumn_ZxYxX / np.max(volumn_ZxYxX)
    # for i, frame in enumerate(volumn_ZxYxX):
    #     print(i)
        # cv2.imshow("re1", frame)
        # cv2.imshow("re2", frame / np.max(frame))
        # cv2.waitKey(0)
    
