import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np

def make_actv(actv):
    if actv == 'relu':
        return nn.ReLU(inplace=True)
    elif actv == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif actv == 'exp':
        return lambda x: torch.exp(x)
    elif actv == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif actv == 'tanh':
        return lambda x: torch.tanh(x)
    elif actv == 'softplus':
        return lambda x: torch.log(1 + torch.exp(x - 1))
    elif actv == 'linear':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid activation function: {:s}'.format(actv)
        )

def make_norm2d(name, plane, affine=True):
    if name == 'batch':
        return nn.BatchNorm2d(plane, affine=affine)
    elif name == 'instance':
        return nn.InstanceNorm2d(plane, affine=affine)
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid normalization function: {:s}'.format(name)
        )

def make_norm3d(name, plane, affine=True, per_channel=True):
    if name == 'batch':
        return nn.BatchNorm3d(plane, affine=affine)
    elif name == 'instance':
        return nn.InstanceNorm3d(plane, affine=affine)
    elif name == 'max':
        return MaxNorm(per_channel=per_channel)
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(
            'invalid normalization function: {:s}'.format(name)
        )


class MaxNorm(nn.Module):
    """ Per-channel normalization by max value """
    def __init__(self, per_channel=True, eps=1e-8):
        super(MaxNorm, self).__init__()

        self.per_channel = per_channel
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c, d, h, w)): raw RSD output.

        Returns:
            x (float tensor, (bs, c, d, h, w)): normalized RSD output.
        """
        assert x.dim() == 5, \
            'input should be a 5D tensor, got {:d}D'.format(x.dim())

        if self.per_channel:
            x = F.normalize(x, p=float('inf'), dim=(-3, -2, -1))
        else:
            x = F.normalize(x, p=float('inf'), dim=(-4, -3, -2, -1))
        return x


class Blur2d(nn.Module):
    """ 2D blur kernel """
    def __init__(self, in_plane):
        super(Blur2d, self).__init__()

        self.in_plane = in_plane

        weight = torch.Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        weight = weight.expand(in_plane, -1, -1).unsqueeze(1)
        self.register_buffer('weight', weight)        # (c, 1, 3, 3)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x = F.conv2d(x, self.weight, groups=self.in_plane)
        return x


class ResConv2d(nn.Module):
    """ Residual block with 2D conv layers """
    def __init__(
        self, 
        in_plane,           # number of input planes.
        plane,              # number of intermediate and output planes
        stride=1,           # stride of first conv layer
        actv='leaky_relu',  # activation function
        norm='instance',    # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
    ):
        super(ResConv2d, self).__init__()

        self.in_plane = in_plane
        self.plane = plane
        self.stride = stride
        bias = True if norm == 'none' or not affine else False

        self.conv1 = nn.Conv2d(
            in_plane, plane, 3, stride, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm1 = make_norm2d(norm, plane, affine)
        
        self.conv2 = nn.Conv2d(
            plane, plane, 3, 1, 1,
            padding_mode='replicate', bias=bias
        )
        self.norm2 = make_norm2d(norm, plane, affine)

        if stride > 1 or in_plane != plane:
            self.res_conv = nn.Conv2d(in_plane, plane, 1, stride, 0, bias=bias)
            self.res_norm = make_norm2d(norm, plane, affine)

        self.actv = make_actv(actv)

    def forward(self, x):
        dx = self.norm1(self.conv1(x))
        dx = self.actv(dx)
        dx = self.norm2(self.conv2(dx))
        if self.stride > 1 or self.in_plane != self.plane:
            x = self.res_norm(self.res_conv(x))
        x = self.actv(x + dx)
        return x


class ResConv3d(nn.Module):
    """ Residual block with 3D conv layers """
    def __init__(
        self, 
        in_plane,           # number of input planes
        plane,              # number of intermediate and output planes
        stride=1,           # stride of first conv layer
        actv='leaky_relu',  # activation function
        norm='none',        # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
    ):
        super(ResConv3d, self).__init__()

        self.in_plane = in_plane
        self.plane = plane
        self.stride = stride
        bias = True if norm == 'none' or not affine else False

        self.conv1 = nn.Conv3d(
            in_plane, plane, 3, stride, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm1 = make_norm3d(norm, plane, affine)
        self.conv2 = nn.Conv3d(
            plane, plane, 3, 1, 1, 
            padding_mode='replicate', bias=bias
        )
        self.norm2 = make_norm3d(norm, plane, affine)

        if stride > 1 or in_plane != plane:
            self.res_conv = nn.Conv3d(in_plane, plane, 1, stride, 0, bias=bias)
            self.res_norm = make_norm3d(norm, plane, affine)

        self.actv = make_actv(actv)
    
    def forward(self, x):
        dx = self.norm1(self.conv1(x))
        dx = self.actv(dx)
        dx = self.norm2(self.conv2(dx))
        if self.stride > 1 or self.in_plane != self.plane:
            x = self.res_norm(self.res_conv(x))
        x = self.actv(x + dx)
        return x


class ResBlock2d(nn.Module):

    def __init__(
        self, 
        in_plane, 
        plane, 
        stride, 
        n_layers,  
        actv='relu', 
        norm='none', 
        affine=False,
    ):
        super(ResBlock2d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv2d(in_plane, plane, stride, actv, norm, affine)
            )
            in_plane = plane
            stride = 1
        self.layers = nn.Sequential(*layers)

        self.out_plane = in_plane

    def forward(self, x):
        x = self.layers(x)
        return x


class ResBlock3d(nn.Module):

    def __init__(
        self, 
        in_plane, 
        plane, 
        stride, 
        n_layers,  
        actv='relu', 
        norm='none', 
        affine=False,
    ):
        super(ResBlock3d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv3d(in_plane, plane, stride, actv, norm, affine)
            )
            in_plane = plane
            stride = 1
        self.layers = nn.Sequential(*layers)

        self.out_plane = in_plane

    def forward(self, x):
        x = self.layers(x)
        return x


class UpSample2d(nn.Module):
    """ 2D up-sampling layer """
    def __init__(
        self, 
        in_plane,           # number of input planes
        out_plane,          # number of output planes
        mode='bilinear',    # up-sampling method 
        refine='conv',      # refinement method following up-sampling
        actv='leaky_relu',  # activation function
        norm='instance',    # normalization function
        affine=True,        # if True, apply learnable affine transform in norm
    ):
        super(UpSample2d, self).__init__()
        assert mode in ('transpose', 'bilinear', 'nearest', 'shuffle'), \
            'invalid up-sampling method: {:s}'.format(mode)
        assert refine in ('none', 'conv', 'blur'), \
            'invalid refinement method: {:s}'.format(refine)

        bias = True if norm == 'none' or not affine else False

        if mode == 'transpose':
            block = nn.Sequential(
                nn.ConvTranspose2d(in_plane, out_plane, 4, 2, 1, bias=bias),
                make_norm2d(norm, out_plane, affine), 
                make_actv(actv),
            )
        elif mode == 'bilinear':
            block = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False
            )
            if refine == 'conv':
                block = nn.Sequential(
                    block,
                    nn.Conv2d(in_plane, out_plane, 3, 1, 1, bias=bias),
                    make_norm2d(norm, out_plane, affine),
                    make_actv(actv),
                )
            elif refine == 'blur':
                block = nn.Sequential(
                    block, 
                    Blur2d(in_plane),
                )
        elif mode == 'nearest':
            block = nn.Upsample(
                scale_factor=2, mode='nearest', align_corners=False
            )
            if refine == 'conv':
                block = nn.Sequential(
                    block,
                    nn.Conv2d(in_plane, out_plane, 3, 1, 1, bias=bias),
                    make_norm2d(norm, out_plane, affine),
                    make_actv(actv),
                )
            elif refine == 'blur':
                block = nn.Sequential(
                    block, 
                    Blur2d(in_plane),
                )
        else:
            block = nn.Sequential(
                nn.PixelShuffle(upscale_factor=2),
                nn.Conv2d(in_plane // 4, out_plane, 3, 1, 1, bias=bias),
                make_norm2d(norm, out_plane, affine),
                make_actv(actv),
            )

        self.block = block

    def forward(self, x):
        x = self.block(x)
        return x


class DownSample3d(nn.Module):
    """ 3D down-sampling layer """
    def __init__(
        self, 
        in_plane, 
        out_plane, 
        actv='leaky_relu', 
        norm='instance', 
        affine=True
    ):
        super(DownSample3d, self).__init__()

        bias = True if norm == 'none' or not affine else False

        self.block = nn.Sequential(
            nn.Conv3d(
                in_plane, out_plane, 4, 2, 1, 
                padding_mode='replicate', bias=bias
            ),
            make_norm3d(norm, out_plane, affine),
            make_actv(actv))

    def forward(self, x):
        x = self.block(x)
        return x


class UpSample3d(nn.Module):
    """ 3D up-sampling layer """
    def __init__(
        self, 
        in_plane, 
        out_plane,
        actv='relu', 
        norm='instance', 
        affine=True,
    ):
        super(UpSample3d, self).__init__()

        bias = True if norm == 'none' or not affine else False

        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_plane, out_plane, 4, 2, 1, bias=bias),
            make_norm3d(norm, out_plane, affine),
            make_actv(actv))

    def forward(self, x):
        x = self.block(x)
        return x


class UNetBlock(nn.Module):
    """ UNet block (a.k.a. one level of UNet) """
    def __init__(
        self, 
        outer_plane, 
        inner_plane, 
        submodule=None, 
        skip=None,
        down_actv='leaky_relu', 
        up_actv='relu', 
        norm='instance', 
        affine=True,
    ):
        super(UNetBlock, self).__init__()

        if submodule is None:
            if skip is None:
                self.block = nn.Sequential(
                    DownSample3d(
                        in_plane=outer_plane, 
                        out_plane=inner_plane, 
                        actv=down_actv, 
                        norm=norm, 
                        affine=affine,
                    ),
                    UpSample3d(
                        in_plane=inner_plane, 
                        out_plane=outer_plane, 
                        actv=up_actv, 
                        norm=norm, 
                        affine=affine,
                    ),
                )
                self.skip = nn.Identity()
                self.out_plane = outer_plane * 2
            else:
                self.block = skip
                self.skip = None
                self.out_plane = outer_plane
        else:
            self.block = nn.Sequential(
                DownSample3d(
                    in_plane=outer_plane, 
                    out_plane=inner_plane, 
                    actv=down_actv, 
                    norm=norm, 
                    affine=affine,
                ),
                submodule,
                UpSample3d(
                    in_plane=submodule.out_plane, 
                    out_plane=outer_plane, 
                    actv=up_actv, 
                    norm=norm, 
                    affine=affine,
                ),
            )
            self.skip = skip if skip is not None else nn.Identity()
            self.out_plane = outer_plane * 2

    def forward(self, x):
        y = self.block(x)
        if self.skip is not None:
            y = torch.cat([self.skip(x), y], dim=1)
        return y


class FFC2d(nn.Module):

    def __init__(
        self,
        in_plane,
        plane,
        actv='relu',
        norm='batch',
        affine=True,
        pe=False,
    ):
        super(FFC2d, self).__init__()

        in_dim = in_plane * 2
        if pe:
            in_dim += 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, plane * 2, 1, bias=False),
            make_norm2d(norm, plane * 2, affine),
            make_actv(actv),
        )

        self.pe = pe

    def forward(self, x):
        # FFT
        x = fft.fftn(x, s=[-1, -1], norm='ortho')
        x = fft.fftshift(x, dim=(-2, -1))
        x = torch.cat([x.real, x.imag], dim=1)
        
        # position encoding
        if self.pe:
            bs, _, h, w = x.size()
            h_tics = torch.linspace(-1, 1, h, device=x.device)
            w_tics = torch.linspace(-1, 1, w, device=x.device)
            pe = torch.stack(torch.meshgrid(h_tics, w_tics, indexing='ij'))
            pe = pe.expand(bs, -1, -1, -1)
            x = torch.cat([x, pe], dim=1)

        # conv
        x = self.conv(x)

        # IFFT
        x = torch.complex(*x.split(x.size(1) // 2, dim=1))
        x = fft.ifftshift(x, dim=(-2, -1))
        x = fft.ifftn(x, s=[-1, -1], norm='ortho')

        return x


class FFC3d(nn.Module):

    def __init__(
        self,
        in_plane,
        plane,
        actv='relu',
        norm='batch',
        affine=True,
        pe=False,
    ):
        super(FFC3d, self).__init__()

        in_dim = in_plane * 2
        if pe:
            in_dim += 3

        self.conv = nn.Sequential(
            nn.Conv3d(in_dim, plane * 2, 1, bias=False),
            make_norm3d(norm, plane * 2, affine),
            make_actv(actv),
        )

        self.pe = pe

    def forward(self, x):
        # FFT
        x = fft.fftn(x, s=[-1, -1, -1], norm='ortho')
        x = fft.fftshift(x, dim=(-3, -2, -1))
        x = torch.cat([x.real, x.imag], dim=1)

        # position encoding
        if self.pe:
            bs, _, t, h, w = x.size()
            t_tics = torch.linspace(-1, 1, t, device=x.device)
            h_tics = torch.linspace(-1, 1, h, device=x.device)
            w_tics = torch.linspace(-1, 1, w, device=x.device)
            pe = torch.stack(
                torch.meshgrid(t_tics, h_tics, w_tics, indexing='ij')
            )
            pe = pe.expand(bs, -1, -1, -1, -1)
            x = torch.cat([x, pe], dim=1)
        
        # conv
        x = self.conv(x)

        # IFFT
        x = torch.complex(*x.split(x.size(1) // 2, dim=1))
        x = fft.ifftshift(x, dim=(-3, -2, -1))
        x = fft.ifftn(x, s=[-1, -1, -1], norm='ortho')

        return x


class ResFFC2d(nn.Module):

    def __init__(
        self, 
        plane,
        bottleneck=True,
        expansion=4,
        actv='relu',
        norm='batch',
        affine=True,
        pe=False,
    ):
        super(ResFFC2d, self).__init__()

        hid_plane = plane // expansion if bottleneck else plane

        self.ffc = FFC2d(hid_plane, hid_plane, actv, norm, affine, pe)

        self.bottleneck = bottleneck
        if bottleneck:
            self.conv1 = nn.Conv2d(plane * 2, hid_plane * 2, 1, bias=False)
            self.conv2 = nn.Conv2d(hid_plane * 2, plane * 2, 1, bias=False)
            self.bn1 = make_norm2d(norm, hid_plane * 2, affine)
            self.bn2 = make_norm2d(norm, plane * 2, affine)
            self.actv = make_actv(actv)

    def forward(self, x):
        if self.bottleneck:
            x = torch.cat([x.real, x.imag], dim=1)
            dx = self.actv(self.bn1(self.conv1(x)))
            dx = torch.complex(*dx.split(dx.size(1) // 2, dim=1))
        else:
            dx = x

        dx = self.ffc(dx)

        if self.bottleneck:
            dx = torch.cat([dx.real, dx.imag], dim=1)
            dx = self.bn2(self.conv2(dx))
            x = self.actv(x + dx)
            x = torch.complex(*x.split(x.size(1) // 2, dim=1))
        else:
            x = x + dx
        
        return x


class ResFFC3d(nn.Module):

    def __init__(
        self, 
        plane,
        bottleneck=True,
        expansion=4,
        actv='relu',
        norm='batch',
        affine=True,
        pe=False,
    ):
        super(ResFFC3d, self).__init__()

        hid_plane = plane // expansion if bottleneck else plane

        self.ffc = FFC3d(hid_plane, hid_plane, actv, norm, affine, pe)

        self.bottleneck = bottleneck
        if bottleneck:
            self.conv1 = nn.Conv3d(plane, hid_plane, 1, bias=False)
            self.conv2 = nn.Conv3d(hid_plane, plane, 1, bias=False)
            self.bn1 = make_norm3d(norm, hid_plane, affine)
            self.bn2 = make_norm3d(norm, plane, affine)
            self.actv = make_actv(actv)

    def forward(self, x):
        if self.bottleneck:
            x = torch.cat([x.real, x.imag], dim=1)
            dx = self.actv(self.bn1(self.conv1(x)))
            dx = torch.complex(*dx.split(dx.size(1) // 2, dim=1))
        else:
            dx = x

        dx = self.ffc(dx)
        
        if self.bottleneck:
            dx = torch.cat([dx.real, dx.imag], dim=1)
            dx = self.bn2(self.conv2(dx))
            x = self.actv(x + dx)
            x = torch.complex(*x.split(x.size(1) // 2, dim=1))
        else:
            x = x + dx

        return x


class ResBlockFFC2d(nn.Module):

    def __init__(
        self, 
        plane,
        n_layers,
        bottleneck=True,
        expansion=4,
        actv='relu',
        norm='batch',
        affine=True,
        pe=False,
    ):
        super(ResBlockFFC2d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ResFFC2d(
                    plane=plane, 
                    bottleneck=bottleneck, 
                    expansion=expansion, 
                    actv=actv, 
                    norm=norm, 
                    affine=affine, 
                    pe=pe,
                )
            )
        self.layers = nn.Sequential(*layers)

        self.out_plane = plane

    def forward(self, x):
        x = self.layers(x)
        return x


class ResBlockFFC3d(nn.Module):

    def __init__(
        self, 
        plane,
        n_layers,
        bottleneck=True,
        expansion=4,
        actv='relu',
        norm='batch',
        affine=True,
        pe=False,
    ):
        super(ResBlockFFC3d, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                ResFFC3d(
                    plane=plane,
                    bottleneck=bottleneck, 
                    expansion=expansion, 
                    actv=actv, 
                    norm=norm, 
                    affine=affine, 
                    pe=pe,
                )
            )
        self.layers = nn.Sequential(*layers)

        self.out_plane = plane

    def forward(self, x):
        x = self.layers(x)
        return x
    
    

class ResConv3D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
        )
        self.inplace = inplace
    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


class RendererV0(nn.Module):
    """
    A rendering module that maps 2D spatial domain features to an image.
    (NOTE: this implementation strictly follows Chen et al., SIGGRAPH Asia 2020 
     for reproducibility)
    """
    def __init__(
        self, 
        in_plane,
        out_plane=1,
        actv="leaky_relu",
        norm="none",
        affine=False,
    ):
        super(RendererV0, self).__init__()
        self.out_plane = out_plane

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, 1, 1),
            ResBlock2d(in_plane, in_plane, 1, 2, actv, norm, affine),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, 1, 1),
            ResBlock2d(in_plane, in_plane, 1, 2, actv, norm, affine),
        )
        self.out_conv = nn.Conv2d(in_plane, out_plane, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out_conv(x)
        return x
    

class RendererV0_UP(nn.Module):
    """
    A rendering module that maps 2D spatial domain features to an image.
    (NOTE: this implementation strictly follows Chen et al., SIGGRAPH Asia 2020 
     for reproducibility)
    """
    def __init__(
        self, 
        in_plane,
        out_plane=1,
        actv="leaky_relu",
        norm="none",
        affine=False,
    ):
        super(RendererV0_UP, self).__init__()

        self.out_plane = out_plane

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, 1, 1),
            ResBlock2d(in_plane, in_plane, 1, 2, actv, norm, affine),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_plane + out_plane, in_plane * 2, 3, 1, 1),
            ResBlock2d(in_plane * 2, in_plane * 2, 1, 2, actv, norm, affine),
        )
        self.out_conv = nn.Conv2d(in_plane * 2, out_plane, 3, 1, 1)

    def forward(self, x):
        x0 = F.interpolate(
            x[:, :self.out_plane], 
            scale_factor=2, 
            mode="bilinear", 
            align_corners=False,
        )
        x = F.interpolate(
            self.conv1(x), 
            scale_factor=2, 
            mode="bilinear", 
            align_corners=False,
        )
        x = self.conv2(torch.cat([x0, x], dim=1))
        x = x0 + self.out_conv(x)
        return x




class Interpsacle2d(nn.Module):
    
    def __init__(self, factor=2, gain=1, align_corners=False):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super(Interpsacle2d, self).__init__()
        self.gain = gain
        self.factor = factor
        self.align_corners = align_corners

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        
        x = nn.functional.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=self.align_corners)
        
        return x


class ResConv2D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv2D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


class Rendering(nn.Module):
    
    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering, self).__init__()
        
        ######################################
        assert out_channels == 1
        
        weights = np.zeros((1, 2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, 1:, :, :] = 1.0
        else:
            weights[:, :1, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        self.resize = Interpsacle2d(factor=2, gain=1, align_corners=False)
        
        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + 1,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )
    
    def forward(self, x0):
        
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:1, :, :]
        x0_dep = x0[:, dim:dim + 1, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ###################################
        x1 = self.conv1(x0)
        x1_up = self.resize(x1)
        
        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)
        
        re = x0_conv_256 + 1 * x2
        
        return re
  


class Rendering_128(nn.Module):
    
    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering_128, self).__init__()
        
        ######################################
        assert out_channels == 1
        
        weights = np.zeros((1, 2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, 1:, :, :] = 1.0
        else:
            weights[:, :1, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        # self.resize = Interpsacle2d(factor=2, gain=1, align_corners=False)
        
        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + 1,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )
    
    def forward(self, x0):
        
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:1, :, :]
        x0_dep = x0[:, dim:dim + 1, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = x0_raw_128 # self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ###################################
        x1 = self.conv1(x0)
        x1_up = x1 #self.resize(x1)
        
        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)
        
        re = x0_conv_256 + 1 * x2
        
        return re
  

