import torch
import torch.nn as nn
import numpy as np
from common.climate_utils import assemble_input

TYPE = "group"

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, 3D, or 4D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=16, type=TYPE):
    if type == "layer":
        return torch.nn.LayerNorm(in_channels, eps=1e-6)
    elif type == "group":
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        raise ValueError(f"unknown normalization type {type}")


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=3):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = conv_nd(dim,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        if len(x.shape) == 6: # interpolate doesn't support 6D
            x = torch.kron(x, torch.ones(2, 2, 2, 2, device=x.device))  # upsample w/ kronecker product
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=3):
        super().__init__()
        self.with_conv = with_conv
        self.dim=dim
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv_nd(dim,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=0)

    def forward(self, x):
        if self.with_conv:
            if self.dim == 3:
                pad = (0,1,0,1,0,1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
            elif self.dim == 4:
                pad = (0,1,0,1,0,1,0,1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
            else:
                pad = (0,1,0,1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
        else:
            if self.dim == 3:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            else:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, dim=3, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.dim = dim

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_nd(dim,
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            padding_mode=padding_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_nd(dim,
                            out_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            padding_mode=padding_mode)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_nd(dim,
                                            in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            padding_mode=padding_mode)
            else:
                self.nin_shortcut = conv_nd(dim,
                                            in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            if self.dim == 3:
                h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None,None]
            else:
                h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels, dim=3):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim

        self.norm = Normalize(in_channels)
        self.q = conv_nd(dim,
                        in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.k = conv_nd(dim,
                        in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.v = conv_nd(dim,
                        in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.proj_out = conv_nd(dim,
                                in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dim == 4:
            b, c, t, d, h, w = q.shape
            num_tokens = h*w*d*t
        elif self.dim == 3:
            b,c,d,h,w = q.shape
            num_tokens = h*w*d
        else:
            b,c,h,w = q.shape
            num_tokens = h*w

        q = q.reshape(b,c,num_tokens)
        q = q.permute(0,2,1)   # b,hwd,c
        k = k.reshape(b,c,num_tokens) # b,c,hwd
        w_ = torch.bmm(q,k)     # b,hwd,hwd    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,num_tokens)
        w_ = w_.permute(0,2,1)   # b,hwd,hwd (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hwd (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        if self.dim == 4:
            h_ = h_.reshape(b,c,t,d,h,w)
        elif self.dim == 3: 
            h_ = h_.reshape(b,c,d,h,w)
        else:
            h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla", dim=3):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    if attn_type == "vanilla":
        return AttnBlock(in_channels, dim=dim)
    elif attn_type == "none":
        return nn.Identity(in_channels)

class Encoder(nn.Module):
    def __init__(self,
                 in_channels, # input channel dim
                 hidden_channels, # width of network
                 z_channels, # output latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 2,
                 resolution = (64, 128), 
                 attn_resolutions = [32], 
                 dropout=0.0, 
                 double_z=False, 
                 cond_channels=0,
                 tanh_out=False,
                 dim=2,
                 padding_mode='zeros',
                 use_attn=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.temb_ch = cond_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        attn_type = "vanilla"
        resamp_with_conv = True
        self.tanh_out = tanh_out
        self.dim = dim 

        # downsampling
        self.conv_in = conv_nd(dim,
                                in_channels,
                                self.hidden_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode=padding_mode)

        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)

        curr_res = resolution[-1]
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.hidden_channels*in_ch_mult[i_level]
            block_out = self.hidden_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         dim=dim,
                                         padding_mode=padding_mode))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dim=dim))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv, dim=dim)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode)
        if use_attn:
            self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dim=dim)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_nd(dim,
                                block_in,
                                2*z_channels if double_z else z_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode=padding_mode)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, conv_nd):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cond=None):
        # x in shape b c h w

        temb = cond
        # downsampling  
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        if self.tanh_out:
            h = torch.tanh(h) # clamp between -1 and 1. Useful for diffusion models

        return h


class Decoder(nn.Module):
    def __init__(self,
                 out_channels, # output channel dim
                 hidden_channels, # width of network
                 z_channels, # input latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 2,
                 resolution = (64, 128), 
                 attn_resolutions = [32], 
                 dropout=0.0, 
                 double_z=True, 
                 cond_channels=0,
                 tanh_out=False,
                 dim=2,
                 padding_mode='zeros',
                 use_attn=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.temb_ch = cond_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = False
        self.tanh_out = tanh_out
        attn_type = "vanilla"
        resamp_with_conv = True
        self.dim = dim

        print("Initializing decoder")

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.hidden_channels*ch_mult[self.num_resolutions-1]

        curr_res = resolution[-1] // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = conv_nd(dim,
                                z_channels,
                                block_in,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode=padding_mode)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode)
        if use_attn:
            self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dim=dim)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.hidden_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         dim=dim,
                                         padding_mode=padding_mode))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dim=dim))
                    print(f"added attn at res {curr_res}")
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, dim=dim)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_nd(dim,
                                block_in,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode=padding_mode)

    def forward(self, z, cond=None):
        # z in shape b c h w

        temb = cond
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w
        if self.tanh_out:
            h = torch.tanh(h)

        return h
    