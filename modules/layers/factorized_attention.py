import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from modules.layers.basics import MLP, LayerNorm, GroupNorm, \
    bias_dropout_add_scale, \
    bias_dropout_add_scale_fused_train, \
    bias_dropout_add_scale_fused_inference, \
    modulate_fused

from modules.layers.embedding import RotaryEmbedding, apply_rotary_pos_emb, RadialBesselBasis


class DotProductKernel(nn.Module):
    # a simple implementation of the dot product kernel
    # by default we use rope
    def __init__(self, dim, dim_head, num_heads,
                 softmax=False,
                 qk_norm=False,     # qk-norm can force the dot product to become cosine similarity
                 scale=None,
                 ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads

        self.softmax = softmax
        if scale is None:
            if self.softmax:
                self.scale = 1. / dim_head
            else:
                self.scale = 1. / np.sqrt(dim_head)
        else:
            self.scale = scale

        self.qk_norm = qk_norm

        self.to_qk = nn.Linear(dim, 2*dim_head*num_heads, bias=False)

    def forward(self, x, rotary_cos=None, rotary_sin=None):
        # x: b n d
        # rotary_cos/sin: n d
        qk = self.to_qk(x)
        qk = rearrange(qk, 'b n (two h d) -> b h two n d', h=self.num_heads, two=2)

        if rotary_cos is not None and rotary_sin is not None:
            with torch.cuda.amp.autocast(enabled=False):

                qk = apply_rotary_pos_emb(qk, rotary_cos, rotary_sin)

        if self.qk_norm:
            qk = F.normalize(qk, p=2, dim=-1)
        q, k = qk[:, :, 0], qk[:, :, 1]

        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn * self.scale
        if self.softmax:
            attn = F.softmax(attn, dim=-1)
        return attn


class PoolingReducer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.to_out = MLP(hidden_dim, out_dim=out_dim)

    def forward(self, x, mesh_weights=None):
        # note that the dimension to be pooled will be the last dimension
        # x: b nx ... c
        # mesh_weights: ...
        x = self.to_in(x)
        # pool all spatial dimension but the first one
        ndim = len(x.shape)
        if mesh_weights is not None:
            # mesh_weights: nx
            # x: b nx ny nz ... c
            x = torch.einsum('b n ... c, ... -> b n ... c', x, mesh_weights)
        x = x.mean(dim=tuple(range(2, ndim-1)))
        x = self.to_out(x)
        return x  # b nx c


class FABlock2D(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 bottleneck_dim,
                 dim_out,
                 depth_dropout=0.1,
                 mlp_dropout=0.0,
                 kernel_expansion_ratio=1.0,
                 use_softmax=True,
                 zero_init=True):
        super().__init__()

        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.heads = heads
        self.dim_head = dim_head
        self.norm1 = LayerNorm(dim, force_fp32=True)      # norm before attention
        self.norm2 = LayerNorm(dim, force_fp32=True)      # norm before ffn
        self.dropout = nn.Dropout(depth_dropout) if depth_dropout > 0 else nn.Identity()

        self.to_v = nn.Linear(dim, dim_head*heads, bias=False)

        self.to_y = PoolingReducer(self.dim, self.dim, self.bottleneck_dim)
        self.to_x = nn.Sequential(
            Rearrange('b ny nx c -> b nx ny c'),
            PoolingReducer(self.dim, self.dim, self.bottleneck_dim),
        )

        # for attention
        self.kernel_x = DotProductKernel(self.bottleneck_dim,
                                         int(self.dim_head*kernel_expansion_ratio), self.heads,
                                         softmax=use_softmax)

        self.kernel_y = DotProductKernel(self.bottleneck_dim,
                                         int(self.dim_head*kernel_expansion_ratio), self.heads,
                                         softmax=use_softmax)

        self.merge_head = nn.Sequential(
            GroupNorm(heads, dim_head * heads, eps=1e-6, affine=False),
            nn.Linear(dim_head * heads, dim_out))

        self.ffn = MLP(dim_out,
                       expansion_ratio=4,
                       dropout=mlp_dropout)

        if zero_init:
            nn.init.zeros_(self.merge_head[1].weight)

    def forward(self, u,
                rotary_cos_sin_y, rotary_cos_sin_x,
                scalar_cond):
        # u in shape: [b ny nx c]
        rotary_cos_y, rotary_sin_y = rotary_cos_sin_y
        rotary_cos_x, rotary_sin_x = rotary_cos_sin_x

        # compute the attention
        u_skip = u
        u = self.norm1(u)
        u_x = self.to_x(u)
        u_y = self.to_y(u)

        k_x = self.kernel_x(u_x, rotary_cos_x, rotary_sin_x)
        k_y = self.kernel_y(u_y, rotary_cos_y, rotary_sin_y)

        u_phi = rearrange(self.to_v(u), 'b i l (h c) -> b h i l c', h=self.heads)
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_y, u_phi)   # convolve over y
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_x, u_phi)   # convolve over x
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)

        u = self.merge_head(u_phi) + self.dropout(u_skip)   # dropout residual

        # standard feedforward
        u = self.ffn(self.norm2(u)) + self.dropout(u)
        return u


class FADiTBlock2D(FABlock2D):
    # contains factorization and attention on each axis
    # use adaln for modulation
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 bottleneck_dim,
                 cond_dim,
                 depth_dropout=0.1,
                 mlp_dropout=0.0,
                 kernel_expansion_ratio=1.0,
                 use_softmax=True):
        super().__init__(dim,
                         dim_head,
                         heads,
                         bottleneck_dim,
                         dim,
                         depth_dropout,
                         mlp_dropout,
                         kernel_expansion_ratio,
                         use_softmax,
                         zero_init=False)

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        self.depth_dropout = depth_dropout # overwrite

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, u,
                rotary_cos_sin_y, rotary_cos_sin_x,
                scalar_cond
                ):
        # u in shape: [b ny nx c]
        # scalar_cond in shape: [b 1 c]
        ny, nx = u.shape[1], u.shape[2]
        rotary_cos_y, rotary_sin_y = rotary_cos_sin_y
        rotary_cos_x, rotary_sin_x = rotary_cos_sin_x

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(scalar_cond)[:, None, None].chunk(6, dim=-1)
        # need the additional None for broadcasting

        # compute the attention
        u_skip = u

        u = modulate_fused(self.norm1(u), shift_msa, scale_msa)
        u_x = self.to_x(u)
        u_y = self.to_y(u)

        k_x = self.kernel_x(u_x, rotary_cos_x, rotary_sin_x)
        k_y = self.kernel_y(u_y, rotary_cos_y, rotary_sin_y)

        u_phi = rearrange(self.to_v(u), 'b i l (h c) -> b h i l c', h=self.heads)
        # can additionally use mesh weights when doing the integral
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_y, u_phi)   # convolve over y
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_x, u_phi)  # convolve over x
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)

        u = bias_dropout_scale_fn(self.merge_head(u_phi),
                                  None,
                                  gate_msa,
                                  u_skip,
                                  self.depth_dropout)
        # standard feedforward
        u_skip = u
        u = modulate_fused(self.norm2(u), shift_mlp, scale_mlp)
        u = bias_dropout_scale_fn(self.ffn(u),
                                  None,
                                  gate_mlp,
                                  u_skip,
                                  self.depth_dropout)
        return u


class FADiTBlockS2(nn.Module):
    # contains factorization and attention on each axis
    # use adaln for modulation
    # instead of using rope, use radial-based positional encoding
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 bottleneck_dim,
                 dim_out,
                 cond_dim,
                 depth_dropout=0.0,
                 mlp_dropout=0.0,
                 kernel_expansion_ratio=1.0,
                 use_softmax=True,
                 zero_init=True):
        super().__init__()

        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.heads = heads
        self.dim_head = dim_head
        self.norm1 = LayerNorm(dim, force_fp32=True)  # norm before attention
        self.norm2 = LayerNorm(dim, force_fp32=True)  # norm before ffn
        self.dropout = nn.Dropout(depth_dropout) if depth_dropout > 0 else nn.Identity()

        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.to_y = PoolingReducer(self.dim, self.dim, self.bottleneck_dim)
        self.to_x = PoolingReducer(self.dim, self.dim, self.bottleneck_dim)


        self.radial_pos_emb_x = RadialBesselBasis(num_kernels=32, num_heads=heads, enforce_periodicity=True)
        self.radial_pos_emb_y = RadialBesselBasis(num_kernels=32, num_heads=heads, enforce_periodicity=False)

        # for attention
        self.kernel_x = DotProductKernel(self.bottleneck_dim,
                                         int(self.dim_head * kernel_expansion_ratio), self.heads,
                                         softmax=False,
                                         scale=1.)

        self.kernel_y = DotProductKernel(self.bottleneck_dim,
                                         int(self.dim_head * kernel_expansion_ratio), self.heads,
                                         softmax=False,
                                         scale=1.)

        self.merge_head = nn.Sequential(
            GroupNorm(heads, dim_head * heads, eps=1e-6, affine=False),
            nn.Linear(dim_head * heads, dim_out))

        self.ffn = MLP(dim_out,
                       expansion_ratio=4,
                       dropout=mlp_dropout)

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        self.depth_dropout = depth_dropout  # overwrite

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, u,
                lat,            # need latitude coordinate
                lat_diff, lon_diff,
                scalar_cond):
        # u in shape: [b nlat nlon c]
        # lat_diff, lon_diff in shape: [nlat, nlat], [nlon, nlon]
        # scalar_cond in shape: [b c]

        # compute the attention
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(scalar_cond)[:, None, None].chunk(6, dim=-1)
        # need the additional None for broadcasting

        # compute the attention
        u_skip = u

        u = modulate_fused(self.norm1(u), shift_msa, scale_msa)
        # print(u.shape)
        # when pooling to x, account for the latitude weight
        lat_weights = torch.cos(lat)
        lat_weights = lat_weights / lat_weights.mean()      # normalize

        u_x = self.to_x(rearrange(u, 'b nlat nlon c -> b nlon nlat c'), lat_weights)
        u_y = self.to_y(u)

        k_x = self.kernel_x(u_x)
        k_y = self.kernel_y(u_y)

        r_xx = self.radial_pos_emb_x(lon_diff)
        r_yy = self.radial_pos_emb_y(lat_diff)

        k_x = torch.einsum('b h i j, h i j -> b h i j', k_x, r_xx)
        k_y = torch.einsum('b h i j, h i j -> b h i j', k_y, r_yy)
        k_x = F.softmax(k_x, dim=-1)
        k_y = F.softmax(k_y, dim=-1)

        u_phi = rearrange(self.to_v(u), 'b i l (h c) -> b h i l c', h=self.heads)
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_y, u_phi)  # convolve over y
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_x, u_phi)  # convolve over x
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)

        u = bias_dropout_scale_fn(self.merge_head(u_phi),
                                  None,
                                  gate_msa,
                                  u_skip,
                                  self.depth_dropout)
        # standard feedforward
        u_skip = u
        u = modulate_fused(self.norm2(u), shift_mlp, scale_mlp)
        u = bias_dropout_scale_fn(self.ffn(u),
                                  None,
                                  gate_mlp,
                                  u_skip,
                                  self.depth_dropout)
        return u

class FABlock3D(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 bottleneck_dim,
                 dim_out,
                 depth_dropout=0.1,
                 mlp_dropout=0.0,
                 kernel_expansion_ratio=1.0,
                 use_softmax=True,
                 zero_init=True):
        super().__init__()

        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.heads = heads
        self.dim_head = dim_head
        self.norm1 = LayerNorm(dim, force_fp32=True)      # norm before attention
        self.norm2 = LayerNorm(dim, force_fp32=True)      # norm before ffn
        self.dropout = nn.Dropout(depth_dropout) if depth_dropout > 0 else nn.Identity()

        self.to_v = nn.Linear(dim, dim_head*heads, bias=False)

        self.to_z = PoolingReducer(self.dim, self.dim, self.bottleneck_dim)
        self.to_y = nn.Sequential(
            Rearrange('b nz ny nx c -> b ny nz nx c'),
            PoolingReducer(self.dim, self.dim, self.bottleneck_dim),
        )
        self.to_x = nn.Sequential(
            Rearrange('b nz ny nx c -> b nx nz ny c'),
            PoolingReducer(self.dim, self.dim, self.bottleneck_dim),
        )

        # for attention
        self.kernel_x = DotProductKernel(self.bottleneck_dim,
                                         int(self.dim_head*kernel_expansion_ratio), self.heads,
                                         softmax=use_softmax)

        self.kernel_y = DotProductKernel(self.bottleneck_dim,
                                         int(self.dim_head*kernel_expansion_ratio), self.heads,
                                         softmax=use_softmax)

        self.kernel_z = DotProductKernel(self.bottleneck_dim,
                                         int(self.dim_head * kernel_expansion_ratio), self.heads,
                                         softmax=use_softmax)

        self.merge_head = nn.Sequential(
            GroupNorm(heads, dim_head * heads, eps=1e-6, affine=False),
            nn.Linear(dim_head * heads, dim_out))

        self.ffn = MLP(dim_out,
                       expansion_ratio=4,
                       dropout=mlp_dropout)

        if zero_init:
            nn.init.zeros_(self.merge_head[1].weight)
            nn.init.zeros_(self.merge_head[1].bias)

    def forward(self, u,
                rotary_cos_sin_z,
                rotary_cos_sin_y,
                rotary_cos_sin_x,
                scalar_cond = None):

        # u in shape: [b nz ny nx c]
        rotary_cos_z, rotary_sin_z = rotary_cos_sin_z
        rotary_cos_y, rotary_sin_y = rotary_cos_sin_y
        rotary_cos_x, rotary_sin_x = rotary_cos_sin_x

        # compute the attention
        u_skip = u
        u = self.norm1(u)
        u_x = self.to_x(u)
        u_y = self.to_y(u)
        u_z = self.to_z(u)

        k_x = self.kernel_x(u_x, rotary_cos_x, rotary_sin_x)
        k_y = self.kernel_y(u_y, rotary_cos_y, rotary_sin_y)
        k_z = self.kernel_z(u_z, rotary_cos_z, rotary_sin_z)

        u_phi = rearrange(self.to_v(u), 'b nz ny nx (h c) -> b h nz ny nx c', h=self.heads)
        u_phi = torch.einsum('bhzj,bhjyxc->bhzyxc', k_z, u_phi)  # convolve over z
        u_phi = torch.einsum('bhyj,bhzjxc->bhzyxc', k_y, u_phi)  # convolve over y
        u_phi = torch.einsum('bhxj,bhzyjc->bhzyxc', k_x, u_phi)  # convolve over x

        u_phi = rearrange(u_phi, 'b h nz ny nx c -> b nz ny nx (h c)', h=self.heads)

        u = self.merge_head(u_phi) + self.dropout(u_skip)   # dropout residual

        # standard feedforward
        u = self.ffn(self.norm2(u)) + self.dropout(u)
        return u


class FADiTBlock3D(FABlock3D):
    # contains factorization and attention on each axis
    # use adaln for modulation
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 bottleneck_dim,
                 cond_dim,
                 depth_dropout=0.1,
                 mlp_dropout=0.0,
                 kernel_expansion_ratio=1.0,
                 use_softmax=True):
        super().__init__(dim,
                         dim_head,
                         heads,
                         bottleneck_dim,
                         dim,
                         depth_dropout,
                         mlp_dropout,
                         kernel_expansion_ratio,
                         use_softmax,
                         zero_init=False)

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        self.depth_dropout = depth_dropout  # overwrite

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, u,
                rotary_cos_sin_z,
                rotary_cos_sin_y,
                rotary_cos_sin_x,
                scalar_cond=None
                ):
        # u in shape: [b nz ny nx c]
        # scalar_cond in shape: [b 1 c]
        # nz， ny, nx = u.shape[1], u.shape[2]
        # print(u.shape)
        # print(scalar_cond.shape)
        # u in shape: [b nz ny nx c]
        rotary_cos_z, rotary_sin_z = rotary_cos_sin_z
        rotary_cos_y, rotary_sin_y = rotary_cos_sin_y
        rotary_cos_x, rotary_sin_x = rotary_cos_sin_x

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(scalar_cond)[:, None, None, None].chunk(6, dim=-1)
        # need the additional None for broadcasting

        # compute the attention
        u_skip = u

        u = modulate_fused(self.norm1(u), shift_msa, scale_msa)
        u_x = self.to_x(u)
        u_y = self.to_y(u)
        u_z = self.to_z(u)

        k_x = self.kernel_x(u_x, rotary_cos_x, rotary_sin_x)
        k_y = self.kernel_y(u_y, rotary_cos_y, rotary_sin_y)
        k_z = self.kernel_z(u_z, rotary_cos_z, rotary_sin_z)

        u_phi = rearrange(self.to_v(u), 'b nz ny nx (h c) -> b h nz ny nx c', h=self.heads)
        u_phi = torch.einsum('bhzj,bhjyxc->bhzyxc', k_z, u_phi)  # convolve over z
        u_phi = torch.einsum('bhyj,bhzjxc->bhzyxc', k_y, u_phi)  # convolve over y
        u_phi = torch.einsum('bhxj,bhzyjc->bhzyxc', k_x, u_phi)  # convolve over x

        u_phi = rearrange(u_phi, 'b h nz ny nx c -> b nz ny nx (h c)', h=self.heads)

        u = bias_dropout_scale_fn(self.merge_head(u_phi),
                                  None,
                                  gate_msa,
                                  u_skip,
                                  self.depth_dropout)
        # standard feedforward
        u_skip = u
        u = modulate_fused(self.norm2(u), shift_mlp, scale_mlp)
        u = bias_dropout_scale_fn(self.ffn(u),
                                  None,
                                  gate_mlp,
                                  u_skip,
                                  self.depth_dropout)
        return u
