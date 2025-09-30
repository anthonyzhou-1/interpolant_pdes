import torch
import torch.nn as nn
from torch.nn import functional as F
import typing
from einops import rearrange

class GroupNorm(nn.Module):
    # group norm with channel at the last dimension
    def __init__(self, num_groups, num_channels,
                 domain_wise=False,
                 eps=1e-8, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.domain_wise = domain_wise
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

    def forward(self, x):
        # b h w c
        b, c = x.shape[0], x.shape[-1]
        if self.domain_wise:
            x = rearrange(x, 'b ... (g c) -> b g (... c)', g=self.num_groups)
        else:
            x = rearrange(x, 'b ... (g c) -> b ... g c', g=self.num_groups)

        x = (x - x.mean(dim=-1, keepdim=True)) / (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        if self.domain_wise:
            x = rearrange(x, 'b g (... c) -> b ... (g c)',
                          g=self.num_groups)
        else:
            x = rearrange(x, 'b ... g c -> b ... (g c)',
                          g=self.num_groups)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 channel_first: bool = False):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.channel_first = channel_first

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        if not self.channel_first:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            return x * torch.rsqrt(x.permute(0, 2, 3, 1).pow(2).mean(-1).unsqueeze(1) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if self.channel_first:
            return output * self.weight.view(1, -1, 1, 1)
        else:
            return output * self.weight


def get_activation(activation: str, **kwargs):
    if activation == "gelu":
        activation_fn = nn.GELU()
    elif activation == "relu":
        activation_fn = nn.ReLU()
    elif activation == "tanh":
        activation_fn = nn.Tanh()
    elif activation == "silu":
        activation_fn = nn.SiLU()
    else:
        raise ValueError("Please specify different activation function")
    return activation_fn


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


def unpatchify(x, patch_size, channels, h=None, w=None, t=1):
    """
    x: (N, S, patch_size**2 * C), s is the number of patches
    out: (N, H, W, C), H = W = sqrt(S)*patch_size
    """
    b = x.shape[0]
    p = patch_size
    c = channels
    s = x.shape[1]

    if h is not None or w is not None:
        if h is None:
            h = s // w
        elif w is None:
            w = s // h
    else: # square image
        h = w = int((s//t) ** 0.5)
    assert h * w == s//t
    if t == 1:
        out = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=h, w=w, p1=p, p2=p, c=c)
        return out
    else:
        out = rearrange(x, 'b (t h w) (p1 p2 p3 c) -> b (t p1) c (h p2) (w p3)',
                        h=h, w=w, p1=p, p2=p, p3=p, c=c, t=t)
        return out


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    # basically a layer norm with out bias
    def __init__(self, dim, force_fp32=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
        self.force_fp32 = force_fp32

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=False):
            if self.force_fp32:
                x = x.float()
            x = F.layer_norm(x, [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


class MLP(nn.Module):
    def __init__(self, dim,
                 out_dim=None,
                 expansion_ratio=4, dropout=0.):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.fc1 = nn.Linear(dim, int(dim * expansion_ratio))
        self.fc2 = nn.Linear(int(dim*expansion_ratio), out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.gelu(x, approximate='tanh')
        return self.fc2(x)