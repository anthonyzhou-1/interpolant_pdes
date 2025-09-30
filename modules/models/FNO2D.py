import torch
from torch import nn
from modules.layers.embedding import FourierEmb

def batchmul2d(input, weights, emb=None):
    '''
    args:
        input: (batch, c_in, modes1, modes2)
        weights: (c_in, c_out, modes1, modes2)
        emb: (batch, modes1, modes2)
    '''
    if emb is not None:
        temp = input * emb.unsqueeze(1)
    else:
        temp = input

    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    out = torch.einsum("bixy,ioxy->boxy", temp, weights)
    return out

class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 4 * modes1 * modes2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 4 * modes1 * modes2, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h)


class SpectralConv2d_cond(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        @author: Zongyi Li
        [paper](https://arxiv.org/pdf/2010.08895.pdf)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        if cond_channels > 0:
            self.cond_emb = FreqLinear(cond_channels, self.modes1, self.modes2)

    def forward(self, x, emb=None):
        if emb is not None:
            emb12 = self.cond_emb(emb)
            # emb12 has shape (batch, modes1, modes2, 2)
            emb1 = emb12[..., 0]
            emb2 = emb12[..., 1]
        else:
            emb1 = None 
            emb2 = None
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1, emb1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2, emb2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FourierBasicBlock(nn.Module):

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation = nn.GELU()

        self.fourier1 = SpectralConv2d_cond(in_planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv2d_cond(planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        if cond_channels > 0:
            self.cond_emb = nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None):
        # x has shape [b, c, x, y]
        # emb has shape [b, cond_channels]
        x1 = self.fourier1(x, emb)
        x2 = self.conv1(x)
        if emb is not None:
            emb_out = self.cond_emb(emb)
            while len(emb_out.shape) < len(x2.shape):
                emb_out = emb_out[..., None]
            out = self.activation(x1 + x2 + emb_out)
        else:
            out = self.activation(x1 + x2)
        x1 = self.fourier2(out, emb)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out

class FNO2d(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 modes1: int = 6,
                 modes2: int = 6,
                 hidden_channels: int = 32,
                 cond_channels: int = 32,
                 cond_dim: int = 1,
                 num_layers: int = 4,) -> None:
        super(FNO2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            modes1 (int): number of Fourier modes in x direction
            modes2 (int): number of Fourier modes in y direction
            hidden_channels (int): number of hidden channels
            cond_channels (int): number of condition channels
            cond_dim (int): dimension of condition vector
            num_layers (int): number of Fourier layers
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.cond_dim = cond_dim

        self.conv_in1 = nn.Conv2d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.hidden_channels,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                FourierBasicBlock(self.hidden_channels, 
                                  self.hidden_channels, 
                                  self.cond_channels, 
                                  modes1=self.modes1, 
                                  modes2=self.modes2)
                for i in range(num_layers)
            ]
        )

        self.activation = nn.GELU()

        if self.cond_dim > 0:
            self.cond_embed = FourierEmb(hidden_dim=self.cond_channels, in_dim=self.cond_dim)
            self.use_cond = True 
        else:
            self.use_cond = False 

    def forward(self, 
                x: torch.Tensor,
                emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape [batch, x, y, channels]
            c (torch.Tensor): condition tensor of shape [batch, cond_dim] 
        Returns: torch.Tensor: output has the shape [batch, x, y, channels]
        """
        x = x.permute(0, 3, 1, 2) # (batch, x, y, c) -> (batch, c, x, y)

        if self.use_cond:
            emb = self.cond_embed(emb) # (batch, cond_dim) -> (batch, cond_channels)

        x = self.activation(self.conv_in1(x)) # (batch, c, x, y) -> (batch, width, x, y)
        x = self.activation(self.conv_in2(x)) # (batch, width, x, y) -> (batch, width, x, y)

        for layer in self.layers:
            x = layer(x, emb) # (batch, width, x, y) -> (batch, width, x, y)

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        x = x.permute(0, 2, 3, 1) # (batch, width, x, y) -> (batch, x, y, width)

        return x