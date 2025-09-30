import torch
import torch.nn as nn
from modules.layers.embedding import FourierEmb

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FourierBasicBlock(nn.Module):

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int = 0, 
        modes1: int = 16,
        modes2: int = 16,
        modes3: int = 16,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.activation = nn.GELU()

        self.fourier1 = SpectralConv3d(in_planes, planes, modes1=self.modes1, modes2=self.modes2, modes3=self.modes3)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv3d(planes, planes, modes1=self.modes1, modes2=self.modes2, modes3=self.modes3)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        if cond_channels > 0:
            self.cond_emb = nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None):
        # x has shape [b, c, x, y, z]
        # emb has shape [b, cond_channels]
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        if emb is not None:
            emb_out = self.cond_emb(emb)
            while len(emb_out.shape) < len(x2.shape):
                emb_out = emb_out[..., None]
            out = self.activation(x1 + x2 + emb_out)
        else:
            out = self.activation(x1 + x2)
        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out


class FNO3d(nn.Module):
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 modes1: int = 6,
                 modes2: int = 6,
                 modes3: int = 6,
                 hidden_channels: int = 32,
                 cond_channels: int = 32,
                 cond_dim: int = 1,
                 num_layers: int = 4,):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y, z), ..., u(10, x, y, z),  x, y, z, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.hidden_channels = hidden_channels
        self.cond_channels = cond_channels
        self.cond_dim = cond_dim

        self.conv_in1 = nn.Conv3d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv3d(
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
                                  modes2=self.modes2,
                                  modes3=self.modes3,)
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
            x (torch.Tensor): input tensor of shape [batch, x, y, z, channels]
            c (torch.Tensor): condition tensor of shape [batch, cond_dim] 
        Returns: torch.Tensor: output has the shape [batch, x, y, z, channels]
        """
        x = x.permute(0, 4, 1, 2, 3) # (batch, x, y, z, c) -> (batch, c, x, y, z)

        if self.use_cond:
            emb = self.cond_embed(emb) # (batch, cond_dim) -> (batch, cond_channels)

        x = self.activation(self.conv_in1(x)) # (batch, c, x, y, z) -> (batch, width, x, y, z)
        x = self.activation(self.conv_in2(x)) # (batch, width, x, y, z) -> (batch, width, x, y, z)

        for layer in self.layers:
            x = layer(x, emb) # (batch, width, x, y, z) -> (batch, width, x, y, z)

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        x = x.permute(0, 2, 3, 4, 1) # (batch, width, x, y, z) -> (batch, x, y, z, width)

        return x