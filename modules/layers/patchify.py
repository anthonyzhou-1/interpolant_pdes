import torch.nn as nn 
from einops import rearrange

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=[2, 2],
            in_chans=3,
            hidden_size=768,
            norm_layer=None,
            flatten=True,
            bias=True,
            ndim=2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        embed_dim = hidden_size
        self.ndim = ndim
        if ndim == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        elif ndim == 3:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_params(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        if self.ndim == 2:
            x = rearrange(x, 'b ny nx c -> b c ny nx')
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            else:
                x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
            x = self.norm(x)
            return x 
        elif self.ndim == 3:
            x = rearrange(x, 'b nx ny nz c -> b c nx ny nz')
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
            else:
                x = x.permute(0, 2, 3, 4, 1)  # BCHWD -> BHWDC
            x = self.norm(x)
            return x
