import torch.nn as nn 
from einops import rearrange
from modules.layers.basics import MLP
from torch import einsum

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
     
class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout = 0.0,
            act='gelu',
            mlp_ratio=4,
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.Attn = CrossAttention(hidden_dim, hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                dropout=dropout) # assume query and context dim are the same
            
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, expansion_ratio=mlp_ratio)

    def forward(self, q, kv):
        fx = self.Attn(self.ln_q(q), self.ln_kv(kv)) + q
        fx = self.mlp(self.ln_2(fx)) + fx

        return fx