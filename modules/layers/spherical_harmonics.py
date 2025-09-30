from math import pi, sqrt
from functools import reduce
from operator import mul
import torch
from functools import wraps, lru_cache
import math 
# caching functions

def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

CACHE = {}


def clear_spherical_harmonics_cache():
    CACHE.clear()


def lpmv_cache_key_fn(l, m, x):
    return (l, m)


####################### Spherical Harmonics utilities ########################
# Code copied from https://github.com/BachiLi/redner/blob/master/pyredner/utils.py
# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def SH_renormalization(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / \
        (4 * math.pi * math.factorial(l + m)))

def SH(l, m, theta, phi):
    if m == 0:
        return SH_renormalization(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    elif m > 0:
        return math.sqrt(2.0) * SH_renormalization(l, m) * \
            torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * SH_renormalization(l, -m) * \
            torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))


@lru_cache(maxsize=1000)
def semifactorial(x):
    return reduce(mul, range(x, 1, -2), 1.)


@lru_cache(maxsize=1000)
def pochhammer(x, k):
    return reduce(mul, range(x + 1, x + k), float(x))


def negative_lpmv(l, m, y):
    if m < 0:
        y *= ((-1) ** m / pochhammer(l + m + 1, -2 * m))
    return y


@cache(cache=CACHE, key_fn=lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    # Check memoized versions
    m_abs = abs(m)

    if m_abs > l:
        return None

    if l == 0:
        return torch.ones_like(x)

    # Check if on boundary else recurse solution down to boundary
    if m_abs == l:
        # Compute P_m^m
        y = (-1) ** m_abs * semifactorial(2 * m_abs - 1)
        y *= torch.pow(1 - x * x, m_abs / 2)
        return negative_lpmv(l, m, y)

    # Recursively precompute lower degree harmonics
    lpmv(l - 1, m, x)

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    # Inplace speedup
    y = ((2 * l - 1) / (l - m_abs)) * x * lpmv(l - 1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l + m_abs - 1) / (l - m_abs)) * CACHE[(l - 2, m_abs)]

    if m < 0:
        y = negative_lpmv(l, m, y)
    return y


def get_spherical_harmonics_element(l, m, theta, phi):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle    [nlat,]
        phi: longitude or azimuth            [nlon,]
    Returns:
        tensor of shape theta
    """
    m_abs = abs(m)
    assert m_abs <= l, "absolute value of order m must be <= degree l"

    N = sqrt((2 * l + 1) / (4 * pi))
    leg = lpmv(l, m_abs, torch.cos(theta))  # [nlat,]

    if m == 0:
        return N * leg.unsqueeze(1).expand(-1, phi.shape[0])

    if m > 0:
        Y = torch.cos(m * phi)
    else:
        Y = torch.sin(m_abs * phi)

    Y = Y.unsqueeze(0)
    leg = leg.unsqueeze(1)
    Y = Y*leg
    N *= sqrt(2. / pochhammer(l - m_abs + 1, 2 * m_abs))
    Y *= N
    return Y

@torch.no_grad()
def get_spherical_harmonics(l, theta, phi):
    """ Tesseral harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    return torch.stack([get_spherical_harmonics_element(l, m, theta, phi) \
                        for m in range(-l, l + 1)],
                       dim=-1)


class SphericalHarmonicsPE(torch.nn.Module):
    def __init__(self, l_max, dim, out_dim, use_mlp=True):
        super().__init__()
        self.l_max = l_max
        self.dim = dim
        self.out_dim = out_dim

        # each l has 2l+1 basis, in total 1 + 3 + 5 + ... + (2l_max+1) = (l_max+1)^2
        if use_mlp:
            basis_weight = torch.randn((self.l_max + 1) ** 2, self.dim) / (self.l_max + 1)
            self.basis_weight = torch.nn.Parameter(basis_weight, requires_grad=True)
            self.pe_mlp = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(self.dim, self.dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.dim, self.out_dim),
            )
        else:
            basis_weight = torch.randn((self.l_max + 1) ** 2, self.out_dim) / (self.l_max + 1)
            self.basis_weight = torch.nn.Parameter(basis_weight, requires_grad=True)
            self.pe_mlp = torch.nn.Identity()

    def sph(self, lat, lon, latlon=None):   
        Y = []
        for l in range(self.l_max + 1):
            Y.append(get_spherical_harmonics(l, lat, lon))
        return Y

    def cache_precomputed_sph_harmonics(self, lat, lon, latlon=None):
        # lat: nlat
        # lon: nlon
        # create meshgrid first
       #  lat, lon = torch.meshgrid([lat, lon], indexing='ij')  # (nlat, nlon)

        sph_lst = self.sph(lat, lon, latlon)  # lst of (nlat, nlon, 2l+1)
        sph_harmonics = torch.cat(sph_lst, dim=-1)  # (nlat, nlon, (l_max+1)^2)
        self.register_buffer('sph_harmonics', sph_harmonics, persistent=False)
        clear_spherical_harmonics_cache()   # clear cache so that it wont affect other computations

    def forward(self, lat, lon, latlon=None):
        if not hasattr(self, 'sph_harmonics'):
            self.cache_precomputed_sph_harmonics(lat, lon, latlon)
        sph_feat = self.sph_harmonics.detach().clone()  # (nlat, nlon, (l_max+1)^2) or (nface, nside, nside, (l_max+1)^2)
        sph_feat = torch.einsum('ijd,dc->ijc', sph_feat, self.basis_weight)  # (nlat, nlon, dim)
        sph_feat = self.pe_mlp(sph_feat)  # (nlat, nlon, dim) or (nface, nside, nside, dim)
        return sph_feat
