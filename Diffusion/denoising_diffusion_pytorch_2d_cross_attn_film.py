import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import pickle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__
#from attention_2d import SpatialTransformer
from attention_2d_v2 import SpatialTransformer
from FiLM import FiLM2DBlock
import wandb

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image
#
# data
def encode_sequence_OE(seq):
    # Character to index mapping
    char_to_index = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'U': 3,  # 'E' and 'U' share the same index
        'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
        'L': 9, 'M': 10, 'O': 10,  # 'M' and 'O' share the same index
        'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18,
        'X': 19, 'B': 19, 'Z': 19,  # 'X', 'B', and 'Z' share the same index
        'Y': 20
    }
    encoded_sequence = [char_to_index.get(seq[i], 0) * 50 + i for i in range(len(seq))]
 
    return encoded_sequence




def decode_embeddings_to_sequence(embeddings, tokenizer):
    """
    Decode a batch of embeddings to sequences using the provided tokenizer.
    
    Args:
        embeddings (torch.Tensor): The embeddings to decode.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for decoding.
    
    Returns:
        List[str]: The decoded sequences.
    """
    decoded_sequences = []
    for embedding in embeddings:
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        
        token_ids = torch.argmax(embedding, dim=-1).tolist()  # Get token indices
        decoded_seq = tokenizer.decode(token_ids, skip_special_tokens=True)
        decoded_sequences.append(decoded_seq)
    
    return decoded_sequences

def encode_sequence_OPE(seq):
    # Character to index mapping
    char_to_num = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'U': 4,  # 'E' and 'U' share the same index
        'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
        'L': 10, 'M': 11, 'O': 11,  # 'M' and 'O' share the same index
        'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19,
        'X': 20, 'B': 20, 'Z': 20,  # 'X', 'B', and 'Z' share the same index
        'Y': 21
    }
    encoded_sequence = np.array([char_to_num.get(char, 0) for char in seq])  # Default to 0 if char not found
 
    return encoded_sequence

def pad_sequence(s, maxlen=64, padding='post', value=0):
    """ Pad sequence to the same length """
    x = np.full((maxlen), value, dtype=np.int32)
    if len(s) > maxlen:
        x = s[:maxlen]
    if padding == 'post':
        x[:len(s)] = s
    elif padding == 'pre':
        x[-len(s):] = s
    return x

def preprocess(data, encode_choice):
    assert (encode_choice == "OE" or encode_choice == "OPE")
    features = []
    encoded_sequences = []
    labels = []
    for idx in range(len(data)):
        row = data.iloc[idx, :]  
        if encode_choice == "OE":
            encoded_seq = encode_sequence_OE(row['sequence'])
        elif encode_choice == "OPE":
            encoded_seq = encode_sequence_OPE(row['sequence'])
       
        #encoded_seq = encoded_seq/21
           
        padded_encoded_seq = pad_sequence(encoded_seq)
        features.append(padded_encoded_seq)
        encoded_sequences.append(encoded_seq)
        labels.append(row['is_acp'])
       
    return np.array(features), encoded_sequences, np.array(labels)

def load_features(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Return the numpy dtype that best matches the provided torch dtype."""
    return torch.empty((), dtype=dtype).numpy().dtype


class BioDataset_modified(Dataset):
    def __init__(
        self,
        sequences,
        features=None,
        raw_sequences=None,
        sequence_dtype: torch.dtype = torch.float32,
        feature_dtype: torch.dtype = torch.float32,
    ):
        """Dataset that keeps numpy arrays lazy and avoids redundant copies."""

        self.sequence_dtype = sequence_dtype
        self.feature_dtype = feature_dtype

        self.sequences = self._prepare_array(sequences, target_dtype=self.sequence_dtype)
        self.features = None if features is None else self._prepare_array(features, target_dtype=self.feature_dtype)
        self.raw_sequences = raw_sequences

    @staticmethod
    def _prepare_array(array_like, target_dtype: torch.dtype):
        if isinstance(array_like, torch.Tensor):
            return array_like.detach().cpu().to(dtype=target_dtype)

        np_array = np.asarray(array_like)
        target_np_dtype = _torch_dtype_to_numpy(target_dtype)
        if np_array.dtype != target_np_dtype:
            np_array = np_array.astype(target_np_dtype, copy=False)
        return np_array

    def __len__(self):
        return len(self.sequences)

    def _get_sequence(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        seq_tensor = torch.as_tensor(seq, dtype=self.sequence_dtype)

        if seq_tensor.ndim == 3:
            seq_tensor = seq_tensor.squeeze(0)

        if seq_tensor.ndim == 2:
            seq_tensor = seq_tensor.permute(1, 0)
        elif seq_tensor.ndim == 1:
            seq_tensor = seq_tensor.unsqueeze(0)

        return seq_tensor.contiguous()

    def _get_feature(self, idx: int):
        if self.features is None:
            return None

        feature = self.features[idx]
        feature_tensor = torch.as_tensor(feature, dtype=self.feature_dtype)
        if feature_tensor.ndim == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        return feature_tensor.contiguous()

    def __getitem__(self, idx):
        sequence_tensor = self._get_sequence(idx)
        feature_tensor = self._get_feature(idx)

        if feature_tensor is not None and self.raw_sequences is not None:
            return sequence_tensor, feature_tensor, self.raw_sequences[idx]
        if feature_tensor is not None:
            return sequence_tensor, feature_tensor
        return sequence_tensor


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)
        #print('memkv shape:',torch.randn(2, heads, num_mem_kv, dim_head).shape)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x,context= None):
        b, c, h, w = x.shape
    
        x = self.norm(x)
        #print(x.shape)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        #print(qkv[0].shape,qkv[1].shape,qkv[2].shape)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)
        #print(q.shape,k.shape,v.shape)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        #print(mk.shape,mv.shape)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))
        #print(k.shape,v.shape)

        out = self.attend(q, k, v)
        #print(out.shape)
        #exit()
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
    

class Cross_Attention_custom(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        context_dim = 0,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)
        
        # print('memkv shape:',torch.randn(2, heads, num_mem_kv, dim_head).shape)

        #hardcoding for now
        h = 16
        w = 8
        #print('ca stuff:',dim,hidden_dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_q = nn.Conv2d(dim, hidden_dim , 1, bias = False)
        self.to_k1 = nn.Linear(context_dim,  h * w, bias=False)
        self.to_k2= nn.Conv2d(1, hidden_dim , 1, bias = False)
        self.to_v1 = nn.Linear(context_dim,  h * w, bias=False)
        self.to_v2= nn.Conv2d(1, hidden_dim , 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x,context= None):
        #convert 1D context to 2D
        #context = context.unsqueeze(1)
        b, c, h, w = x.shape

        x = self.norm(x)

        #qkv = self.to_qkv(x).chunk(3, dim = 1)
        #print(x.shape)
        q = self.to_q(x)
        #print(q.shape)


        k = self.to_k1(context)
        k = rearrange(k, 'b c (h w) -> b c h w',h=h)
        k = self.to_k2(k)
   



        v = self.to_v1(context)
        v = rearrange(v, 'b c (h w) -> b c h w',h=h)
        v = self.to_v2(v)
        

        #print('qkv before:',q.shape,k.shape,v.shape)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), (q,k,v))

        
        #print('qkv:',q.shape,k.shape,v.shape)
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        #print('query dim:',query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        #print(x.shape,context.shape)
        q = self.to_q(x)
        context = default(context, x)
        #print("checking if context is same as input:",context == x)
        k = self.to_k(context)
        v = self.to_v(context)
        #print(q.shape,k.shape,v.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        #print(q.shape,k.shape,v.shape)
   
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)





# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        context_dim = None,
        context_type = 'cross-attention'#options are cross-attention or film or icassp
    ):
        super().__init__()

        # determine dimensions
        self.context_dim = context_dim
        print("Context dim set to:",self.context_dim)
        self.context_type = context_type
        print("Context type set to:",self.context_type)
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        print(attn_dim_head,attn_heads)
       

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        #modified to full spatial attention for cross attention
        FullAttention_only = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)
        cross_attn_use = False

        if context_type =='cross-attention':
            print("If Note: Model using cross attention at all down/up block:",context_type)
            FullAttention_ca =  SpatialTransformer
            cross_attn_use = True

        elif context_type =='icassp':
            print("ElIf Note: Model using cross attention only at at mid block:",context_type)
            FullAttention_ca = SpatialTransformer
            cross_attn_use = False
            
        else:
            print("Else Note: Model using at down block:",context_type)
            FullAttention_ca = FiLM2DBlock
        
        
            

        cross_attn = Cross_Attention_custom

        #cross attn parameter:(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.)
        #query dim is the input dim of x

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)


            

            attn_klass_cross = FullAttention_ca
            #attn_klass_self = FullAttention_only



            #FiLM2DBlock(in_channels=dim_in, n_heads=layer_attn_heads,d_head=layer_attn_dim_head, ctx_dim=context_dim, depth=1)
            #attn_klass_cross(in_channels = dim_in, d_head = layer_attn_dim_head, n_heads = layer_attn_heads,context_dim=context_dim),

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass_cross(in_channels = dim_in, d_head = layer_attn_dim_head, n_heads = layer_attn_heads,context_dim=context_dim,
                                 use_self_attn = is_last, use_cross_attn = cross_attn_use,context_type = self.context_type
                                 ),
                #attn_klass_cross(in_channels = dim_in, d_head = layer_attn_dim_head, n_heads = layer_attn_heads,context_dim=context_dim,
                #                 use_self_attn = False, use_cross_attn = True,context_type = self.context_type
                #                 ),
                #attn_klass_self(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        
        #v3
        #self.mid_attn = cross_attn(dim = mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1],context_dim=context_dim)

        #v2
        self.mid_attn = FullAttention_ca(in_channels = mid_dim, n_heads = attn_heads[-1], d_head = attn_dim_head[-1],context_dim=context_dim, use_self_attn=True, use_cross_attn=True,context_type = self.context_type)
        
        #v1
        #self.mid_attn = FullAttention_only(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            #setup self attention on first up block only
            if ind == 0:
                self_attn_var = True
            else:
                self_attn_var = False

            #attn_klass = FullAttention if layer_full_attn else LinearAttention
            attn_klass = FullAttention_only


            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass_cross(in_channels = dim_out, d_head = layer_attn_dim_head, n_heads = layer_attn_heads,context_dim= context_dim,
                                 use_self_attn= self_attn_var , use_cross_attn = cross_attn_use,context_type = self.context_type),
                #attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None,context = None):
        if self.context_dim is None:
            context = None

        # # debugging
        # print(f"Initial X Shape: {x.shape}")  # Shape before processing
        #print(f"Initial Context Shape: {context.shape}")  # Ensure context isn't None
        
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)

            # Debug: Print before cross-attention
            # print(f"X Shape Before Cross-Attention: {x.shape}")
            # print(f"Context Shape Before Cross-Attention: {context.shape}")

            x = attn(x,context = context) + x
            h.append(x)

            x = downsample(x)
        # Debug: Final output before return
        # print(f"Final Output Shape: {x.shape}")
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x,context = context) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x,context = context) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False,context= None):
        model_output = self.model(x, t, x_self_cond,context = context)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True,context = None):
        preds = self.model_predictions(x, t, x_self_cond,context= context)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None,context = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond,context = context, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False,context = None):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond,context = context)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False,context = None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True,context = context)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False,context = None):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps,context = context)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None,context = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond,context = context)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class
class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        device,
        decoder,
        *,
        train_batch_size=16,
        valid_batch_size=16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        specific_save_steps = None,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        train_num_workers = None,
        valid_num_workers = None,
        train_prefetch_factor = 2,
        valid_prefetch_factor = 2,
        pin_memory = True,
    ):
        super().__init__()

        self.device = device  # Directly use the passed device
        self.decoder = decoder
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.specific_save_steps = set(specific_save_steps) if specific_save_steps else None

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        #assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        
        def _resolve_workers(value, fallback):
            if value is None:
                return fallback
            return max(0, int(value))

        default_workers = max(1, min(8, cpu_count() // 2))
        train_workers = _resolve_workers(train_num_workers, default_workers)
        valid_workers = _resolve_workers(valid_num_workers, max(1, min(4, default_workers)))

        def _dataloader(dataset, *, batch_size, shuffle, num_workers, prefetch_factor):
            loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
            if num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                pf = 2 if prefetch_factor is None else max(1, int(prefetch_factor))
                loader_kwargs["prefetch_factor"] = pf
            return DataLoader(dataset, **loader_kwargs)

        train_dl = _dataloader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_workers,
            prefetch_factor=train_prefetch_factor,
        )
        valid_dl = _dataloader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=valid_workers,
            prefetch_factor=valid_prefetch_factor,
        )

        self.train_dl = cycle(self.accelerator.prepare(train_dl))
        self.valid_dl = self.accelerator.prepare(valid_dl)

        #assert len(self.train_dl) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        #dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        #dl = self.accelerator.prepare(dl)
        #self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        #Tiara edit
        # from torch.optim.lr_scheduler import CosineAnnealingLR
        # self.scheduler = CosineAnnealingLR(self.opt, T_max=self.train_num_steps, eta_min=1e-6)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.train_dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def _prepare_batch(self, batch, device):
        """Move batch tensors to device and normalize context handling."""
        raw_sequences = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                data, context, raw_sequences = batch
            elif len(batch) == 2:
                data, context = batch
            else:
                data = batch[0]
                context = batch[1] if len(batch) > 1 else None
        else:
            data = batch
            context = None

        data = data.to(device, non_blocking=True)

        if isinstance(context, torch.Tensor):
            context = context.to(device, non_blocking=True)
            if context.ndim >= 3 and context.size(1) == 1:
                context = context.squeeze(1)
            if context.ndim >= 3 and context.size(-1) == 1:
                context = context.squeeze(-1)
            if context.numel() == 0:
                context = None
        else:
            context = None

        return data, context, raw_sequences

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone,*args):
        accelerator = self.accelerator
        device = self.device
        
        if args:
            print("Additional path detected. Using path:",args[0])
            path_dir = args[0]
            data = torch.load(path_dir, map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data.get('step', 0)

        opt_state = data.get('opt')
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        else:
            if self.accelerator.is_main_process:
                print("Warning: optimizer state missing in checkpoint; optimizer reinitialized.")

        ema_state = data.get('ema')
        if ema_state is not None and self.accelerator.is_main_process:
            self.ema.load_state_dict(ema_state)
        elif self.accelerator.is_main_process:
            print("Warning: EMA state missing in checkpoint; EMA reinitialized.")

        if 'version' in data:
            print(f"loading from version {data['version']}")

        scaler_state = data.get('scaler')
        if exists(self.accelerator.scaler) and scaler_state is not None:
            self.accelerator.scaler.load_state_dict(scaler_state)
        elif exists(self.accelerator.scaler) and self.accelerator.is_main_process:
            print("Warning: GradScaler state missing in checkpoint; scaler reinitialized.")

    @torch.no_grad()
    def decode_data(self,data):
                
        #incoming data is [batch_size,1,embedding size,seq length]
        #change it to [batch_size,seq length,embedding size]
        data = data.squeeze(1)
        data = data.permute(0,2,1)
        #print(data.shape)
        data = self.decoder(data)

        data = data.permute(0,2,1)
        data = data.unsqueeze(1)
        
        return data

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.valid_dl)

        with torch.no_grad():
            for batch in self.valid_dl:
                data, context, _ = self._prepare_batch(batch, self.device)

                data = self.decode_data(data)

                with self.accelerator.autocast():
                    loss = self.model(data, context=context)
                    total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def sample(self):
            raise Exception("It should not go through trainer sample!")
            self.model.eval()
            try:
                with torch.no_grad():
                    batches = num_to_groups(self.num_samples, self.train_batch_size)
                    #batches = self.train_batch_size
                    all_samples_list = []
                    for batch_size in batches:
                        batch = next(self.train_dl)
                        _, context, _ = self._prepare_batch(batch, self.device)

                        if context is not None and context.size(0) != batch_size:
                            context = context[:batch_size]

                        print(f"Sampling batch of size: {batch_size}")
                        ctx_shape = tuple(context.shape) if context is not None else 'None'
                        print(f"Context shape: {ctx_shape}")
                        samples = self.ema.ema_model.sample(batch_size=batch_size, context=context)
                        print(f"Samples shape: {samples.shape}")
                        all_samples_list.append(samples)

                    all_samples = torch.cat(all_samples_list, dim=0)
                    #torch.save(all_samples, str(self.results_folder / f'sample-final.png'))
            except Exception as e:
                print("Error during sampling:")
                print(e)


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        # Initialize wandb for the diffusion model
        wandb.init("Diffusion Model Training")

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                total_train_loss = 0.0
                self.opt.zero_grad()  # Zero the gradients at the beginning of each step

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.train_dl)
                    data, context, raw_sequences = self._prepare_batch(batch, device)

                    # Attempt to out a first stage decoder here first to see if it can decode properly
                    data = self.decode_data(data)

                    with self.accelerator.autocast():
                        loss = self.model(data, context=context)
                        

                        total_train_loss += loss.item()

                    self.accelerator.backward(loss)

                # Update the progress bar description with the average train loss
                pbar.set_description(f"Average Train loss: {total_train_loss / self.gradient_accumulate_every:.4f}")

                # Log train loss with wandb
                wandb.log({"Average Train Loss": total_train_loss / self.gradient_accumulate_every}, step=self.step)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                #self.scheduler.step()
                
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()

                    trigger_eval = False
                    trigger_save = False

                    if self.specific_save_steps:
                        if self.step > 0 and self.step in self.specific_save_steps:
                            trigger_eval = True
                            trigger_save = True
                    elif self.step > 0 and self.step % self.save_and_sample_every == 0:
                        trigger_eval = True
                        if self.step % (self.save_and_sample_every * 10) == 0:
                            trigger_save = True

                    if trigger_eval:
                        self.ema.ema_model.eval()
                        valid_loss = self.validate()
                        print(f'Average Validation loss: {valid_loss}')
                        wandb.log({"Average Validation Loss": valid_loss}, step=self.step)

                        if trigger_save:
                            self.save(self.step)

                pbar.update(1)
        #self.save("ACP_DIFF_final_steps:"+str(self.train_num_steps))
        accelerator.print('training complete')
