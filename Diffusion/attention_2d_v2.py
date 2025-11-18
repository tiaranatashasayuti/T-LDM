from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

#from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def Normalize_wae(in_channels):
    return torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
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
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., chunk_size=128, kv_chunk_size=None, use_flash_attention=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.chunk_size = chunk_size
        self.kv_chunk_size = kv_chunk_size
        self.use_flash_attention = use_flash_attention and hasattr(F, "scaled_dot_product_attention")

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):

        # print(x.shape,context.shape)
        # exit()
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # shape -> [B, H, N, D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        b, _, n, _ = q.shape
        _, _, m, _ = k.shape

        mask_bool = None
        if exists(mask):
            mask_bool = rearrange(mask, 'b ... -> b (...)').to(dtype=torch.bool)

        flash_mask = None
        if mask_bool is not None:
            # scaled_dot_product_attention treats True as "mask out", so invert here
            flash_mask = (~mask_bool).unsqueeze(1).unsqueeze(2)

        use_flash = self.use_flash_attention and q.is_cuda

        if use_flash:
            # Enable flash/memory efficient kernels when possible
            sdp_context = torch.backends.cuda.sdp_kernel(enable_flash=True,
                                                         enable_mem_efficient=True,
                                                         enable_math=False)
            try:
                with sdp_context:
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=flash_mask,
                        dropout_p=0.0,
                        is_causal=False
                    )
            except RuntimeError:
                use_flash = False
        if not use_flash:
            # reshape to merge batch and head for chunked computation
            q_flat = rearrange(q, 'b h n d -> (b h) n d')
            k_flat = rearrange(k, 'b h m d -> (b h) m d')
            v_flat = rearrange(v, 'b h m d -> (b h) m d')

            mask_flat = None
            if mask_bool is not None:
                mask_flat = repeat(mask_bool, 'b j -> (b h) j', h=h)
                mask_flat = mask_flat.unsqueeze(1)

            chunk_size = max(1, min(self.chunk_size or n, n))
            kv_chunk = max(1, min(self.kv_chunk_size or self.chunk_size or m, m))
            max_neg_value = -torch.finfo(torch.float32).max
            out_chunks = []
            for q_chunk in q_flat.split(chunk_size, dim=1):
                # streaming softmax over key chunks to avoid materialising large tensors
                dtype = q_chunk.dtype
                o_chunk = torch.zeros(q_chunk.shape[0], q_chunk.shape[1], v_flat.shape[-1], device=q_chunk.device, dtype=torch.float32)
                lse = torch.zeros(q_chunk.shape[0], q_chunk.shape[1], 1, device=q_chunk.device, dtype=torch.float32)
                m_chunk = torch.full(q_chunk.shape[:2] + (1,), -torch.inf, device=q_chunk.device, dtype=torch.float32)
                q_chunk = q_chunk.float()

                for start in range(0, m, kv_chunk):
                    end = start + kv_chunk
                    k_chunk = k_flat[:, start:end, :].float()
                    v_chunk = v_flat[:, start:end, :]
                    scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * self.scale
                    mask_chunk = None
                    if mask_flat is not None:
                        mask_chunk = mask_flat[:, :, start:end]
                        scores = scores.masked_fill(~mask_chunk, max_neg_value)

                    current_max = scores.max(dim=-1, keepdim=True).values
                    m_next = torch.maximum(m_chunk, current_max)

                    exp_scores = torch.exp(scores - m_next)
                    if mask_chunk is not None:
                        exp_scores = exp_scores * mask_chunk
                    exp_prev = torch.exp(m_chunk - m_next)

                    lse = exp_prev * lse + exp_scores.sum(dim=-1, keepdim=True)
                    o_chunk = exp_prev * o_chunk + torch.matmul(exp_scores, v_chunk.float())

                    m_chunk = m_next

                o_chunk = (o_chunk / lse.clamp_min(1e-6)).to(dtype)
                out_chunks.append(o_chunk)
            out = torch.cat(out_chunks, dim=1)
            out = rearrange(out, '(b h) n d -> b h n d', b=b, h=h)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 use_self_attn=True, use_cross_attn=True
                 ):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        #return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint
        #print('context in basic trans:',context.shape)
        return self._forward(x,context)

    def _forward(self, x, context=None):
        #print('context in basic trans in _forward:',context.shape)
        if self.use_self_attn:
            x = self.attn1(self.norm1(x)) + x
        #print('---cross attention---')
        if self.use_cross_attn:
            x = self.attn2(self.norm2(x), context=context) + x
        #print('---end of cross attention---')
        x = self.ff(self.norm3(x)) + x
        return x

class ModulatedPrompts(nn.Module):
    def __init__(self, n_tokens=32, d_model=256):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2*n_tokens*d_model),
            nn.GELU(),
            nn.Linear(2*n_tokens*d_model, 2*n_tokens*d_model),
        )
        self.n_tokens, self.d_model = n_tokens, d_model

    def forward(self, ctx):                     # [B,1,256] or [B,256]
        if ctx.dim() == 3: ctx = ctx.squeeze(1)
        scale_shift = self.mlp(ctx).view(ctx.size(0), self.n_tokens, 2*self.d_model)
        scale, shift = scale_shift.chunk(2, dim=-1)
        #return scale * self.prompts + shift      # [B, 32, 256]
        return scale * (self.prompts + self.pos) + shift




class SpatialTransformer(nn.Module):
    
    # Transformer block for image-like data.
    # First, project the input (aka embedding)
    # and reshape to b, t, d.
    # Then apply standard transformer action.
    # Finally, reshape to image
    
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_self_attn=True,use_cross_attn=True, context_type = None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.context_type = context_type

        self.proj_in = nn.Conv2d(in_channels,
                                  inner_dim,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

        #If context dimension is None, we set default as 256 for unconstraint generation
        if context_dim is None:
            context_dim = 256
            self.unconstrained = True
        else:
            self.unconstrained = False

        n_tokens = 32
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,use_self_attn=use_self_attn,use_cross_attn=use_cross_attn)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                               padding=0))
        
        #add modulated prompt, remove if it screws up the model
        if not self.unconstrained:
            self.modulated_prompt = ModulatedPrompts(n_tokens= n_tokens, d_model=context_dim)
        else:
            print('NOTE: Unconstrained generation, no context provided, using learned null vector as context')
            self.E_null = nn.Parameter(torch.randn(n_tokens, context_dim)) 

        if self.context_type =='iscasp':
            print('NOTE:iscassp is being used, model will directly load context into cross attention')

 
    def forward(self, x, context=None):

        #remove if it jepoardise the model
        if not self.unconstrained:
            context = self.modulated_prompt(context)
        elif not self.unconstrained and self.context_type =='icassp':
            context = context
        else:
           #unconstrainted generation, set context to a learned null vector
           context = self.E_null.expand(x.size(0), -1, -1)
            #print('context set to :',context.shape)


        # note: if no context is given, cross-attention defaults to self-attention
        #print('context:',context.shape)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        #print("spatial transform before:",x.shape)
        x = rearrange(x, 'b c h w -> b (h w) c')
        #print("spatial transform:",x.shape)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)

        # 1D case
        #b, c, h = x.shape
        #x_in = x 
        #x = self.norm(x)
        #x = self.proj_in(x)
        #x = rearrange(x, 'b c h -> b (h) c')
        #for block in self.transformer_blocks:
        #    x = block(x, context=context)
        #x = rearrange(x, 'b (h) c -> b c h', h=h)
        #x = self.proj_out(x)
        return x + x_in
    

class SpatialTransformer_modified_for_wae(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize_wae(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                  inner_dim,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)


        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                               padding=0))
 
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        #print("spatial transform before:",x.shape)
        x = rearrange(x, 'b c h w -> b (h w) c')
        #print("spatial transform:",x.shape)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)

        # 1D case
        #b, c, h = x.shape
        #x_in = x 
        #x = self.norm(x)
        #x = self.proj_in(x)
        #x = rearrange(x, 'b c h -> b (h) c')
        #for block in self.transformer_blocks:
        #    x = block(x, context=context)
        #x = rearrange(x, 'b (h) c -> b c h', h=h)
        #x = self.proj_out(x)
        return x + x_in
    
