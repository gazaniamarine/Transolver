"""
Transolver model for 2D irregular-mesh stator electromagnetics.

Input:  (x, y) coordinates  → space_dim = 2
Output: Az (scalar)         → out_dim   = 1

Physics Attention (Irregular Mesh) is used because every stator diameter
produces a different unstructured mesh.  The slice mechanism groups mesh
nodes into a learned partition that acts as physics-informed latent tokens.
"""

import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------
ACTIVATION = {
    'gelu':      nn.GELU,
    'tanh':      nn.Tanh,
    'sigmoid':   nn.Sigmoid,
    'relu':      nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus':  nn.Softplus,
    'ELU':       nn.ELU,
    'silu':      nn.SiLU,
}


# ---------------------------------------------------------------------------
# Physics Attention — Irregular Mesh
# ---------------------------------------------------------------------------
class Physics_Attention_Irregular_Mesh(nn.Module):
    """
    Three-stage attention designed for irregular / unstructured meshes:

      (1) Slice  – soft-partition N mesh nodes into G latent 'physics tokens'
                   via learned, normalised attention weights.
      (2) Attend – run standard self-attention over the G (≪ N) tokens.
      (3) Deslice – broadcast token information back to every mesh node.

    This gives O(N·G + G²) complexity instead of O(N²), while letting the
    model discover physics-meaningful groupings (e.g. air-gap, teeth, yoke).
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0., slice_num: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head  = dim_head
        self.heads     = heads
        self.scale     = dim_head ** -0.5
        self.softmax   = nn.Softmax(dim=-1)
        self.dropout   = nn.Dropout(dropout)

        # Learnable temperature controls sharpness of slice assignment.
        # Initialised to 0.5 — kept positive through clamping in forward().
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # Two separate projections: one for position (x), one for field (fx)
        self.in_project_x     = nn.Linear(dim, inner_dim)
        self.in_project_fx    = nn.Linear(dim, inner_dim)

        # Maps each node's head-dimension to G slice logits.
        # Orthogonal init encourages diverse, non-redundant slices.
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        # Standard QKV for attention among slice tokens
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)  — batch of N mesh-node embeddings of width C
        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape

        # ------------------------------------------------------------------ #
        # (1) Slice: aggregate N nodes → G physics tokens
        # ------------------------------------------------------------------ #
        # Project to (B, H, N, D) for both field and position streams
        fx_mid = self.in_project_fx(x) \
                     .reshape(B, N, self.heads, self.dim_head) \
                     .permute(0, 2, 1, 3).contiguous()   # B H N D

        x_mid  = self.in_project_x(x) \
                     .reshape(B, N, self.heads, self.dim_head) \
                     .permute(0, 2, 1, 3).contiguous()   # B H N D

        # Soft slice assignment weights: (B, H, N, G)
        # Temperature is clamped so training stays stable
        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / temperature)  # B H N G

        # Normalise by the total weight each slice receives across all nodes
        slice_norm  = slice_weights.sum(2)                # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc",
                                   fx_mid, slice_weights) # B H G D
        slice_token = slice_token / (
            (slice_norm + 1e-5)[..., None].expand_as(slice_token))

        # ------------------------------------------------------------------ #
        # (2) Attend: standard scaled dot-product over G tokens
        # ------------------------------------------------------------------ #
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # B H G G
        attn = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attn, v)                    # B H G D

        # ------------------------------------------------------------------ #
        # (3) Deslice: broadcast G tokens back to N mesh nodes
        # ------------------------------------------------------------------ #
        out_x = torch.einsum("bhgc,bhng->bhnc",
                              out_slice_token, slice_weights)  # B H N D
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')       # B N (H*D)
        return self.to_out(out_x)                               # B N C


# ---------------------------------------------------------------------------
# Feed-forward MLP (with optional residual)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int,
                 n_layers: int = 1, act: str = 'gelu', res: bool = True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Activation '{act}' not supported.")
        Act = ACTIVATION[act]
        self.res        = res
        self.n_layers   = n_layers
        self.linear_pre  = nn.Sequential(nn.Linear(n_input,  n_hidden), Act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears     = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), Act())
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


# ---------------------------------------------------------------------------
# Single Transolver encoder block
# ---------------------------------------------------------------------------
class Transolver_block(nn.Module):
    """
    Pre-norm transformer block:
        x ← x + PhysicsAttn(LayerNorm(x))
        x ← x + FFN(LayerNorm(x))
    Last block additionally applies a linear head to produce the output field.
    """

    def __init__(self, num_heads: int, hidden_dim: int, dropout: float,
                 act: str = 'gelu', mlp_ratio: int = 4,
                 last_layer: bool = False, out_dim: int = 1,
                 slice_num: int = 32):
        super().__init__()
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn  = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads     = num_heads,
            dim_head  = hidden_dim // num_heads,
            dropout   = dropout,
            slice_num = slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp  = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)

        if self.last_layer:
            self.ln_3  = nn.LayerNorm(hidden_dim)
            self.mlp2  = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.Attn(self.ln_1(fx)) + fx     # residual attention
        fx = self.mlp(self.ln_2(fx))   + fx     # residual FFN
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


# ---------------------------------------------------------------------------
# Full Transolver model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """
    Transolver for 2D irregular-mesh stator electromagnetics.

    Architecture
    ------------
    Preprocess MLP : (space_dim) → n_hidden     [embeds (x,y) coordinates]
    N × Transolver blocks with Physics Attention (Irregular Mesh)
    Output head    : n_hidden → out_dim          [predicts Az scalar]

    Parameters
    ----------
    space_dim  : dimensionality of the input coordinates (2 for x,y)
    n_layers   : number of Transolver blocks
    n_hidden   : hidden feature width
    n_head     : number of attention heads (must divide n_hidden)
    dropout    : dropout probability
    act        : activation function name
    mlp_ratio  : FFN hidden / hidden_dim ratio
    fun_dim    : extra function / forcing dimension concatenated to coords
    out_dim    : output field dimension (1 for scalar Az)
    slice_num  : number of physics latent tokens (G)
    """

    def __init__(self,
                 space_dim:  int   = 2,
                 n_layers:   int   = 6,
                 n_hidden:   int   = 256,
                 dropout:    float = 0.0,
                 n_head:     int   = 8,
                 act:        str   = 'gelu',
                 mlp_ratio:  int   = 2,
                 fun_dim:    int   = 0,
                 out_dim:    int   = 1,
                 slice_num:  int   = 32):
        super().__init__()
        self.__name__ = 'Transolver_Stator'

        # Input embedding: coords (+ optional extra features) → hidden
        in_dim = space_dim + fun_dim
        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden,
                              n_layers=0, res=False, act=act)

        self.n_hidden  = n_hidden
        self.space_dim = space_dim

        # Stack of Transolver blocks; only the last one has the output head
        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads  = n_head,
                hidden_dim = n_hidden,
                dropout    = dropout,
                act        = act,
                mlp_ratio  = mlp_ratio,
                out_dim    = out_dim,
                slice_num  = slice_num,
                last_layer = (i == n_layers - 1),
            )
            for i in range(n_layers)
        ])

        # Learned placeholder added to every embedded node (acts as a bias
        # that the model can freely tune to capture global field offset)
        self.placeholder = nn.Parameter(
            (1.0 / n_hidden) * torch.rand(n_hidden))

        self._initialize_weights()

    # ------------------------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, space_dim)  batch of node coordinate tensors
        Returns:
            out : (B, N, out_dim)  predicted Az field at every node
        """
        # Embed coordinates → hidden dimension
        fx = self.preprocess(x)                           # B N C
        fx = fx + self.placeholder[None, None, :]         # learnable offset

        # Pass through Transolver blocks
        for block in self.blocks:
            fx = block(fx)

        # fx is (B, N, out_dim) after the last block's linear head
        return fx
