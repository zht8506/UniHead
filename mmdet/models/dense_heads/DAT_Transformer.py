import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
import numpy as np

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Efficient Dual-axial Attention (EDA)
class EDA_Attention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.idx=idx
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x,H,W,H_sp,W_sp):
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x,func,H,W,H_sp,W_sp):
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        lepe=lepe.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp).reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, q,k,v,H,W):
        """
        x: B L C
        """
        H_sp=W_sp=0
        if self.idx == -1:
            H_sp, W_sp = H, W
        elif self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size
        else:
            print("ERROR MODE", self.idx)
            exit(0)

        # q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        # H = W = self.resolution
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q,H,W,H_sp,W_sp)
        k = self.im2cswin(k,H,W,H_sp,W_sp)
        v, lepe = self.get_lepe(v, self.get_v,H,W,H_sp,W_sp)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

# Dual-axial Aggregation Transformer (DAT)
class DAT(nn.Module):
    def __init__(self, dim, num_heads,
                 split_size=7,reso=64, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, int(dim * 2.5), bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                EDA_Attention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                EDA_Attention(
                    dim//2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B,C,H,W=x.shape

        x=x.view(B,C,-1).permute(0,2,1).contiguous()
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.split_size - W % self.split_size) % self.split_size
        pad_b = (self.split_size - H % self.split_size) % self.split_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, nH, nW, _ = x.shape
        x = x.view(B, nH*nW, C)

        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 5, C//2).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[0],qkv[1],qkv[4],nH,nW)
            x2 = self.attns[1](qkv[2],qkv[3],qkv[4],nH,nW)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.view(B, nH , nW, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        return x.permute(0,3,1,2).contiguous()
