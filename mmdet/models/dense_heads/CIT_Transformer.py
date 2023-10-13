import torch
import torch.nn as nn
import torch.nn.functional as F

#from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath

class PosCNN(nn.Module):
    """
    This is PEG module from https://arxiv.org/abs/2102.10882
    """
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H=None, W=None):
        # B, N, C = x.shape
        # feat_token = x
        # cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.s == 1:
            x = self.proj(x) + x
        else:
            x = self.proj(x)
        # x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., group=-1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group > 0:
            self.fc1 = nn.Conv1d(in_features, hidden_features, 1, groups=group)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if group > 0:
            self.fc2 = nn.Conv1d(hidden_features, out_features, 1, groups=group)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.group = group

    def forward(self, x):
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.group > 0:
            x = x.permute(0, 2, 1).contiguous()
        return x

# Locality Enhancement Block (LEB)
class LEB(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        # padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=1,
                                     padding=0, groups=out_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(out_features)
        self.conv2 = torch.nn.Conv2d(out_features, out_features, kernel_size=3,
                                     padding=1, groups=out_features)
        self.act2 = act_layer()
        self.bn2 = nn.BatchNorm2d(out_features)
        self.conv3 = torch.nn.Conv2d(out_features, out_features, kernel_size=3,
                                     padding=1, groups=out_features)
        self.act3 = act_layer()
        self.bn3 = nn.BatchNorm2d(out_features)
        self.conv4 = torch.nn.Conv2d(out_features, in_features, kernel_size=1,
                                     padding=0, groups=out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x=self.conv4(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x

# Cross-task Channel-wise Attention (CCA)
class CCA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., group=-1, reducedim=False,out=False,fusion=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.out=out
        self.reducedim = reducedim
        self.fusion=fusion
        if self.out:
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            if reducedim:
                #if group > 0:
                #    self.qkv = nn.Conv1d(dim, dim*2, 1, groups=group, bias=qkv_bias)
                #else:
                self.qkv = nn.Linear(dim, dim*2, bias=qkv_bias)
            else:
                self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.group = group
        if self.group < 0:
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = nn.Conv1d(dim, dim, 1, groups=group)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,x_out,H,W):
        B, N, C = x.shape
        if self.out:
            kv=self.kv1(x).reshape(B, N, 2, self.num_heads, C //2// self.num_heads)
            kv = kv.permute(2, 0, 3, 1, 4) # (2,B,self.num_heads,N,C // self.num_heads)
            k1,v1 = kv[0],kv[1]

            q_out = self.q2(x_out).reshape(B, N, 1, self.num_heads, C // self.num_heads)
            q_out = q_out.permute(2, 0, 3, 1, 4)
            q2 = q_out[0]

            kv_out=self.kv2(x_out).reshape(B, N, 2, self.num_heads, C //2// self.num_heads)
            kv_out = kv_out.permute(2, 0, 3, 1, 4) # (2,B,self.num_heads,N,C // self.num_heads)
            k2,v2 = kv_out[0],kv_out[1]

            q=q2
            k=torch.cat([k1,k2],dim=3)
            v=torch.cat([v1,v2],dim=3)
        else:
            if self.reducedim:
                #if self.group > 0:
                #    qkv = self.qkv(x.permute(0,2,1).contiguous()).permute(0,2,1).contiguous().reshape(B, N, 2, self.num_heads, C // self.num_heads)
                #else:
                qkv = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[0]
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        if self.group < 0:
            x = self.proj(x)
        else:
            x = x.permute(0, 2, 1).contiguous()
            x = self.proj(x)
            x = x.permute(0, 2, 1).contiguous()
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

# Cross-task Interaction Transformer (CIT)
class CIT(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tokens=196, eta=None, lmlp=True, ffnmlp=True, normlatter=False, group=-1,
                 reducedim=False,out=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_1 = norm_layer(dim)
        self.out=out
        self.attn = CCA_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, group=group, reducedim=reducedim,out=self.out)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop, group=group)

        self.norm3 = norm_layer(dim)
        self.local_mp = LEB(in_features=dim, act_layer=act_layer)

        self.lmlp = lmlp
        self.ffnmlp = ffnmlp
        self.normlatter = normlatter

        if self.normlatter:
            self.norml = norm_layer(dim)


    def forward(self,feature,feature_out):
        B, C, H, W = feature.shape
        x=feature
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        x_out=feature_out.view(B, C, -1).permute(0, 2, 1).contiguous()

        x = x + self.drop_path(self.attn(self.norm1(x),self.norm1_1(x_out),H,W))
        if self.lmlp:
            x = x + self.drop_path(self.local_mp(self.norm3(x), H, W))
        if self.ffnmlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.normlatter:
            x = self.norml(x)
        return x.permute(0, 2, 1).view(B, C, H, W).contiguous()

