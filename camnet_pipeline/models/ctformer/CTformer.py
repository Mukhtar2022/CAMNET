"""
CTformer Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
import numpy as np

from .token_transformer import Token_transformer    
from .token_performer import Token_performer
from .T2T_transformer_block import Block, get_sinusoid_encoding

class MultiHeadDense(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch, out_ch))
    
    def forward(self, x):
        x = F.linear(x, self.weight)
        return x

class T2T_module(nn.Module):
    def __init__(self, img_size=64, tokens_type='performer', in_chans=1, embed_dim=256, token_dim=64, kernel=32, stride=32):
        super().__init__()
        if tokens_type == 'transformer':
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1),dilation=(2,2))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_transformer(dim=in_chans*7*7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim*3*3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)
        elif tokens_type == 'performer':
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1),dilation=(2,2))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)
        self.num_patches = 529   ## calculate myself
        
    def forward(self, x):
        x = self.soft_split0(x)
        x = self.attention1(x.transpose(1, 2))
        res_11 = x
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = torch.roll(x, shifts=(2, 2), dims=(2, 3))
        x = self.soft_split1(x)
        x = self.attention2(x.transpose(1, 2))
        res_22 = x
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = torch.roll(x, shifts=(2, 2), dims=(2, 3))
        x = self.soft_split2(x)
        x = self.project(x.transpose(1, 2))
        return x, res_11, res_22
    
class Token_back_Image(nn.Module):
    def __init__(self, img_size=64, tokens_type='performer', in_chans=1, embed_dim=256, token_dim=64, kernel=32, stride=32):
        super().__init__()
        if tokens_type == 'transformer':
            self.soft_split0 = nn.Fold((64,64),kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Fold((29,29),kernel_size=(3, 3), stride=(1, 1),dilation=(2,2))
            self.soft_split2 = nn.Fold((25,25),kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_transformer(dim=token_dim, in_dim=in_chans*7*7, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim, in_dim=token_dim*3*3, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(embed_dim,token_dim * 3 * 3)
        elif tokens_type == 'performer':
            self.soft_split0 = nn.Fold((64,64),kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Fold((29,29),kernel_size=(3, 3), stride=(1, 1),dilation=(2,2))
            self.soft_split2 = nn.Fold((25,25),kernel_size=(3, 3), stride=(1, 1))
            self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.project = nn.Linear(embed_dim,token_dim * 3 * 3)

    def forward(self, x, res_11,res_22):    
        x = self.project(x).transpose(1, 2) 
        x = self.soft_split2(x)
        x = torch.roll(x, shifts=(-2, -2), dims=(-1, -2))
        x = rearrange(x,'b c h w -> b c (h w)').transpose(1,2)
        x = x + res_22
        x = self.attention2(x).transpose(1, 2)
        x = self.soft_split1(x)
        x = torch.roll(x, shifts=(-2, -2), dims=(-1, -2))
        x = rearrange(x,'b c h w -> b c (h w)').transpose(1,2)
        x = x + res_11
        x = self.attention1(x).transpose(1, 2)
        x = self.soft_split0(x) 
        return x

class CTFormer(nn.Module):
    def __init__(self, img_size=512, tokens_type='performer', in_chans=1, num_classes=0, embed_dim=64, depth=1,
                 num_heads=8, kernel=4, stride=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim,kernel=kernel, stride=stride)
        num_patches = self.tokens_to_token.num_patches
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.dconv1 = Token_back_Image(img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim, kernel=kernel, stride=stride)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        res1 = x
        x, res_11, res_22 = self.tokens_to_token(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        out = res1 - self.dconv1(x,res_11,res_22)
        return out
