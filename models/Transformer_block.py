from functools import partial

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
        "Implement the PE function."
    """

    def __init__(self, d_model, dropout, max_len=5000, cls_token=False):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if cls_token:
            if type(cls_token) == bool:
                pe = torch.cat([torch.zeros(1, d_model), pe], dim=0)
            elif type(cls_token) == int:
                pe = torch.cat([torch.zeros(cls_token, d_model), pe], dim=0)
        pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)
        self.pe = pe
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        device = x.device
        x = x + self.pe[:, :x.size(1)].to(device)
        return self.dropout(x)


class SelfAttention(nn.Module):
    """
    num_attention_heads； 头个数
    d_model: 输入token维度
    d_hidden: 所有头的token维度,一般等于d_model
    attn_drop_ration： 注意力dropout比例
    proj_drop_ration： 线性投影...比例
    """

    def __init__(self, num_attention_heads, d_model, d_hidden, bias=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(SelfAttention, self).__init__()
        if d_hidden % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (d_hidden, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.d_attention_head = int(d_hidden / num_attention_heads)
        self.all_head_size = d_hidden

        self.key_layer = nn.Linear(d_model, d_hidden, bias=bias)
        self.query_layer = nn.Linear(d_model, d_hidden, bias=bias)
        self.value_layer = nn.Linear(d_model, d_hidden, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(d_hidden, d_model)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.d_attention_head)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3) # B, num_h, L, d

    def forward(self, x, atten_mask=None):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.d_attention_head)
        if atten_mask is not None:
            attention_scores = attention_scores.masked_fill(atten_mask.unsqueeze(0).unsqueeze(1) == 1, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        x = torch.matmul(attention_probs, value_heads)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_size = x.size()[: -2] + (self.all_head_size,)
        x = x.view(*new_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1, act_layer=nn.GELU):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = act_layer()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))


class Transformer_Block(nn.Module):
    def __init__(self,
                 d_model,
                 num_attention_heads,
                 mlp_ratio=4.,
                 bias=False,
                 attn_drop_ratio=0.,
                 drop_ratio=0.,
                 # drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Transformer_Block, self).__init__()
        self.norm1 = norm_layer(d_model) if norm_layer else nn.Identity()
        self.attn = SelfAttention(num_attention_heads, d_model, d_model, attn_drop_ratio=attn_drop_ratio,
                                  proj_drop_ratio=drop_ratio, bias=bias)
        # self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model) if norm_layer else nn.Identity()
        mlp_d_hidden = int(d_model * mlp_ratio)
        self.mlp = PositionwiseFeedForward(d_model, mlp_d_hidden, dropout=drop_ratio, act_layer=act_layer)

    def forward(self, x, atten_mask=None):
        x = x + self.attn(self.norm1(x), atten_mask)
        x = x + self.mlp(self.norm2(x))

        return x


class Patch_Embed(nn.Module):
    """
    sequence_length: 序列长度
    patch_length: 一个编码patch序列的长度
    in_c: 序列输入维度
    d_model: 输出token的维度
    norm_Layer: 归一化方法
    """

    def __init__(self, sequence_length, patch_length, stride_length, in_c, d_model, norm_Layer=None):
        super(Patch_Embed, self).__init__()
        self.sequence_size = (in_c, sequence_length)
        self.patch_length = patch_length
        self.num_patches = sequence_length // patch_length
        self.proj = nn.Conv1d(in_channels=in_c, out_channels=d_model, kernel_size=patch_length, stride=stride_length)
        self.norm = norm_Layer(d_model) if norm_Layer else nn.Identity()

    def forward(self, seq):
        B, C, L = seq.shape
        assert C == self.sequence_size[0], \
            f"Input size ({C}) doesn't match model ({self.sequence_size[0]})."
        x = self.proj(seq).transpose(1, 2)
        x = self.norm(x)
        return x

