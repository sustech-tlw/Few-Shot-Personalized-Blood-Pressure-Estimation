import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.Transformer as Transformer


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, dropout=0.0, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        else:
            self.lora_A = self.lora_B = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            result += self.scaling * lora_out
        return result

    def freeze_base(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, d_model, d_hidden, bias=None, attn_drop_ratio=0., proj_drop_ratio=0., lora_r=4, lora_alpha=1.0):
        super(SelfAttention, self).__init__()
        if d_hidden % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (d_hidden, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.d_attention_head = int(d_hidden / num_attention_heads)
        self.all_head_size = d_hidden

        # 替换 Q 和 V 为 LoRA
        self.key_layer = nn.Linear(d_model, d_hidden, bias=bias)
        self.query_layer = LoRALinear(d_model, d_hidden, r=lora_r, alpha=lora_alpha, bias=bias)
        self.value_layer = LoRALinear(d_model, d_hidden, r=lora_r, alpha=lora_alpha, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(d_hidden, d_model)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.d_attention_head)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, atten_mask=None):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.d_attention_head)
        if atten_mask is not None:
            atten_mask = (atten_mask == 0)
            attention_scores = attention_scores.masked_fill(atten_mask.unsqueeze(1).unsqueeze(2) == 1, float('-inf'))

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
    def __init__(self, d_model, d_hidden, dropout=0.1, act_layer=nn.GELU, lora_r=4, lora_alpha=1.0, bias=None):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = LoRALinear(d_model, d_hidden, r=lora_r, alpha=lora_alpha, bias=bias)
        self.act = act_layer()
        self.w_2 = LoRALinear(d_hidden, d_model, r=lora_r, alpha=lora_alpha, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))

class Transformer_Block_LoRA(nn.Module):
    def __init__(self,
                 d_model,
                 num_attention_heads,
                 mlp_ratio=4.,
                 bias=False,
                 attn_drop_ratio=0.,
                 drop_ratio=0.,
                 lora_r=4,
                 lora_alpha=1.0,
                 # drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Transformer_Block_LoRA, self).__init__()
        self.norm1 = norm_layer(d_model)
        self.attn = SelfAttention(num_attention_heads, d_model, d_model, attn_drop_ratio=attn_drop_ratio,
                                  proj_drop_ratio=drop_ratio, bias=bias, lora_r=lora_r, lora_alpha=lora_alpha)
        # self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model)
        mlp_d_hidden = int(d_model * mlp_ratio)
        self.mlp = Transformer.PositionwiseFeedForward(d_model, mlp_d_hidden, dropout=drop_ratio, act_layer=act_layer)

    def forward(self, x, atten_mask=None):
        x = x + self.attn(self.norm1(x), atten_mask)
        x = x + self.mlp(self.norm2(x))

        return x

