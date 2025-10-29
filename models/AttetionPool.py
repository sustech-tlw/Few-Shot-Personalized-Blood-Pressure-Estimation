import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """ Attention pooling w/ latent query
    """
    def __init__(self, num_attention_heads, d_model, bias=True, attn_drop_ratio=0., drop_ratio=0., latent_len=1, mlp_ratio=4):
        super(AttentionPool, self).__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (d_model, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.d_attention_head = int(d_model / num_attention_heads)
        self.all_head_size = d_model

        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, d_model))

        self.key_layer = nn.Linear(d_model, d_model, bias=bias)
        self.query_layer = nn.Linear(d_model, d_model, bias=bias)
        self.value_layer = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(drop_ratio)

        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(drop_ratio)
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.latent, std=0.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape

        q_latent = self.latent.expand(B, -1, -1)
        q = self.query_layer(q_latent).reshape(B, self.latent_len, self.num_attention_heads, self.d_attention_head).transpose(1, 2)

        k = self.key_layer(x).reshape(B, N, self.num_attention_heads, self.d_attention_head).permute(0, 2, 1, 3)
        v = self.value_layer(x).reshape(B, N, self.num_attention_heads, self.d_attention_head).permute(0, 2, 1, 3)

        q = q * self.d_attention_head ** -0.5
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        x = x[:, 0]

        # attn_mean = attn.mean(dim=[0,1,2])
        return x

if __name__ == '__main__':
    model = AttentionPool(d_model=512, num_attention_heads=8, mlp_ratio=4)
    x = torch.rand([2, 10, 512])
    out = model(x)
    print(out.shape)