from functools import partial

import torch
import torch.nn as nn
import models.Transformer_block as Transformer_block
import models.AttetionPool


class Transformer(nn.Module):
    """
    Build a Masked Autoencoder model for PPG embedding and reconstruction.
    """

    def __init__(self, sequence_length=1250, patch_length=50, stride_length=50, in_c=1, dim_encoder=64, depth=8,
                 num_head=8, mlp_ratio=4, drop_ratio=0., attn_drop_ratio=0., bias=True, norm_layer=None, act_layer=None,
                 global_pool='mean', head_type='nonLinear', num_classes=1, channel_num=2,
                 auxiliary_task=False, dim_decoder=32, decoder_depth=3, decoder_num_head=8):
        super(Transformer, self).__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.global_pool = global_pool
        self.head_type = head_type
        self.channel_num = channel_num
        self.auxiliary_task = auxiliary_task
        # MAE encoder module-----------------------------------------
        self.patch_embed = Transformer_block.Patch_Embed(
            sequence_length=sequence_length, patch_length=patch_length, stride_length=stride_length, in_c=in_c,
            d_model=dim_encoder
        )
        self.w = self.patch_embed.num_patches
        self.PE = Transformer_block.PositionalEncoding(d_model=dim_encoder, dropout=drop_ratio, cls_token=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_encoder))
        self.blocks = nn.Sequential(*[
            Transformer_block.Transformer_Block(
                d_model=dim_encoder, num_attention_heads=num_head, mlp_ratio=mlp_ratio, bias=bias,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(depth)])
        self.norm = nn.LayerNorm(dim_encoder, elementwise_affine=False)
        # --------------------------------------------------------------
        if self.global_pool == 'attention_pool':
            self.attention_pool = models.AttetionPool.AttentionPool(num_attention_heads=num_head, d_model=dim_encoder * channel_num)

        if head_type == 'Linear':
            self.head = nn.Linear(dim_encoder * channel_num, num_classes)
        elif head_type == 'nonLinear':
            self.head = nn.Sequential(
                    nn.Linear(dim_encoder * channel_num, dim_encoder * channel_num // 2),
                    nn.ReLU(),
                    nn.Linear(dim_encoder * channel_num // 2, num_classes)
                )
        # --------------------------------------------------------------------
        # decord head for auxiliary task
        if self.auxiliary_task:
            self.decoder_embedding = nn.Linear(dim_encoder, dim_decoder)
            self.PE_decoder = Transformer_block.PositionalEncoding(d_model=dim_decoder, dropout=drop_ratio,
                                                                    cls_token=True)
            self.decoder_blocks = nn.Sequential(*[
                Transformer_block.Transformer_Block(
                    d_model=dim_decoder, num_attention_heads=decoder_num_head, mlp_ratio=mlp_ratio, bias=bias,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio, norm_layer=norm_layer, act_layer=act_layer)
                for i in range(decoder_depth)
            ])
            self.decoder_norm = norm_layer(dim_decoder)
            self.decoder_pred = nn.Linear(dim_decoder, patch_length * in_c)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_normal_(w.view([w.shape[0], -1]))

        # initialize token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # pos embed
        x = self.PE(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embedding(x)
        # add pos embed
        x = self.PE_decoder(x)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward(self, x):
        b, h, t = x.shape #[b, h, t]
        x = x.reshape([b * h, t]).unsqueeze(dim=1) # [b * h , 1, t]
        mean = x.mean(dim=-1, keepdim=True)  # [b, 1, 1]
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        x = x / std
        latent = self.forward_encoder(x) #[b * h , l, d]
        _, l, d = latent.shape
        latent_ = latent.reshape([b, h, l, d]).transpose(1, 2).reshape([b, l, (h * d)])
        if self.global_pool == 'mean':
            latent_ = torch.mean(latent_[:, 1:, :], dim=1)
        elif self.global_pool == 'token':
            latent_ = latent_[:, 0]
        elif self.global_pool == 'attention_pool':
            latent_ = self.attention_pool(latent_[:, 1:, :])
        pred = self.head(latent_).squeeze(dim=-1)
        if self.auxiliary_task:
            regression = self.forward_decoder(latent)
            return {'pred': pred, 'regression': regression}
        else:
            return {'pred': pred}


def Transformer_model(channel_num=2):
    return Transformer(sequence_length=1250, patch_length=50, stride_length=50, in_c=1, dim_encoder=512,
                       depth=8, num_head=8, mlp_ratio=4, drop_ratio=0., attn_drop_ratio=0., bias=True,
                       norm_layer=None, act_layer=None, global_pool='attention_pool', head_type='nonLinear', num_classes=1,
                       channel_num=channel_num, auxiliary_task=False, dim_decoder=256, decoder_depth=3, decoder_num_head=8)