from functools import partial

import torch
import torch.nn as nn
import models.Transformer_block
import matplotlib.pyplot as plt
import os


class MAE(nn.Module):
    """
    Build a Masked Autoencoder model for PPG embedding and reconstruction.
    """

    def __init__(self, sequence_length=1250, patch_length=50, stride_length=50, in_c=1, dim_encoder=64, depth=4,
                 num_head=8, dim_decoder=32, decoder_depth=3, decoder_num_head=8, mlp_ratio=4, drop_ratio=0.,
                 attn_drop_ratio=0., channel_num=2, bias=True, norm_layer=None, act_layer=None, norm_pix_loss=False):
        super(MAE, self).__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # MAE encoder module-----------------------------------------
        self.patch_embed = models.Transformer.Patch_Embed(
            sequence_length=sequence_length, patch_length=patch_length, stride_length=stride_length, in_c=in_c,
            d_model=dim_encoder
        )
        self.channel_num = channel_num
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_encoder))
        self.PE = models.Transformer.PositionalEncoding(d_model=dim_encoder, dropout=drop_ratio, cls_token=True)
        self.blocks = nn.Sequential(*[
            models.Transformer.Transformer_Block(
                d_model=dim_encoder, num_attention_heads=num_head, mlp_ratio=mlp_ratio, bias=bias,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(depth)])
        self.norm = norm_layer(dim_encoder)
        # MAE decoder module-------------------------------------------
        self.decoder_embedding = nn.Linear(dim_encoder, dim_decoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim_decoder))
        self.PE_decoder = models.Transformer.PositionalEncoding(d_model=dim_decoder, dropout=drop_ratio, cls_token=True)
        self.decoder_blocks = nn.Sequential(*[
            models.Transformer.Transformer_Block(
                d_model=dim_decoder, num_attention_heads=decoder_num_head, mlp_ratio=mlp_ratio, bias=bias,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(dim_decoder)
        self.decoder_pred = nn.Linear(dim_decoder, patch_length * in_c)
        # --------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_normal_(w.view([w.shape[0], -1]))

        # initialize token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def random_mask(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, signals):
        """
        signals:(B, C, L)
        return: (B, h, p*C)
        split input signals into patches
        """
        p = self.patch_embed.patch_length
        assert signals.shape[2] % p == 0

        h = signals.shape[2] // p
        c = signals.shape[1]
        x = signals.reshape(shape=(signals.shape[0], signals.shape[1], h, p))
        x = torch.einsum('nchp->nhpc', x)
        x = x.reshape(shape=(x.shape[0], h, p * c))

        return x

    def unpatchify(self, x, c):
        """
        x: (B, L, patch_length*C)
        c: channel
        return:(B, C, L*patch_length)
        re-organize patches into signals
        """
        p = self.patch_embed.patch_length
        l = x.shape[1]
        c = c
        x = x.reshape(shape=(x.shape[0], l, p, c))
        x = torch.einsum('nlpc->nclp', x)
        signal = x.reshape(shape=(x.shape[0], c, l * p))
        return signal

    def get_reconstruct_signal(self, x, target, epoch, mask_ratio=0.75,
                               save_file='fig'):
        """
        use one signal input MAE, get its reconstruction and target signal.
        Plot them and Save.
        x: input signal [1, C, L]
        target: target signal
        epoch: training epoch
        """
        if not os.path.isdir(save_file):
            os.makedirs(save_file)
            print(f"folder {save_file} created")
        self.eval()
        b, h, t = x.shape
        x = x.reshape([b * h, t]).unsqueeze(dim=1)
        target = target.reshape([b * h, t]).unsqueeze(dim=1)
        mean_x = x.mean(dim=-1, keepdim=True)  # [b, h, 1]
        x = x - mean_x
        std_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        x = x / std_x
        mean_target = target.mean(dim=-1, keepdim=True)  # [b * h, 1, 1]
        target = target - mean_target
        std_target = torch.sqrt(torch.var(target, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        target = target / std_target
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss_mae = self.forward_loss(target, pred, mask)
        pred = self.unpatchify(pred, target.size(1))
        for i in range(h):
            plt.subplot(h + 1, 1, 1)
            plt.title(f'MAE:{loss_mae.item():.4f}')
            plt.plot(pred[i, 0, :].detach().cpu().squeeze(), label='Reconstructed')
            plt.legend()
            plt.subplot(h + 1, 1, 2)
            plt.plot(target[i, 0, :].detach().cpu().squeeze(), label='Raw')
            plt.legend()
            plt.subplot(h + 1, 1, 3)
            plt.plot(pred[i, 0, :].detach().cpu().squeeze(), label='Reconstructed')
            plt.plot(target[i, 0, :].detach().cpu().squeeze(), label='Raw')
            plt.legend()
            save_name = 'PPG_epoch_' + str(epoch) + 'Channel' + str(i+1)
            save_path = os.path.join(save_file, save_name)
            plt.savefig(save_path, )
            plt.close()

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)  # [b * h, w, d]
        # add pos embed with cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.PE(x)
        # Divide x into cls_token and x for mask x
        cls_tokens_1 = x[:, :1, :]
        x = x[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_mask(x, mask_ratio)  # [b * h, w', d]

        # append cls token
        x = torch.cat((cls_tokens_1, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embedding(x)  # [b * h, w', d']
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle [b * h, w, d']
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
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

    def forward_loss(self, target, pred, mask):
        """
        target: [N, L*dim]
        pred: [N, L, dim]
        mask: [N, L], 0 is keep, 1 is remove,
        calculate MSE on reconstructed signal patches
        """
        target = self.patchify(target)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, target, mask_ratio=0.75):
        # channel-independence
        b, c, t = x.shape
        x = x.reshape([b * c, t]).unsqueeze(dim=1)
        target = target.reshape([b * c, t]).unsqueeze(dim=1)
        # z-score normalization for each signal
        mean_x = x.mean(dim=-1, keepdim=True)  # [b * c, 1, 1]
        x = x - mean_x
        std_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        x = x / std_x
        mean_target = target.mean(dim=-1, keepdim=True)  # [b * c, 1, 1]
        target = target - mean_target
        std_target = torch.sqrt(torch.var(target, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        target = target / std_target
        # masked auto-encoder
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        # MSE loss
        loss_mse = self.forward_loss(target, pred, mask)

        return loss_mse


def MAE_model(channel_num=2):
    return MAE(sequence_length=1250, patch_length=50, stride_length=50, in_c=1, dim_encoder=512, depth=8,
               num_head=8, dim_decoder=256, decoder_depth=3, decoder_num_head=8, mlp_ratio=4, drop_ratio=0.,
               attn_drop_ratio=0., channel_num=channel_num, bias=True, norm_layer=None, act_layer=None, norm_pix_loss=False)



if __name__ == '__main__':
    model = MAE(sequence_length=1250, patch_length=50, stride_length=50, in_c=1, dim_encoder=512, depth=8,
                num_head=8, dim_decoder=256, decoder_depth=3, decoder_num_head=8, mlp_ratio=4, drop_ratio=0.,
                attn_drop_ratio=0., bias=True, norm_layer=None, act_layer=None, norm_pix_loss=False)
    model.requires_grad_(True)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))