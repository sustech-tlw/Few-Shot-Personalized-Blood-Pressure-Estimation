import torch
import torch.nn as nn
import torch.nn.functional as F
import models.AttetionPool as AttentionPool


def random_mask(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    """
    N, L, D = x.shape  # batch, length, dim
    x_masked = torch.zeros(x.shape, device=x.device)
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids = torch.argsort(noise, dim=1)  # small is keep, large is remove
    ids_restore = torch.argsort(ids, dim=1)

    # keep the first subset
    ids_keep = ids[:, :len_keep]
    for i in range(len(ids_keep)):
        x_masked[i, ids_keep[i], :] = x[i, ids_keep[i], :]

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask


def forward_loss(target, pred, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, group=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.gn1 = nn.GroupNorm(group, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.gn2 = nn.GroupNorm(group, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(group, out_ch)
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return self.relu(out)


# ========== 3. ResUNet1D ==========
class ResUNet1D(nn.Module):
    """
    1D ResUNet for masked reconstruction.
    encoder-decoder with skip connections, suitable for pseudo-periodic biosignals.
    """

    def __init__(self, in_ch=1, base_ch=32, depth=4, kernel_size=7):
        super().__init__()
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.channel_num = in_ch
        ch = in_ch
        for d in range(depth):
            out_ch = base_ch * (2 ** d)
            self.enc_blocks.append(ResidualBlock1D(ch, out_ch, kernel_size))
            ch = out_ch
            self.pools.append(nn.MaxPool1d(2, 2))  # 下采样2倍

        # Bottleneck
        self.bottleneck = ResidualBlock1D(ch, ch * 2, kernel_size)
        ch = ch * 2
        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for d in reversed(range(depth)):
            out_ch = base_ch * (2 ** d)
            self.upconvs.append(nn.ConvTranspose1d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            self.dec_blocks.append(ResidualBlock1D(out_ch * 2, out_ch, kernel_size))
            ch = out_ch

        # Final reconstruction
        self.final_conv = nn.Conv1d(base_ch, in_ch, kernel_size=1)

    def forward_encoder(self, x):
        enc_feats = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            enc_feats.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        return x, enc_feats

    def forward(self, x, target, mask_ratio=0.75, patch_length=50):
        b, c, t = x.shape
        x = x.reshape([b * c, t])
        target = target.reshape([b * c, t])
        # z-score normalization for each signa
        mean_x = x.mean(dim=-1, keepdim=True)  # [b * c, 1]
        x = x - mean_x
        std_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        x = x / std_x
        mean_target = target.mean(dim=-1, keepdim=True)  # [b * c, 1]
        target = target - mean_target
        std_target = torch.sqrt(torch.var(target, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        target = target / std_target
        #
        x = x.reshape([b * c, -1, patch_length])
        x, mask = random_mask(x, mask_ratio)
        x = x.reshape([b, c, -1])
        target = target.reshape([b * c, -1, patch_length])
        x, enc_feats = self.forward_encoder(x)
        for up, dec, enc_feat in zip(self.upconvs, self.dec_blocks, reversed(enc_feats)):
            x = up(x)
            # padding if size do not match
            if x.shape[-1] != enc_feat.shape[-1]:
                diff = enc_feat.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    x = x[..., :enc_feat.shape[-1]]
            x = torch.cat([x, enc_feat], dim=1)
            x = dec(x)
        x = self.final_conv(x)
        x = x.reshape(target.shape)

        loss = forward_loss(target, x, mask)
        return loss


class ResNet1D(nn.Module):
    """
    1D ResUNet for masked reconstruction.
    encoder-decoder with skip connections, suitable for pseudo-periodic biosignals.
    """

    def __init__(self, in_ch=1, base_ch=32, depth=4, kernel_size=7, num_classes=1, global_pool='mean'):
        super().__init__()
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.multi_channel = in_ch
        self.global_pool = global_pool
        ch = in_ch
        for d in range(depth):
            out_ch = base_ch * (2 ** d)
            self.enc_blocks.append(ResidualBlock1D(ch, out_ch, kernel_size))
            ch = out_ch
            self.pools.append(nn.MaxPool1d(2, 2))  # 下采样2倍

        # Bottleneck
        self.bottleneck = ResidualBlock1D(ch, ch * 2, kernel_size)
        ch = ch * 2
        self.fc = nn.Linear(ch, num_classes)
        if self.global_pool == 'attention_pool':
            self.pool = AttentionPool.AttentionPool(num_attention_heads=8, d_model=ch)
        elif self.global_pool == 'mean':
            self.mean_pool = nn.AdaptiveAvgPool1d((1))

    def forward(self, x):
        mean_x = x.mean(dim=-1, keepdim=True)  # [b * h, 1]
        x = x - mean_x
        std_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6)
        x = x / std_x
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            x = pool(x)

        x = self.bottleneck(x)
        if self.global_pool == 'attention_pool':
            x = self.pool(x.transpose(1, 2))
        elif self.global_pool == 'mean':
            x = self.mean_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return {'pred': x.squeeze()}


def ResUNet_model(channel_num, base_ch=64, depth=4, kernel_size=11):
    return ResUNet1D(in_ch=channel_num, base_ch=base_ch, depth=depth, kernel_size=kernel_size)


def ResNet_model(channel_num, base_ch=64, depth=4, kernel_size=11):
    return ResNet1D(in_ch=channel_num, base_ch=base_ch, depth=depth, kernel_size=kernel_size, global_pool='attention_pool')


if __name__ == '__main__':
    input = torch.randn([2, 2, 1250])
    model = ResNet1D(in_ch=2, base_ch=64, depth=4, kernel_size=11, global_pool='attention_pool')
    output_ = model(input)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(output_['pred'].shape)
    print(output_['pred'].shape)
