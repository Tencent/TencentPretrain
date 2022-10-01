import torch
import torch.nn as nn

from tencentpretrain.layers.layer_norm import LayerNorm


class MaskedPatchEmbedding(nn.Module):
    """
    Masked Patch Embedding for BEiT
    """

    def __init__(self, args, _):
        super(MaskedPatchEmbedding, self).__init__()
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, args.emb_size))
        self.mask_emb = nn.Parameter(torch.zeros(1, args.emb_size))
        self.image_height = args.image_height
        self.image_width = args.image_width
        patch_size = (args.patch_size, args.patch_size)
        channels_num = args.channels_num
        self.projection = nn.Conv2d(channels_num, args.emb_size, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, src, _):
        src, mask = src
        batch_size, channels_num, height, width = src.shape
        if height != self.image_height or width != self.image_width:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_height}*{self.image_width})."
            )
        patch_emb = self.projection(src).flatten(2).transpose(1, 2)
        cls_emb = self.cls_emb.expand(batch_size, -1, -1)
        emb = torch.cat((cls_emb, patch_emb), dim=1)

        for sample_idx in range(batch_size):
            mask_emb = self.mask_emb.repeat(len(mask[sample_idx]), 1)
            mask_idx = torch.tensor([[i] * emb.size(2) for i in mask[sample_idx]], device=patch_emb.device)
            emb[sample_idx].scatter_(0, mask_idx, mask_emb)

        return emb
