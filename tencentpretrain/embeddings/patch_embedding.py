import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer.
    """

    def __init__(self, args, _):
        super(PatchEmbedding, self).__init__()
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, args.emb_size))
        self.image_height = args.image_height
        self.image_width = args.image_width
        patch_size = (args.patch_size, args.patch_size)
        channels_num = args.channels_num

        self.projection = nn.Conv2d(channels_num, args.emb_size, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, src, _):
        # batch_size, channels_num, height, width
        batch_size, _, height, width = src.shape
        if height != self.image_height or width != self.image_width:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_height}*{self.image_width})."
            )
        patch_emb = self.projection(src).flatten(2).transpose(1, 2)
        cls_emb = self.cls_emb.expand(batch_size, -1, -1).to(patch_emb.device)
        patch_emb = torch.cat((cls_emb, patch_emb), dim=1)

        return patch_emb
