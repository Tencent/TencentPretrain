import yaml
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from taming.models.vqgan import VQModel, GumbelVQ
from taming.models.cond_transformer import Net2NetTransformer
from PIL import Image
from torchvision.utils import make_grid, save_image
from math import sqrt, log
#https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/vae.py#L160

def load_vqgan(config, ckpt_path=None, is_gumbel=False, is_transformer=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    elif is_transformer:
        model = Net2NetTransformer(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)

    if is_transformer:
        model = model.first_stage_model
    return model


def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x


def build_vqgan_model(args):
    config = OmegaConf.load(args.vqgan_config_path)
    vqgan_model = load_vqgan(config, ckpt_path=args.vqgan_model_path,
                             is_transformer=args.image_tokenizer["is_transformer"],
                             is_gumbel=args.image_tokenizer["is_gumbel"])
    return vqgan_model


def image_tokenize(vqgan_model, image, is_gumbel=False):
    image = torch.stack([preprocess_vqgan(image)], 0)
    with torch.no_grad():
        _, _, [_, _, indices] = vqgan_model.encode(image)
        if is_gumbel:
            image_tokens = rearrange(indices, 'b h w -> b (h w)', b = 1).flatten().tolist()
        else:
            image_tokens = rearrange(indices, '(b n) -> b n', b = 1).flatten().tolist()

    return image_tokens


def image_tokenize_batch(vqgan_model, images, is_gumbel=False):
    image_src = torch.stack([preprocess_vqgan(image) for image in images], 0)
    with torch.no_grad():
        _, _, [_, _, indices] = vqgan_model.encode(image_src)
        if is_gumbel:
            image_tokens = rearrange(indices, 'b h w -> b (h w)', b = len(images)).tolist()
        else:
            image_tokens = rearrange(indices, '(b n) -> b n', b = len(images)).tolist()

    return image_tokens


def image_detokenize(vqgan_model, image_tokens, image_vocab_size=1024, is_gumbel=False, save_path=None):
    with torch.no_grad():
        b, n = 1, len(image_tokens)
        one_hot_indices = F.one_hot(torch.tensor([image_tokens]), num_classes = image_vocab_size).float().to(vqgan_model.device)
        z = one_hot_indices @ vqgan_model.quantize.embed.weight if is_gumbel \
            else (one_hot_indices @ vqgan_model.quantize.embedding.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n))).to(vqgan_model.device)
        img = vqgan_model.decode(z)
        img = (img.clamp(-1., 1.) + 1) * 0.5

    if save_path:
        save_image(img, save_path, normalize=False)
    return img


