import torch
import sys


def count_lines(file_path):
    lines_num = 0
    if file_path.endswith(".json"):
        import json
        with open(file_path, 'rb') as f:
            data = json.load(f)
            return len(data)
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(2 ** 20)
            if not data:
                break
            lines_num += data.count(b'\n')
    return lines_num


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def pooling(memory_bank, seg, pooling_type):
    seg = torch.unsqueeze(seg, dim=-1).type_as(memory_bank)
    memory_bank = memory_bank * seg
    if pooling_type == "mean":
        features = torch.sum(memory_bank, dim=1)
        features = torch.div(features, torch.sum(seg, dim=1))
    elif pooling_type == "last":
        features = memory_bank[torch.arange(memory_bank.shape[0]), torch.squeeze(torch.sum(seg!=0, dim=1).type(torch.int64) - 1), :]
    elif pooling_type == "max":
        features = torch.max(memory_bank + (seg - 1) * sys.maxsize, dim=1)[0]
    else:
        features = memory_bank[:, 0, :]
    return features


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


def expand2square(img, background_color=(122, 116, 104)):
    from PIL import Image
    width, height = img.size
    if img.mode != "RGB":
        img = img.convert("RGB")
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), background_color)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), background_color)
        result.paste(img, ((height - width) // 2, 0))
        return result
