import torch


def load_model(model, model_path, lora_pretrained_model_path):
    """
    Load model from saved weights.
    """
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        if lora_pretrained_model_path is not None:
            model.module.load_state_dict(torch.load(lora_pretrained_model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        if lora_pretrained_model_path is not None:
            model.load_state_dict(torch.load(lora_pretrained_model_path, map_location="cpu"), strict=False)
    return model
