import torch
from tencentpretrain.utils.lora import lora_state_dict


def save_model(model, model_path, use_lora=False):
    """
    Save model weights to file.
    """
    if hasattr(model, "module"):
        if use_lora:
            torch.save(lora_state_dict(model.module), model_path)
        else:
            torch.save(model.module.state_dict(), model_path)
    else:
        if use_lora:
            torch.save(lora_state_dict(model), model_path)
        else:
            torch.save(model.state_dict(), model_path)
