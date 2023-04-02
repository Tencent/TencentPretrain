import torch


def save_model(model, model_path, use_lora=False):
    """
    Save model weights to file.
    """
    if hasattr(model, "module"):
        if use_lora:
            import loralib as lora
            torch.save(lora.lora_state_dict(model.module), model_path)
        else:
            torch.save(model.module.state_dict(), model_path)
    else:
        if use_lora:
            import loralib as lora
            torch.save(lora.lora_state_dict(model), model_path)
        else:
            torch.save(model.state_dict(), model_path)
