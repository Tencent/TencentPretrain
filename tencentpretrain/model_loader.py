import os
import torch
from tencentpretrain import mpu


def load_model(model, model_path, lora_pretrained_model_path=None):
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


def _load_state_dict_into_model(model_to_load, model_path, start_prefix=""):
    # Convert old format to new format if needed from a PyTorch state_dict

    # copy state_dict so _load_from_state_dict can modify it
    state_dict = torch.load(model_path, map_location="cpu")
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            import deepspeed
            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return model_to_load


def load_mp_model(model, model_path):

    prefix = os.listdir(model_path)
    weight_list = sorted([os.path.join(model_path, f) for f in prefix])

    tp_rank = mpu.get_tensor_model_parallel_rank()

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(weight_list[tp_rank], map_location="cpu")['module'], strict=False)
    else:
        model.load_state_dict(torch.load(weight_list[tp_rank], map_location="cpu"), strict=True)
    
    return model
