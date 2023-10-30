# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import os
import time
import datetime
import numpy as np
import torch
from tencentpretrain import mpu
from tencentpretrain.mpu import (
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)


def init_env(args):

    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.global_rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed)
    
    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed(dist_backend=args.backend)
        if args.use_mp:
            set_global_variables(args)
            args = get_args()
            finish_mpu_init()
            # Initialize memory buffers.
            _initialize_mem_buffs()
            args.global_rank = mpu.get_data_parallel_rank()
        else:
            args.global_rank = torch.distributed.get_rank()
    elif args.dist_train:
        # Initialize multiprocessing distributed training environment.
        args.global_rank = args.gpu_ranks[args.local_rank]
        torch.distributed.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=args.global_rank)
    elif args.single_gpu:
        args.global_rank = None
    else:
        args.global_rank = None

    return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        # from megatron.data.dataset_utils import compile_helper
        # compile_helper()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = (
        args.num_attention_heads / args.tensor_model_parallel_size
    ) * args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (
        seq_len > 16
        and seq_len <= 2048
        and seq_len % 4 == 0
        and attn_batch_size % 4 == 0
    )
    # Print a warning.
    if not (
        (args.fp16 or args.bf16)
        and custom_kernel_constraint
        and args.masked_softmax_fusion
    ):
        if args.global_rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def setup_deepspeed_random_and_activation_checkpointing(args):
    """Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    """
    num_layers = args.layers_num // args.deepspeed_checkpoint_layers_num
    num_layers = (
        num_layers
        if args.layers_num % args.deepspeed_checkpoint_layers_num == 0
        else num_layers + 1
    )

    import deepspeed
    deepspeed.checkpointing.configure(
        mpu
    )

    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = (
        deepspeed.checkpointing.model_parallel_cuda_manual_seed
    )


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.global_rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.global_rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.global_rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.global_rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.cuda.set_device(device)

        torch.distributed.init_process_group(backend=args.backend,
                                            init_method=args.master_ip,
                                            world_size=args.world_size,
                                            rank=args.global_rank)
        print(f"  > (rank={args.global_rank}) process group initialized")

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
            )

    if args.deepspeed and args.deepspeed_checkpoint_activations:
        setup_deepspeed_random_and_activation_checkpointing(args)


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def initialize_wandb_experiment():
    """Initialize wandb experiment."""
    assert wandb is not None, "Fail to import wandb"

    args = get_args()
    config = args.__dict__

    wandb_id_path = os.path.join(args.save, "wandb_id.txt")
    if os.path.exists(wandb_id_path):
        wandb_id = open(wandb_id_path, "r").read().strip()
    else:
        wandb_id = wandb.util.generate_id()
        open(wandb_id_path, "w").write(wandb_id)

    wandb.init(id=wandb_id, project="megatron", config=config, resume="allow")


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        mpu.init_checkpointed_activations_memory_buffer()


def set_global_variables(args):
    args = _parse_args(args)


def _parse_args(args):
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

    return _GLOBAL_ARGS


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")

    return _GLOBAL_ARGS


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)
