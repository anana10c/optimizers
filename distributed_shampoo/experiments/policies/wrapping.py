# holds various wrapping policies for fsdp


import functools
from typing import Type

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    wrap,
)

from transformers.models.t5.modeling_t5 import T5Block


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_t5_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )

    return t5_auto_wrap_policy
