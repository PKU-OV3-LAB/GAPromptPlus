"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import time, pickle
from typing import Union

import functools
import numpy as np
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use `launch()` to start processes and initialize pytorch process group."
    "If you need to start processes in other ways, please call comm.create_local_process_group(num_workers_per_machine)"
    "after calling torch.distributed.init_process_group()."
)
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    """
    Create a process group that contains ranks within the same machine.

    `launch()` will call this function. If you start workers without launch(), you'll have to also call this.
    Otherwise utilities like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine. Typically the number of GPUs.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    """
    Returns:
        A torch process group which only includes processes that are on the same
        machine as the current process. This group can be useful for communication
        within a machine, e.g. a per-machine SyncBN.
    """
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return _LOCAL_PROCESS_GROUP


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_dict_arr(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be CUDA Tensor of the same shape (can only be diff at dim-0).
        average (bool): whether to do average or sum

    Returns:
        a dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        dim0_list = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
            dim0_list.append(input_dict[k].shape[0])  # to enable diff-size tensor in the input_dict (only diff at dim-0)
        values = torch.cat(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        values = values.split(dim0_list)
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


@torch.no_grad()
def all_gather_first_dim(data, world_size=None):
    """
    Run all_gather on diff-size (only diff in 1st dim) tensor
    Args:
        data: tensor
    Returns:
        list[data]: list of data, ready for torch.stack
    """
    if world_size is None:
        world_size = get_world_size()
    if world_size == 1:
        return [data]

    dims = data.shape
    tensor = data.to("cuda")

    if len(dims) == 0:
        # scalar data
        max_shape = torch.Size([])
        tensor_list = [torch.zeros(size=max_shape, dtype=data.dtype, device=data.device) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        data_list = tensor_list

    else:
        # tensor
        dims_0 = dims[0]
        dims_1s = dims[1:]

        # obtain Tensor size of each rank
        local_size = torch.tensor([dims_0], dtype=torch.int64, device=data.device)
        size_list = [torch.tensor([0], dtype=torch.int64, device=data.device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)
        max_shape = torch.Size([max_size]) + dims_1s

        # receiving Tensor from all ranks
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        tensor_list = [torch.zeros(size=max_shape, dtype=data.dtype, device=data.device) for _ in size_list]
        if local_size != max_size:
            padding = torch.zeros(size=torch.Size([max_size - local_size]) + dims_1s, dtype=data.dtype, device=data.device)
            tensor = torch.cat([data, padding], dim=0)

        dist.all_gather(tensor_list, tensor)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            data_list.append(tensor[:size])

    return data_list


def sync_tensor_across_gpus(t: Union[torch.Tensor, None], group_size: Union[int, None] = None) -> Union[torch.Tensor, list, None]:
    # t needs to have dim 0 for troch.cat below.
    # if not, you need to prepare it.
    if t is None:
        return None
    if group_size is None:
        group = dist.group.WORLD
        group_size = torch.distributed.get_world_size(group)
    t = t.contiguous()
    gather_t_tensor = [torch.zeros_like(t) for _ in range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu.
   # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with
   # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the
   # same definition... see [here](https://pytorch.org/docs/stable/distributed.html).
   #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for
   # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in
   # the doc is  vague...

    # return torch.cat(gather_t_tensor, dim=0)
    return gather_t_tensor
