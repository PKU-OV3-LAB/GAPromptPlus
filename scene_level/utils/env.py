import os
import psutil
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id, num_workers, rank, world_size, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        world_size (int): Number of processes.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + world_size + seed
    set_seed(worker_seed)


def cleanup():
    # close all sub-process on exit
    for child in torch.multiprocessing.active_children():
        child.terminate()
        child.join()
        child.close()

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    return
