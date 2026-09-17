import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
from utils import logger
from utils import comm
from utils import env
from configs import load_config, log_config
import torch
import sys
import re
import gc
import copy
import time
import glob
import pickle
import psutil
import argparse
import importlib
import traceback
import numpy as np
from collections import defaultdict
sys.path.insert(0, f'{os.getcwd()}')

if torch.__version__ >= "2.6":
    torch.serialization.add_safe_globals([
        np.dtype,
        np.ndarray,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
        # np._core.multiarray.scalar,
    ])
# torch.multiprocessing.set_sharing_strategy("file_system")

# Custom libs

class SingularAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        seen = getattr(namespace, "_seen_args", set())
        if self.dest in seen:
            parser.error(f"Duplicate option {option_string} detected.")
        seen.add(self.dest)
        setattr(namespace, "_seen_args", seen)
        setattr(namespace, self.dest, values)

class SingularArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *args, action=SingularAction, **kwargs):
        super().add_argument(*args, action=action, **kwargs)


def get_last_train(cfg):
    saving_path = sorted(glob.glob(f'results/{cfg.name}/*'))
    return saving_path[-1] if saving_path else None


def check_save_path(cfg, msg=None):
    if not cfg.save_path and re.fullmatch(r'.*results/.*/model/model_.*\.pth', cfg.weight):
        save_path = os.path.dirname(os.path.dirname(cfg.weight))
    elif not cfg.save_path and not cfg.debug:
        raise ValueError(
            f'should provide save_path or use debug' + msg if msg else '')
    else:
        save_path = cfg.save_path
    return save_path


def solve_envs(epilog=None):
    parser = SingularArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('--cfg_path',
                        type=str,
                        required=True,
                        help='config path')
    parser.add_argument('--save_path',
                        type=str,
                        default=None,
                        help='specified save path')
    parser.add_argument('--gpus',
                        type=str,
                        default=1,
                        help='the number/ID of GPU(s) to use [default: 1] *per machine*, 0 to use cpu only')
    parser.add_argument('--cpus',
                        type=int,
                        default=8,
                        help='the limit the max number of cpus to use')
    parser.add_argument('--machines',
                        type=int,
                        default=1,
                        help='the total number of machines')
    parser.add_argument('--machine_rank',
                        type=int,
                        default=0,
                        help='the rank of this machine (unique per machine)')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        help='train/val/test')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='random seed')
    parser.add_argument('--weight',
                        type=str,
                        default=None,
                        help='pretrained model weight path')
    parser.add_argument('--set',
                        type=str,
                        help='external source to set the config - str of dict / yaml file')
    parser.add_argument('--resume',
                        type=str,
                        nargs='?',
                        const=True,
                        default=None,
                        help='if resume, or with resume weights')
    parser.add_argument('--debug',
                        type=str,
                        nargs='?',
                        const=True,
                        default=None,
                        help='specify debug mode')
    # PyTorch still may leave orphan proces in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument('--dist_url', default='auto', help='initialization URL for pytorch distributed backend.')
    # default="tcp://127.0.0.1:{}".format(port), 'auto'

    FLAGS = parser.parse_args()

    from configs import Base, load_config
    cfg = load_config(cfg_path=FLAGS.cfg_path)

    # Apply external config first, then let explicit CLI arguments take precedence.
    if FLAGS.set:
        for arg in FLAGS.set.split(';'):
            cfg.update(arg, exclude=["gpu_devices", "gpu_num"])
    for arg in ['gpus', 'cpus', 'machines', 'machine_rank', 'mode', 'seed', 'weight', 'save_path', 'resume', 'debug', 'dist_url']:
        if getattr(FLAGS, arg) is not None:
            setattr(cfg, arg, getattr(FLAGS, arg))

    # env setting: visible gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    print("Available #GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, {mem / 1024 ** 3:.1f} GB avail")
    if cfg.cpus:
        torch.set_num_threads(int(cfg.cpus))
    if cfg.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(float(cfg.gpu_frac))

    # env setting: seed
    if not isinstance(cfg.seed, int):
        cfg.seed = env.get_random_seed()

    if cfg.weight:
        assert os.path.isfile(cfg.weight), f'no weight found: {cfg.weight}'

    if cfg.resume:
        model_path = cfg.resume if isinstance(cfg.resume, str) else cfg.weight
        if os.path.isdir(model_path) and os.path.basename(model_path) != 'model':
            model_path = os.path.join(model_path, 'model/model_last.pth')
        elif os.path.isdir(model_path) and os.path.basename(model_path) == 'model':
            model_path = os.path.join(model_path, 'model_last.pth')
        else:
            assert os.path.isfile(model_path), f'cannot resume from: {model_path}'
        cfg.resume = True
        cfg.weight = model_path
        cfg.save_path = check_save_path(cfg, msg=f'\t- resume={cfg.resume}')
        if not any(h.type == 'CheckpointLoader' for h in cfg.hooks):
            cfg.hooks = [Base(dict(type='CheckpointLoader'))] + cfg.hooks

    if cfg.debug:  # debug mode
        cfg.log_file = ''
        cfg.save_path = 'test'

    if cfg.mode == 'train':
        if not cfg.save_path:
            cfg.save_path = f'results/{cfg.name}'
            assert 'train' in cfg.mode, f'no save_path for mode={cfg.mode}'
        os.makedirs(cfg.save_path, exist_ok=True)
        if not cfg.log_file:
            cfg.log_file = os.path.join(cfg.save_path, 'log_train.txt')

    elif cfg.mode in ['val', 'test']:
        assert os.path.isfile(cfg.weight), f'cannot have mode={cfg.mode} without specifying weight'
        os.makedirs(cfg.save_path, exist_ok=True)
        cfg.save_path = check_save_path(cfg, msg=f'\t- mode={cfg.mode}')
        cfg.log_file = os.path.join(cfg.save_path, f"log_test.txt")

    else:
        raise ValueError(f'not support mode={cfg.mode}')
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    assert cfg.mode in ['train', 'test']
    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else psutil.cpu_count(logical=True)
    cfg.num_worker_per_gpu = cfg.num_worker_per_gpu if cfg.num_worker_per_gpu else cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert not cfg.batch_size_val or cfg.batch_size_val % world_size == 0
    assert not cfg.batch_size_test or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = cfg.batch_size_val // world_size if cfg.batch_size_val else 1
    cfg.batch_size_test_per_gpu = cfg.batch_size_test // world_size if cfg.batch_size_test else 1

    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0

    # setting - random seed
    rank = comm.get_rank()
    seed = None if not isinstance(cfg.seed, int) else cfg.seed + rank * cfg.num_worker_per_gpu

    # seed = None if not isinstance(cfg.seed, int) else cfg.seed + rank
    env.set_seed(seed)
    return cfg


# construct trainer & train
def main_worker(cfg):
    # per-process setting & seeding
    cfg = default_setup(cfg)

    # actual training
    from pointcept.engines.train import TRAINERS
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

    # env.set_seed(seed)
    return


def test_worker(cfg):
    cfg = default_setup(cfg)
    torch.cuda.empty_cache()
    if isinstance(cfg.test.use_shared_memory, bool):
        from torch.utils.data._utils.collate import default_collate
        default_collate._use_shared_memory = cfg.test.use_shared_memory
    split = cfg.data.test.split
    test_model = re.split(r"[_-]", os.path.basename(cfg.weight))[-1].split(".")[0]
    log_file = cfg.log_file if cfg.log_file else os.path.join(cfg.save_path, f"log_{split}.txt_{test_model}")
    from pointcept.utils.logger import get_logger
    # new logger (diff prefix with root logger)
    logger = get_logger(f"{split}_{test_model}", log_file=log_file)
    from pointcept.engines.test import TESTERS
    test_cfg = dict(cfg.test, cfg=cfg, logger=logger)
    tester = TESTERS.build(test_cfg)
    tester.test()

    return

# parse env & launch


def main():
    cfg = solve_envs()
    num_gpus = cfg.gpu_num
    num_machines = cfg.machines
    machine_rank = cfg.machine_rank
    dist_url = cfg.dist_url if cfg.dist_url else 'auto'

    if cfg.mode == 'train':
        worker = main_worker
    elif cfg.mode == 'test':
        worker = test_worker
    else:
        raise ValueError(f'worker not found for mode={cfg.mode}')

    from utils.launch import launch

    with logger.redirect_err(log_file=cfg.log_file):
        launch(worker, num_gpus_per_machine=num_gpus, num_machines=num_machines, machine_rank=machine_rank, dist_url=dist_url, args=(cfg,))

    return None


if __name__ == "__main__":
    main()
