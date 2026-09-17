import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
from tools import pretrain_run_net as pretrain
from tools import test_run_net as test_net
from tools import finetune_run_net as finetune
from tools import module_run_net as module_tune
from tools import finetune_seg_run_net as finetune_seg
from tools import module_seg_run_net as module_seg_tune
from tools import module_seg_test_net as module_seg_test
from tools import module_seman_seg_block_run_net as module_seman_seg_block_tune
from tools import finetune_seman_seg_block_run_net as finetune_seman_seg_block
from tools import finetune_seman_seg_run_net as finetune_seman_seg
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import torch
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer

    train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    # config
    config = get_config(args, logger = logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size
        config.dataset.val.others.bs = config.total_bs // world_size
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs * 2
    # log
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # Older classification configs may not define task; preserve backward compatibility.
    task = getattr(config, "task", "classification")
    # run
    if getattr(config, 'evaluation_only', False) and not args.test:
        raise ValueError(
            'This legacy compatibility config is evaluation-only. '
            'Use --test --ckpts /path/to/fine_tuned_checkpoint.pth.'
        )
    if args.test:
        if task == "segmentation":
            module_seg_test(args, config, train_writer, val_writer)
        else:
            test_net(args, config, train_writer, val_writer)
    else:
        if task == 'classification':
            if args.finetune_model:
                print('finetuning starts!')
                finetune(args, config, train_writer, val_writer)
            else:
                print('module tuning starts!')
                module_tune(args, config, train_writer, val_writer)
                # pretrain(args, config, train_writer, val_writer)
        elif task == 'segmentation':
            if args.finetune_model:
                print('finetuning for segmentation starts!')
                finetune_seg(args, config, train_writer, val_writer)
            else:
                print('module tuning for segmentation starts!')
                module_seg_tune(args, config, train_writer, val_writer)
        elif task == 'semantic_segmentation_block':
            if args.finetune_model:
                print('finetuning for segmentation starts!')
                finetune_seman_seg_block(args, config, train_writer, val_writer)
            else:
                print('module tuning for semantic segmentation starts!')
                module_seman_seg_block_tune(args, config, train_writer, val_writer)
        elif task == 'semantic_segmentation':
            if args.finetune_model:
                print('finetuning for segmentation starts!')
                finetune_seman_seg(args, config, train_writer, val_writer)
            else:
                print('module tuning for semantic segmentation starts!')


if __name__ == '__main__':
    main()
