import torch
import torch.nn as nn
from tools import builder
from utils import Metric, misc, dist_utils
import time
from utils.logger import *
import os
import ipdb
import numpy as np
from datasets import data_transforms
from datasets.data_util import get_features_by_keys
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from utils.config import cfg_from_yaml_file
from tqdm import tqdm
import sys
from utils import provider
from utils.Metric import AverageMeter, ConfusionMatrix, get_mious
from datasets.S3DISDataset import S3DISDataset
from datasets.transforms import build_transforms_from_cfg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    config.dataset.val.others.bs=1
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)
    weights = torch.Tensor(train_dataloader.dataset.labelweights).cuda()

    # train_dataloader.dataset.transform = build_transforms_from_cfg(config.dataset.train.others.split, config.datatransforms)
    # test_dataloader.dataset.transform = build_transforms_from_cfg(config.dataset.test.others.split, config.datatransforms)

    num_classes = test_dataloader.dataset.num_classes if hasattr(test_dataloader.dataset, 'num_classes') else None

    config.classes = test_dataloader.dataset.classes if hasattr(test_dataloader.dataset, 'classes') else np.arange(num_classes)
    config.cmap = np.array(test_dataloader.dataset.cmap) if hasattr(test_dataloader.dataset, 'cmap') else None

    # build model
    classifier = builder.model_builder(config.model)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2) #weight=weights
    classifier.apply(inplace_relu)
    # print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0


    if args.ckpts is not None:
        classifier.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Training from scratch', logger = logger)

    if args.use_gpu:
        classifier.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        classifier = nn.DataParallel(classifier).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(classifier, config)

    from utils.misc import summary_parameters
    summary_parameters(classifier, logger=logger)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_iou = 0
    time_sec_tot = 0.
    epoch_start_time = time.time()

    # test_metrics = validate(logger, num_part, classifier, test_dataloader, num_classes, config)
    import shutil
    shutil.copy('models/Point_MAE_sem_segment.py', str(args.experiment_path))

    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    total_iter = 0

    test_metrics = validate(logger, classifier, test_dataloader, num_classes, config, criterion)
    # training
    classifier.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        classifier.train()

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        classifier.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        mean_correct = []
        print_log('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, config.max_epoch), logger = logger)
        '''Adjust learning rate and BN momentum'''
        classifier = classifier.train()
        num_iter = 0
        loss_meter = AverageMeter()
        cm = ConfusionMatrix(num_classes=num_classes, ignore_index=config.get('ignore_index',None))
        '''learning one epoch'''
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            num_iter += 1
            n_itr = epoch * n_batches + batch_idx
            data_time.update(time.time() - batch_start_time)
            keys = data.keys() if callable(data.keys) else data.keys
            for key in keys:
                data[key] = data[key].cuda(non_blocking=True)
            # data['x'] = get_features_by_keys(data, config.feature_keys)
            target = data['y']
            seg_pred, contrast_loss = classifier(data['pos'], features=data, contrast=True, target=target)
            pred_choice = seg_pred.max(-1)[1]

            acc = pred_choice.eq(target).cpu().sum()
            mean_correct.append(acc.item() / (pred_choice.shape[0] * pred_choice.shape[1]))

            loss = criterion(seg_pred.transpose(1,2), target)
            loss += contrast_loss
            loss.backward()

            cm.update(pred_choice, target)
            loss_meter.update(loss.detach().cpu().item())

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            # break

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        miou, macc, oa, ious, accs = cm.all_metrics()
        print_log('Train overall accuracy is: {:.3f}%'.format(oa), logger = logger)
        print_log('Train mIoU is: {:.3f}%'.format(miou), logger = logger)
        print_log('Train loss: {:.3f}'.format(loss_meter.val()), logger = logger)
        print_log('lr: {:.6f}'.format(optimizer.param_groups[0]['lr']), logger = logger)


        if epoch % args.val_freq == 0:
            test_metrics = validate(logger, classifier, test_dataloader, num_classes, config, criterion)
            # Save ckeckpoints
        if (test_metrics['mIoU'] >= best_class_avg_iou):
            best_metrics = test_metrics
            builder.save_checkpoint(classifier, optimizer, epoch, test_metrics, best_metrics, 'ckpt-best', args, logger = logger)
            print_log("--------------------------------------------------------------------------------------------", logger=logger)
        if test_metrics['overall_accuracy'] > best_acc:
            best_acc = test_metrics['overall_accuracy']
        if test_metrics['mIoU'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['mIoU']

        print_log('Best class avg mAcc is: {:.2f}%'.format(test_metrics['class_avg_accuracy']), logger = logger)
        print_log('Best overall accuracy is: {:.2f}%'.format(best_acc), logger = logger)
        print_log('Best class avg mIoU is: {:.2f}%'.format(best_class_avg_iou), logger = logger)

        builder.save_checkpoint(classifier, optimizer, epoch, test_metrics, best_metrics, 'ckpt-last', args, logger = logger)
        print_log('Epoch {:d} test Overall Accuracy: {:.2f}%   Class avg mIOU: {:.2f}%'.format(epoch + 1, test_metrics['overall_accuracy'], test_metrics['mIoU']), logger = logger)
        global_epoch += 1


def validate(logger, classifier, testDataLoader, num_classes, config, criterion):
    NUM_CLASSES = num_classes
    NUM_POINT = config.npoints
    BATCH_SIZE = 1
    test_metrics = {}
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        classifier = classifier.eval()

        cm = ConfusionMatrix(num_classes=num_classes, ignore_index=config.get('ignore_index',None))

        for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            keys = data.keys() if callable(data.keys) else data.keys
            for key in keys:
                data[key] = data[key].cuda(non_blocking=True)
            target = data['y']
            # data['x'] = get_features_by_keys(data, config.feature_keys)

            seg_pred = classifier(data['pos'], features=data)
            cm.update(seg_pred.argmax(dim=-1), target)

            pred_val = seg_pred.contiguous().cpu().data.numpy()
            # seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            loss = criterion(seg_pred.transpose(1,2), target)
            loss_sum += loss
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (target.shape[0]*target.shape[1])
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        tp, union, count = cm.tp, cm.union, cm.count
        miou, macc, oa, ious, accs = get_mious(tp, union, count)

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float)))    # np.float -> float
        print_log('eval mean loss: %f' % (loss_sum / float(num_batches)), logger=logger)
        print_log('[mIoU] eval point avg class IoU: {:.2f}%'.format(mIoU * 100.0), logger=logger)
        print_log('[OA] eval point accuracy: {:.2f}%'.format(total_correct / float(total_seen) * 100.0), logger=logger)
        print_log('[mAcc] eval point avg class acc: {:.2f}%'.format(np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float))) * 100.0), logger=logger)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class {:s} weight: {:.1f}%, IoU: {:.3f}% \n'.format(
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1]*100,
                total_correct_class[l] / float(total_iou_deno_class[l]) * 100.0)

        print_log(iou_per_class_str, logger=logger)

        test_metrics['overall_accuracy'] = oa
        test_metrics['class_avg_accuracy'] = macc
        test_metrics['mIoU'] = miou

    return test_metrics
