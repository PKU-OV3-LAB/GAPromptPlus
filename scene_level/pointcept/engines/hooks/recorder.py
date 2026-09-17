"""
Default Hook for Tester
"""

import os
import torch
import numpy as np

from collections import defaultdict
from sklearn.metrics import confusion_matrix

from utils.storage import dict_list
from utils.metrics import metrics_from_confusions

from .builder import HOOKS

class HookRecorder(object):
    """
    Base class for hooks that can be registered with :class:`TesterBase`.
    """

    tester = None  # A weak reference to the tester object.

    def record_init(self, record):
        # - init record
        pass

    def record_sample(self, data_name, input_dict, output, pred, record):
        # - input/output of partial cloud / scene - batched & aug-ed for testing
        pass

    def record_cloud(self, data_name, data_dict, result_dict, record):
        # - data/result of the whole cloud / scene
        pass

    def record_dataset(self, record):
        # - record of full dataset (val/test set) - after sync
        pass


@HOOKS.register_module()
class LatentRecorder(HookRecorder):
    def __init__(self, target, save=False):
        super().__init__()
        self.save = save

        self.cfg = None
        self.logger = None
        self.target = target

    def record_init(self, record: dict):
        self.cfg = self.tester.cfg
        self.logger = self.tester.logger

        # import torch
        import torch.nn as nn
        from pointcept.models.modules import PointModule
        model: PointModule = self.tester.model

        import re
        from functools import partial

        valid = False
        if self.target == 'prefix':
            from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import SerializedAttention
            from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import SerializedAttention_base, SerializedAttention_sonata
            for name, module in model.named_modules():
                if isinstance(module, (SerializedAttention, SerializedAttention_base, SerializedAttention_sonata)):
                    module.forward = partial(module.forward, _collect_attn=True)  # SerializedAttention.forward_prefix
                    module.register_forward_hook(partial(self.forward_pre_hook, stage=module.stage))
                    valid = True

        elif self.target == 'alattn_in':
            from pointcept.models.blocks.encoding import LatAttnAdapter
            for name, module in model.named_modules():
                if isinstance(module, LatAttnAdapter):
                    module.register_forward_pre_hook(partial(self.forward_pre_hook, stage=module.stage))
                    valid = True
        else:
            raise
        assert valid, f"not matching any module"

        self.cloud_record = dict()  # per-cloud record => per-stage
        self.forward_record = self._init_forward_record()
        return

    def _init_forward_record(self):
        return defaultdict(lambda: {  # per-forward record
            'latents' : 0,
            'forward_cnt' : 0,
            'feat': 0,
            'offset': None,
        })

    @torch.inference_mode
    def forward_pre_hook(self, module, input, output=None, stage=None):
        # assume batch=1
        from pointcept.models.modules import Point
        if stage is None:
            stage = module.stage

        if self.target == 'prefix':  # SerializedAttention.forward_prefix - forward hook
            point = output
            feat = point._feat_q
            latents = point._latents_k

        elif self.target == 'alattn_in':  # LatAttnAdapter
            point = module.down(Point(input[0]))

            module = module.latent_attn  # LatentAttentions
            latents = module.latents if module.latents is not None else point.latents

            feat = point.feat
            module = module.cross_in  # CrossAttention
            if module.pre_norm:
                latents = module.norm(latents)
                feat = module.norm_kv(feat)
            if module.q is not None:
                latents = module.q(latents)
            if module.kv is not None:
                feat = module.kv(feat)
            feat = feat[:, :latents.shape[-1]]

        else:
            raise
        assert len(point.offset) == 1

        feat = point.feat
        _point = point
        while "pooling_parent" in _point.keys():
            # - back-prop to input point cloud
            assert "pooling_inverse" in _point.keys()
            inverse = _point["pooling_inverse"]
            _point = _point["pooling_parent"]
            feat = feat[inverse]
        feat = feat.detach().cpu()

        # NOTE: merging blks within same stage...
        # aggregate first
        self.forward_record[stage]['feat'] += feat
        self.forward_record[stage]['latents'] += latents
        self.forward_record[stage]['forward_cnt'] += 1
        # self.forward_record[stage]['offset'] = point.offset  # use the stage-0 offset
        # not modifying input
        return

    def record_sample(self, data_name, data_size, input_dict, pred_part, record):
        cloud_record = self.cloud_record  # per-cloud record

        for stage, stage_sample in self.forward_record.items():
            if stage not in cloud_record:
                cloud_record[stage] = dict(latents=0, forward_cnt=0)
            stage_record = cloud_record[stage]  # per-stage record

            stage_record['latents'] += stage_sample['latents']
            stage_record['forward_cnt'] += stage_sample['forward_cnt']

            # aggregate to original cloud
            feat = stage_sample['feat'].cpu()
            offset = input_dict['offset'].cpu()
            if 'feat' not in stage_record:
                stage_record['feat'] = torch.zeros([data_size, feat.shape[-1]], device='cpu')
                stage_record['feat_cnt'] = torch.zeros([data_size, 1], device='cpu')

            bs = 0
            idx_part = input_dict['index'].cpu()
            for be in offset:
                stage_record['feat'][idx_part[bs:be], :] += feat[bs:be]
                stage_record['feat_cnt'][idx_part[bs:be], :] += 1
                bs = be
            self.forward_record = self._init_forward_record()
        return

    def record_cloud(self, data_name, data_dict, result_dict, record):
        import torch
        import torch.nn.functional as F
        logger = self.logger
        logger.info(f'recording: {data_name}')

        save_dict = dict()
        for stage_n, stage_record in self.cloud_record.items():
            assert (stage_record['feat_cnt'] > 0).all()

            feat = (stage_record['feat'] / stage_record['feat_cnt']).cpu()  # [N, d]
            latents = (stage_record['latents'] / stage_record['forward_cnt']).cpu()  # [M, d] - M < d << N

            # dot, cos
            dist = [feat @ latents.T, F.normalize(feat) @ F.normalize(latents).T]  # [N, M, #similarity (distance)]
            dist_names = ['dot', 'cos']

            save_dict[stage_n] = {k: v for k, v in zip(dist_names, dist)}
            logger.info(f'\tstage_n={stage_n}: feat={feat.shape}, latents={latents.shape}')

        if self.save or 'latents' in self.cfg.test.save:
            save_dict['name'] = 'latents'
            save_path = os.path.join(self.cfg.test.save_path, f'{data_name}_latents.pth')
            os.makedirs(self.cfg.test.save_path, exist_ok=True)
            torch.save(save_dict, save_path)
        else:
            record[data_name]['latents'] = save_dict

        self.cloud_record = dict()  # re-init
        return

@HOOKS.register_module()
class EigRecorder(HookRecorder):

    def record_cloud(self, data_name, data_dict, result_dict, record):
        cfg = self.tester.cfg
        pred = result_dict["pred"]
        segment = result_dict["segment"]
        if "eigval" in data_dict:
            eigval = data_dict["eigval"]
        else:
            eigval = np.load(os.path.join(cfg.data.test.data_root, cfg.data.test.split, data_name, "eigval.npy"))

        eigval_dict = dict(
            L=(eigval[:, 0] - eigval[:, 1]) / eigval[:, 0],
            Sig1=eigval[:, 0] / eigval.sum(-1),
        )
        ncls = cfg.data.num_classes
        eigval_rst = defaultdict(dict)
        for k, v in eigval_dict.items():
            for thr_n in [0.01, 0.1, "mean"]:
                if isinstance(thr_n, float):
                    thr = thr_n
                    thr_n = str(thr_n)
                elif thr_n == "mean":
                    thr = v.mean()
                else:
                    raise ValueError(f"not support thr_n/thr={thr_n}/{thr}")
                eigval_rst[f"eig-{k}"].update({
                    f">{thr_n}": confusion_matrix(segment[v > thr], pred[v > thr], labels=np.arange(ncls)),
                    f"<{thr_n}": confusion_matrix(segment[v <= thr], pred[v <= thr], labels=np.arange(ncls)),
                })
            record[data_name].update(eigval_rst)
        return

    def record_dataset(self, record):
        eig_list = [{kk: vv for kk, vv in v.items() if kk.startswith("eig-")} for k, v in record.items()]
        eig_rst = dict_list(eig_list)

        logger = self.tester.logger
        logger.info("--- Eig")
        eig_str_max = max(len(k) for k in eig_rst)
        for eig_k, rst in eig_rst.items():
            info = "\t{eig_k:<{eig_str_max}}\t".format(eig_k=eig_k, eig_str_max=eig_str_max)
            for mask_k, conf_list in rst.items():
                miou = metrics_from_confusions(sum(conf_list))["mIoU"]
                info += f"{mask_k} : {miou:.3f}\t"
            info.rstrip("\t")
            logger.info(info)
        return
