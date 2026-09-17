"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import re
import os
import shutil
import time
import gc
import wandb
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.scheduler import CosineScheduler
import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1, freq=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0
        self._freq = freq

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._epoch_start = time.perf_counter()
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        if "iter_info" in self.trainer.comm_info.keys() and (self.trainer.comm_info["iter"] + 1) % self._freq == 0:
            remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
            info = (
                "data {data_time_avg:.3f} "
                "batch {batch_time_avg:.3f} "
                "remain {remain_time} ".format(
                    # data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    # batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            # if torch.cuda.is_available():
            #     max_mem = torch.cuda.max_memory_allocated() / 1024.0 ** 3
            #     info += f"max_mem {max_mem:>4.1f}g "
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()

    def after_epoch(self):
        epoch_time = (time.perf_counter() - self._epoch_start) / 60
        self.trainer.comm_info["epoch_info"] += f"epoch_time: {epoch_time:.1f}min "

    def after_train(self):
        train_time = (time.perf_counter() - self._start_time) / 60 ** 2
        self.trainer.comm_info["train_info"] += f"train_time: {train_time:.1f}h "

@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self, freq=1):
        self.curr_iter = 0
        self.model_output_keys = []
        self.freq = freq

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.trainer.comm_info["epoch_info"] = ""
        self.trainer.comm_info["train_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("params/*", step_metric="Iter")
            wandb.define_metric("train_batch/*", step_metric="Iter")
            wandb.define_metric("train/*", step_metric="Epoch")

    def before_epoch(self):
        self.trainer.logger.info(f"\n****EPOCH {self.trainer.epoch+1}****")

    def before_step(self):
        self.curr_iter += 1
        if (self.trainer.comm_info["iter"] + 1) % self.freq == 0:
            # MSC pretrain do not have offset information. Comment the code for support MSC
            # info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] " \
            #        "Scan {batch_size} ({points_num}) ".format(
            #     epoch=self.trainer.epoch + 1, max_epoch=self.trainer.max_epoch,
            #     iter=self.trainer.comm_info["iter"], max_iter=len(self.trainer.train_loader),
            #     batch_size=len(self.trainer.comm_info["input_dict"]["offset"]),
            #     points_num=self.trainer.comm_info["input_dict"]["offset"][-1]
            # )
            max_epoch = str(self.trainer.max_epoch)
            max_iter = str(len(self.trainer.train_loader))
            info = "Train: [{epoch:>{fill_ep}}/{max_epoch}][{iter:>{fill_it}}/{max_iter}] ".format(
                epoch=self.trainer.epoch + 1,
                max_epoch=max_epoch,
                fill_ep=len(max_epoch),
                iter=self.trainer.comm_info["iter"] + 1,
                max_iter=max_iter,
                fill_it=len(max_iter),
            )
            self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            # model_output_dict = {k: v for k, v in model_output_dict.items() if len(v.shape) == 0 or list(v.shape) == [1]}
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        if (self.trainer.comm_info["iter"] + 1) % self.freq == 0:
            for key in self.model_output_keys:
                self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                    key=key, value=self.trainer.storage.history(key).val
                )
            lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
            self.trainer.comm_info["iter_info"] += "lr: {lr:f}".format(lr=round(lr, 12))
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])
            self.trainer.comm_info["iter_info"] = ""  # reset iter info

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {"Iter": self.curr_iter, "params/lr": lr}, step=self.curr_iter
                )
                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Iter": self.curr_iter,
                            f"train_batch/{key}": self.trainer.storage.history(key).val,
                        },
                        step=wandb.run.step,
                    )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        epoch_info += self.trainer.comm_info["epoch_info"]
        self.trainer.logger.info(epoch_info.strip())
        self.trainer.comm_info["epoch_info"] = ""
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )
            if self.trainer.cfg.enable_wandb:
                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Epoch": self.trainer.epoch + 1,
                            f"train/{key}": self.trainer.storage.history(key).avg,
                        },
                        step=wandb.run.step,
                    )

    def after_train(self):
        if self.trainer.comm_info["train_info"]:
            train_info = "Train finish: " + self.trainer.comm_info["train_info"]
            self.trainer.logger.info(train_info.strip())

@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None, save_max=None, cleanup=False):
        self.save_freq = save_freq  # None or int, None indicate only save model last
        self.save_max = save_max  # max ckpt to keep, if save_freq enabled
        self.cleanup = cleanup  # True to delete the model last

    def _get_save_path(self, name=None):
        name = name if name else "model_last.pth"
        return os.path.join(self.trainer.cfg.save_path, "model", name)

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        f"Epoch {self.trainer.epoch+1}:\nBest validation {current_metric_name} updated to: {current_metric_value:.4f}"
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(current_metric_name, self.trainer.best_metric_value)
                )

            filename = self._get_save_path()  # model_last
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            if not self.cleanup:
                self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                self.trainer.logger.info("Best saved.")
                shutil.copyfile(
                    filename,
                    self._get_save_path(name="model_best.pth")
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0 and self.trainer.epoch>0.6*self.trainer.max_epoch:
                save_name = self._get_save_path(name=f"model_{self.trainer.epoch + 1}.pth")
                shutil.copyfile(filename, save_name)
                if self.save_max:
                    # delete exceeding snapshots
                    name_list = os.path.listdir(os.path.dirname(save_name))
                    name_list = [i for i in name_list if re.fullmatch(r"epoch_\d+\.pth", i)]
                    if len(name_list) > self.save_max:
                        for name in sorted(name_list, key=lambda x: int(re.search(r"\d+", x).group())):
                            os.remove(self._get_save_path(name=name))

    def after_train(self):
        if is_main_process() and self.cleanup:
            filename = self._get_save_path()
            os.remove(filename)


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, skip=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.skip = re.compile(skip if isinstance(skip, str) else "|".join(skip)) if skip else False
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight:
            assert os.path.isfile(self.trainer.cfg.weight), f'weight not found: {self.trainer.cfg.weight}'
            path = self.trainer.cfg.weight
            if os.path.islink(path):
                path = f"{path} -> {os.path.relpath(os.path.realpath(path))}"
            self.trainer.logger.info(f"Loading weight at: {path}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            weight_skipped = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                if self.skip and self.skip.search(key):
                    weight_skipped[key] = value
                    continue
                weight[key] = value
            ckpt_keys = list(weight.keys())
            has_backbone_prefix = any(k.startswith("backbone.") for k in ckpt_keys)
            missing_keys, unexpected_keys = [], []

            if has_backbone_prefix:
                missing_keys, unexpected_keys = self.trainer.model.load_state_dict(weight, strict=self.strict)
                self.trainer.logger.info("=== Load full finetuned checkpoint ===")
            else:
                missing_keys, unexpected_keys = self.trainer.model.backbone.load_state_dict(weight, strict=self.strict)
                self.trainer.logger.info("=== Load pretrain backbone-only checkpoint ===")

            self.trainer.logger.info("\n\t".join(["Missing keys:", *missing_keys]))
            self.trainer.logger.info("\n\t".join(["Unexpected keys:", *unexpected_keys]))
            if weight_skipped:
                self.trainer.logger.info("\n\t".join(["Skipped keys:", *weight_skipped.keys()]))
            if self.trainer.cfg.resume:
                self.trainer.logger.info(f"Resuming train at eval epoch: {checkpoint['epoch']}")
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
        # summarize the trainable and frozen parameters
        summary_parameters(self.trainer.model, detail=False, logger=self.trainer.logger)


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    # voting-based eval
    def __init__(self, test_model=None, test_last=False, verbose=2, save=False):
        if test_model is None:  # default to "best", test_last for compatibility
            test_model = "last" if test_last else "best"
        test_model = str(test_model)
        assert test_model in ["best", "last"] or test_model.isdigit(), f"invalid test_model={test_model}"
        self.test_model = test_model
        self.verbose = verbose
        self.save = save

    def after_train(self):
        if self.verbose > 1:
            self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>")
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        split = cfg.data.test.split
        log_file = os.path.join(self.trainer.cfg.save_path, f"log_{split}.txt_{self.test_model}")
        from pointcept.utils.logger import get_logger
        logger = get_logger(f"{split}_{self.test_model}", log_file=log_file)  # new logger (diff prefix with root logger)
        verbose = cfg.test.verbose if cfg.test.verbose != "" else self.verbose
        save = cfg.test.save if cfg.test.save != "" else self.save

        from pointcept.engines.test import TESTERS
        test_cfg = dict(cfg.test, cfg=cfg, model=self.trainer.model, logger=logger, verbose=verbose, save=save)
        tester = TESTERS.build(test_cfg)
        if self.test_model == "last":
            pass
        else:
            model_path = os.path.join(
                self.trainer.cfg.save_path, "model", f"model_{self.test_model}.pth"
            )
            checkpoint = torch.load(model_path, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            tester.model.load_state_dict(weight, strict=True)
        if self.verbose:
            test_msg = f"=> Testing on model_{self.test_model} ..."
            logger.info(test_msg)
            self.trainer.logger.info(test_msg)
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.reset_accumulated_memory_stats()
        # torch.cuda.synchronize()
        # baseline_MB = torch.cuda.memory_allocated()

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            # with torch.enable_grad():
            with torch.inference_mode():
                with record_function("model_forward"):
                    output_dict = self.trainer.model(input_dict)
                    loss = output_dict["loss"]
                    # loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )

        # peak_MB = torch.cuda.max_memory_allocated()
        # self.trainer.logger.info(f"[overhead] Baseline params: {baseline_MB / 1024 ** 3:.3f}G")
        # self.trainer.logger.info(f"[overhead] Peak during fwd: {peak_MB / 1024 ** 3:.3f}G")
        # self.trainer.logger.info(f"[overhead] Activation extr: {(peak_MB - baseline_MB) / 1024 ** 3:.3f}G")

        # stats = torch.cuda.memory_stats()
        # total_flow = stats["allocated_bytes.all.allocated"]
        # self.trainer.logger.info(f"[overhead] Total alloc  mem: {total_flow / 1024 ** 3:.3f}G")  # total mem overhead

        # cuda_ms = None
        # for item in prof.key_averages():
        #         if item.key == "ProfilerStep*":
        #             # cuda_time_total is in micro-seconds
        #             cuda_ms = item.cuda_time_total / item.count / 1000.0  # → milliseconds
        #             break
        # self.trainer.logger.info(f"[overhead]: cuda_ms={cuda_ms:.3f}ms")

        prof.stop()
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()

def summary_parameters(model, detail=False, logger = None):
    if detail:
        print('>> Trainable Parameters:')
        trainable_paramters = [(str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel())) for n, v in model.named_parameters() if v.requires_grad and not getattr(v, '_is_frozen', False)]
        max_lens = [max([len(item) + 4 for item in col]) for col in zip(*trainable_paramters)]
        raw_format = '|' + '|'.join(['{{:{}s}}'.format(max_len) for max_len in max_lens]) + '|'
        raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
        print(raw_split)
        print(raw_format.format('Name', 'Dtype', 'Shape', '#Params'))
        print(raw_split)
        for name, dtype, shape, number in trainable_paramters:
            print(raw_format.format(name, dtype, shape, number))
            print(raw_split)

    num_trainable_params = sum([v.numel() for v in model.parameters() if v.requires_grad])
    num_trainable_fixed_params = sum([v.numel() for v in model.parameters() if v.requires_grad and getattr(v, '_is_frozen', False)])
    total_params = sum([v.numel() for v in model.parameters()])
    non_trainable_params = total_params - num_trainable_params
    if logger:
        logger.info('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format('# TrainingParams:', (num_trainable_params-num_trainable_fixed_params) / (1e6), (num_trainable_params-num_trainable_fixed_params) / (1e3)))
        logger.info('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format('# TrainableParams:', num_trainable_params / (1e6), num_trainable_params / (1e3)))
        logger.info('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1e6)))
        logger.info('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1e6)))
        logger.info('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', (num_trainable_params-num_trainable_fixed_params) / total_params * 100.))
        logger.info('\n')
    else:
        print('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format('# TrainingParams:', (num_trainable_params-num_trainable_fixed_params) / (1e6), (num_trainable_params-num_trainable_fixed_params) / (1e3)))
        print('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format('# TrainableParams:', num_trainable_params / (1e6), num_trainable_params / (1e3)))
        print('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1e6)))
        print('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1e6)))
        print('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', (num_trainable_params-num_trainable_fixed_params) / total_params * 100.))
        print('\n')
