import yaml
from easydict import EasyDict
import os
import shutil
from pathlib import Path
from .logger import print_log

OBJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {'DATA_PATH', 'PC_PATH', 'GS_PATH', 'ROOT', 'DATA_ROOT', 'dataPath', 'ckpts', 'encoder_weight'}

def _resolve_config_path(value, config_dir):
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if path.is_absolute():
        return str(path)
    local = (config_dir / path).resolve()
    project = (OBJECT_ROOT / path).resolve()
    return str(local if local.exists() else project)

def _resolve_runtime_paths(value, config_dir, key=None):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            value[child_key] = _resolve_runtime_paths(child_value, config_dir, child_key)
    elif isinstance(value, list):
        value = [_resolve_runtime_paths(item, config_dir) for item in value]
    elif key in PATH_KEYS and isinstance(value, str):
        value = _resolve_config_path(value, config_dir)
    return value

def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config, config_dir):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                base_path = Path(_resolve_config_path(new_config['_base_'], config_dir))
                with base_path.open('r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val, base_path.parent)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, config_dir)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    cfg_path = Path(cfg_file).expanduser().resolve()
    with cfg_path.open('r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config, config_dir=cfg_path.parent)
    return _resolve_runtime_paths(config, cfg_path.parent)

def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    shutil.copy2(args.config, config_path)
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger = logger )
