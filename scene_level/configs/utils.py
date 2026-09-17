import os, re, sys, types
from collections import abc

SCENE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
RUNTIME_PATH_KEYS = {'weight', 'data_root'}

def _resolve_runtime_paths(value, key=None):
    if isinstance(value, abc.Mapping):
        for child_key, child_value in list(value.items()):
            value[child_key] = _resolve_runtime_paths(child_value, child_key)
    elif value.__class__.__module__.endswith("configs.base"):
        for child_key, child_value in list(vars(value).items()):
            setattr(value, child_key, _resolve_runtime_paths(child_value, child_key))
    elif isinstance(value, list):
        value = [_resolve_runtime_paths(item) for item in value]
    elif isinstance(value, tuple):
        value = tuple(_resolve_runtime_paths(item) for item in value)
    elif key in RUNTIME_PATH_KEYS and isinstance(value, str) and value:
        value = os.path.realpath(os.path.join(SCENE_ROOT, os.path.expandvars(os.path.expanduser(value))))
    return value

def _is_property(obj, name):
    return isinstance(getattr(type(obj), name, None), property)
def _is_method(x):
    return type(x) in [types.MethodType, types.FunctionType, types.MethodWrapperType, types.BuiltinMethodType]

def is_config(cfg, base=None, mod=None):
    if mod != None and type(cfg) == str:
        if cfg.startswith('_'):
            return False
        cfg = getattr(mod, cfg)
    if base == None:
        assert mod != None, 'must provide either `base` (class Base) or `mod` (python module)'
        base = mod.Base
    return isinstance(cfg, base) or isinstance(type(cfg), base)  # config can be either class or instance

def log_config(config, title='', f_out=None, prefix='', base=None, _ignore=['_idx_name', '_idx_name_pre']):
    if f_out is None:
        f_out = sys.stdout
    if base is None:
        from .base import Base as base

    print(f'{prefix}<<< ======= {config._cls} ======= {title if title else config.name}', file=f_out)
    max_len = max([len(k) for k in dir(config) if not k.startswith('_')] + [0])
    for k, v in config.items():  # dir would sort
        # if k.startswith('_') or _is_method(getattr(config, k)):
        #     continue
        if _ignore and k in _ignore:
            continue

        cur_attr = v
        if isinstance(cur_attr, (list, tuple)) and len(str(cur_attr)) > 200:  # overlong list
            cur_attr = f'\n{prefix}\t\t'.join([''] + [str(s) for s in cur_attr]) + f'\n{prefix}\t'
            cur_attr = f'({cur_attr})' if isinstance(cur_attr, tuple) else f'[{cur_attr}]'

        if is_config(cur_attr, base=base):
            print('\t%s' % (prefix + k), file=f_out)
            log_config(cur_attr, f_out=f_out, prefix=prefix+'\t', base=base, _ignore=None)
        else:
            print('\t%s%s\t= %s' % (prefix + k, ' ' * (max_len-len(k)), str(cur_attr)), file=f_out)

    print(file=f_out, flush=True)

def load_config(cfg_path):
    from .base import Base, Config
    cfg = Base(cfg_path)
    if not cfg.name:
        cfg.name = os.path.splitext(os.path.basename(cfg_path))[0]

    if '_base_' in cfg and cfg['_base_']:
        cfg_base = Base()
        for cfg_b in cfg['_base_']:
            cfg_b = os.path.realpath(os.path.join(os.path.dirname(cfg_path), cfg_b))
            cfg_base.update(cfg_b)
        cfg = cfg_base.update(cfg)

    cfg = Config(cfg)
    return _resolve_runtime_paths(cfg)
