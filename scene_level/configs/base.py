import re, os, sys
import inspect
import importlib
from .utils import _is_property, _is_method
# import addict

def _try_eval(s):
    try:
        s = eval(s)
    except:
        pass
    return s

class _None:
    pass

class _Base(type):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        return ''

class Base(metaclass=_Base):
    _cls = 'Config'
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        return ''

    # dict-like interface
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, item):
        setattr(self, key, item)
    def __contains__(self, key):
        return key in dir(self)

    # def __len__(self):
    #     return len(self.keys())
    # def __iter__(self):
    #     return (k for k in self.keys())

    def __repr__(self) -> str:
        return str(self.dict())

    # # pickle interface
    # def __getstate__(self):
    #     return vars(self)
    # def __setstate__(self, state):
    #     vars(self).update(state)

    def __eq__(self, value):
        if self.__class__ != value.__class__:
            return False
        k_list = dir(self)
        if k_list != dir(value):
            return False
        for k in k_list:
            v = getattr(self, k)
            if not _is_method(v) and v != getattr(value, k):
                return False
        return True

    def __init__(self, cfg=None, parse=False):
        # TODO: register mapping for callable if overrided ??? - invoke when trying to call?
        # self.__collisions = {k: None for k in dir(self) if not k.startswith('__') and _is_method(getattr(self, k))}
        if cfg:
            self.update(cfg, exclude=[])
        if parse:
            self.parse()
        # self.__exclude = ['pretty_text']

    def __keys__(self, exclude=[]):  # self-reserved .keys()
        exclude = [exclude] if isinstance(exclude, str) else exclude
        k_list = [k for k in dir(self) if not k.startswith('_') and k not in exclude and not _is_method(getattr(self, k))]
        if 'type' in k_list and k_list[0] != 'type':
            k_list.remove('type')
            k_list.insert(0, 'type')
        return k_list

    def keys(self, exclude=[]):  # enable ** operation on Base
        return self.__keys__(exclude=exclude)

    # Mapping-protocal also enabled by .keys() & .__getitem__
    # - enable dict conversion on Base
    def dict(self, exclude=[]):
        kv = {}
        k_list = self.__keys__(exclude=exclude)
        for k in k_list:
            v = getattr(self, k)
            if isinstance(v, Base):
                v = v.dict(exclude=exclude)
            kv[k] = v
        return kv

    def items(self, exclude=[]):
        # one-layer expanding (not converting sub-config into dict)
        kv = []
        k_list = self.__keys__(exclude=exclude)
        for k in k_list:
            v = getattr(self, k)
            kv.append((k, v))
        return kv

    def pop(self, key, default=_None):
        if default is not _None:
            v = getattr(self, key)
            try:
                delattr(self, key)
            except:
                v = default
        else:
            v = getattr(self, key)
            delattr(self, key)
        return v

    def _yaml_default_constructor(self, loader, cls, node):
        kv_dict = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            value = loader.construct_object(value_node, deep=True)
            kv_dict[key] = value
        return Base(kv_dict)

    def update(self, cfg, key=None, exclude=['name', '_idx_name', '_idx_name_pre']):
        exclude = exclude if exclude else []

        if key != None:  # store cfg as a sub-config
            setattr(self, key, Base(cfg))

        elif isinstance(cfg, str):  # path / dict in str
            if os.path.isfile(cfg) and any(cfg.endswith(i) for i in ['.yaml', '.yml']):
                # import ruamel.yaml
                # parser = ruamel.yaml.YAML()
                # cfg = dict(parser.load(cfg))
                import yaml
                loader = yaml.FullLoader
                loader.add_implicit_resolver(  # enable loading 1e-6 as float
                    u'tag:yaml.org,2002:float',
                    re.compile(u'''^(?:
                    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$''', re.X),
                    list(u'-+0123456789.'))
                yaml.add_multi_constructor('tag:yaml.org,2002:python/object:config', self._yaml_default_constructor)

                with open(cfg) as f:
                    cfg = dict(yaml.load(f, Loader=loader))

            elif os.path.isfile(cfg) and cfg.endswith('.py'):
                cfg_dir_path = os.path.dirname(cfg)
                cfg_dir = os.path.basename(cfg_dir_path)
                sys.dont_write_bytecode = True  # not creating __pycache__
                sys.path.insert(0, cfg_dir_path)  # adding path
                cfg_mod_name = os.path.basename(cfg)[:-3]  # module name from filename
                cfg = importlib.import_module(cfg_mod_name, package=cfg_dir)
                cfg = {k: v for k, v in cfg.__dict__.items() if not k.startswith('__')}
                sys.path = sys.path[1:]
                sys.dont_write_bytecode = False
                del sys.modules[cfg_mod_name]

            elif cfg.startswith('{') and cfg.endswith('}'):
                cfg = eval(cfg)
            else:
                # cfg = cfg.replace('\:', '：')  # ues chinese char to escape
                # cfg = dict([[i.strip().replce('：', ':') for i in t.split(':')] for t in cfg.split(',')])
                try:
                    cfg = dict([[i.strip() for i in t.replace('=', ':').split(':')] for t in cfg.split(',')])
                    cfg = {_try_eval(k): _try_eval(v) for k, v in cfg.items()}
                except Exception as e:
                    raise type(e)(f'When updating with cfg={cfg}') from e

            self.update(cfg, exclude=exclude)

        elif isinstance(cfg, dict):  # update from dict
            for k, v in cfg.items():
                if k in exclude:
                    pass
                elif _is_property(self, k) or _is_method(getattr(self, k)):
                    # print(f'{k} is a property / method in {self}', file=sys.stderr)
                    try:
                        setattr(self, k, v)
                    except:
                        raise KeyError(f'{k} is a property / method in {self}')
                elif isinstance(v, dict):  # nesting dict
                    if hasattr(self, k) and isinstance(getattr(self, k), dict):  # replace the dict (if it exists & is indeed a dict)
                        setattr(self, k, v)
                    elif hasattr(self, k) and isinstance(getattr(self, k), Base):  # update the sub-config
                        getattr(self, k).update(v, exclude=exclude)
                    else:  # extend to be a sub-config
                        self.update(v, key=k, exclude=exclude)
                elif isinstance(k, str) and '.' in k:
                    k = k.split('.')
                    kk = k[0]
                    getattr(self, kk).update({'.'.join(k[1:]): v})
                elif isinstance(v, (list, tuple)):
                    # NOTE: one-layer expansion of list/tuple - can be extended to be recursive...
                    v = v.__class__(Base(vv) if isinstance(vv, dict) else vv for vv in v)
                    setattr(self, k, v)
                else:
                    setattr(self, k, v)

        else:  # update from another cfg
            for attr in [i for i in dir(cfg) if not i.startswith('_') and i not in self._attr_dict]:
                if not _is_property(self, attr) and not _is_method(getattr(cfg, attr)) and attr not in exclude:
                    setattr(self, attr, getattr(cfg, attr))
        return self

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def parse(self):
        pass

    def freeze(self):
        cfg = Base()
        cfg.update(self, exclude=[])  # turn all property into frozen args
        return cfg

    def dump(self, stream=None, exclude=[], **kwargs):
        if isinstance(stream, str):
            if not stream.lower().endswith('.yaml'):
                stream = f'{stream}.yaml'
            stream = open(stream, 'w')
        import yaml
        return yaml.dump(self.dict(exclude=exclude), stream, **kwargs)

    def code(self):
        from yapf.yapflib.yapf_api import FormatCode
        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(
                    f"dict({_indent(_format_dict(v_), indent)})," for v_ in v
                ).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self.dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style="facebook",
            column_limit=120,
            # blank_line_before_nested_class_or_def=True,
            # split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style)

        return text



class Config(Base):
    mode = 'train'

    @property
    def gpu_devices(self):
        if isinstance(self.gpus, int) or self.gpus.isdigit():
            cuda_dev = [str(i) for i in range(int(self.gpus))]
        else:
            assert ',' in self.gpus, f'unexpected cfg.gpus = {self.gpus}'
            cuda_dev = [i.strip() for i in self.gpus.split(',') if i.strip()]
        cuda_dev = ','.join(cuda_dev)
        return cuda_dev
    @property
    def gpu_num(self):
        return len([i for i in self.gpu_devices.split(',') if i])

    def __init__(self, cfg=None, parse=True):
        super(Config, self).__init__(cfg, parse)
