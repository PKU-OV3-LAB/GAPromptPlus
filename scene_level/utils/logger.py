import os, sys
import logging
import traceback
import numpy as np
from termcolor import colored


def print_dict(d, prefix='', except_k=[], fn=None, head=None, dict_type=(dict,), list_type=(list, tuple), expand_len=120):
    # NOTE: restruct to return str & print at last: utilize traverse_list + traverse_dict for printing
    # from utils.storage import traverse_list, traverse_dict
    if head is not None:
        d = {head: d}
    for k, v in d.items():
        if k in except_k:
            continue
        if isinstance(d[k], dict_type):
            print(f'{prefix}{str(k)}:')
            print_dict(d[k], prefix=f'{prefix}\t', except_k=except_k, fn=fn, expand_len=120)
        else:
            if fn:
                rst = None
                try:
                    if isinstance(v, list_type):
                        rst = v.__class__([fn(vv) for vv in v])
                    else:
                        rst = fn(v)
                except:
                    pass
                v = rst if rst else v
            line = f'{prefix}{str(k)}\t{str(v)}'
            if isinstance(v, list_type) and expand_len and len(str(line)) > expand_len:  # overlong
                line_pre = f'{prefix}{str(k)}\t' + ('[' if isinstance(v, list) else '(')
                line_post = f'\n{prefix}\t' + (']' if isinstance(v, list) else ')')
                if set(dict_type).issuperset(set([type(s) for s in v])):  # all dict in list
                    print(line_pre)
                    for s in v[:-1]:
                        print_dict(s, prefix=f'{prefix}\t\t')
                        print(f'{prefix}\t\t,')
                    print_dict(v[-1], prefix=f'{prefix}\t\t')
                    line = line_post
                else:
                    line =  line_pre + f'\n{prefix}\t\t'.join([''] + [str(s) for s in v]) + line_post

            print(line)

def print_table(t, prefix='', sep='  '):  # assume a 2D-list
    max_len = np.array([[len(str(ii)) for ii in l] for l in t], dtype=int).max(axis=0)
    for line in t:
        print(prefix + sep.join([str(ii) + ' ' * (max_len[i] - len(str(ii))) for i, ii in enumerate(line)]))

class StreamToLogger(object):
    def __init__(self, logger: logging.Logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.is_open = True
        self.linebuf = []

    def write(self, buf):
        if not self.is_open:
            raise IOError(f'closed StreamToLogger')
        # - try to have the correct newlines
        if buf.endswith('\n'):
            self.linebuf.append(buf.rstrip('\n'))
            self.logger.log(self.log_level, ''.join(self.linebuf))
            self.linebuf = []
        else:
            self.linebuf.append(buf)
        # for line in buf.rstrip().splitlines():
        #     self.logger.log(self.log_level, line.rstrip())
    def flush(self):
        pass
    def close(self):
        self.is_open = False

class redirect_io(object):
    def __init__(self, log_file, filemode='w', debug=False):
        self.log_file = log_file
        self.debug = debug
        self.filemode = filemode
    def __enter__(self):
        if self.debug or not self.log_file:
            return
        self.stdout, self.stderr = sys.stdout, sys.stderr

        if isinstance(self.log_file, str):
            self.log_file = open(self.log_file, self.filemode)
        elif isinstance(self.log_file, logging.Logger):
            self.log_file = StreamToLogger(self.log_file, logging.INFO)
        else:
            pass
        sys.stdout = sys.stderr = self.log_file

    def __exit__(self, exc_type, exc_value, tb):
        if self.debug or not self.log_file:
            return
        if sys.exc_info() != (None, None, None):
            traceback.print_exc()
        self.log_file.close()
        sys.stdout, sys.stderr = self.stdout, self.stderr

class redirect_err(object):
    """ context manager as convenient try-except-finally
    """
    def __init__(self, log_file=None, final=None):
        self.log_file = log_file
        # self.debug = debug
        self.final = final
        if final is not None:
            assert callable(final), f'final={final} is not callable'

    def __enter__(self):
        # try - entering
        pass

    def __exit__(self, exc_type, exc_value, tb):
        # except - logging
        if (exc_type, exc_value, tb) != (None, None, None) and self.log_file:
            log_file = open(self.log_file, 'a') if os.path.exists(self.log_file) else open(self.log_file, 'w')
            traceback.print_exc(file=log_file)
            log_file.close()

        # finally - cleanup, eg env.cleanup
        if self.final is not None:
            self.final()
        return


root_status = 0
logger_initialized = {}

class SimplifiedFormatter(logging.Formatter):

    def __init__(self, fmt="%(message)s", datefmt=None, style='%', color=False):
        levelname = '%(levelname)s'
        default_fmt = "[%(asctime)s %(levelname)s %(filename)s: %(module)s: line %(lineno)d %(process)d] %(message)s"
        if color:  # add color on demand
            color = color if isinstance(color, str) else 'red'
            levelname_color = colored(levelname, color)
            fmt.replace(levelname, levelname_color)
            default_fmt.replace(levelname, levelname_color)

        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.info_formatter = super()
        self.default_formatter = logging.Formatter(default_fmt)

        return

    def format(self, record):
        if record.levelno == logging.INFO:
            formatter = self.info_formatter
        # elif record.levelno == logging.DEBUG:
        #     formatter = self.dbg_formatter
        else:
            formatter = self.default_formatter
        record = formatter.format(record)

        # if self.use_tqdm:
        #     self.tqdm.write(record)
        return record


def get_logger(name, rank=0, log_file=None, log_level=logging.INFO, file_mode="a"):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'a'.
        color (bool): Colorful log output. Defaults to True

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(SimplifiedFormatter(color=True))
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    # if dist.is_available() and dist.is_initialized():
    #     rank = dist.get_rank()
    # else:
    #     rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(SimplifiedFormatter())
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def get_root_logger(log_file=None, log_level=logging.INFO, file_mode="a", **kwargs):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name.

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
        file_mode (str): File Mode of logger. (w or a)

    Returns:
        logging.Logger: The root logger.
    """
    logger = get_logger(
        name="root", log_file=log_file, log_level=log_level, file_mode=file_mode, **kwargs
    )
    return logger
