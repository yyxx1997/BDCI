import numpy as np
import random
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import logging
import logging.config
from pathlib import Path
import ruamel.yaml as yaml


def read_yaml(path):
    config = yaml.load(open(path, 'r'), Loader=yaml.Loader)
    return config

def compose_new_attr(logger, logging_level, is_master=False):

    logging_origin = logger.__getattribute__(logging_level)

    def dist_logger(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            logging_origin(*args, **kwargs)

    logger.__setattr__(logging_level, dist_logger)


def setup_for_distributed_logger(logger, is_master=False):
    """
    This function disables logging when not in master process
    """
    levelToName = {50: 'CRITICAL', 40: 'ERROR', 30: 'WARNING', 20: 'INFO', 10: 'DEBUG'}
    for level, name in levelToName.items():
        compose_new_attr(logger, name.lower(), is_master)


def create_logger(config):
    """
    This function is responsible for creating the log folder,and returning the class used for logging operations.
    The log file is recorded according to the timing, and records each training process independently.
    https://zhuanlan.zhihu.com/p/476549020
    https://blog.csdn.net/dadaowuque/article/details/104527196
    https://www.cnblogs.com/liqi175/p/16557213.html

    args:
    --------
        config -> config dictionary which contains goal path and setting.
    
    returns:
    --------
        logger -> a handler that can perform log operations;
        # sub_output_path -> final output directory depend on timestamp.

    goal:
    --------
        /root -> name according to settings

            /sub_root -> name according to timing
            
                /checkpoints -> big files
                    ...

                tensorboard.logs

                example.log

                global_config.yaml
            ...
    """

    output_path = os.path.join(config.output_dir, time.strftime('%Y-%m-%d-%H-%M'))
    ckpt_output_path = os.path.join(output_path, 'checkpoints')
    config.output_dir = output_path

    if is_main_process():
        Path(output_path).mkdir(parents=True, exist_ok=True)
        Path(ckpt_output_path).mkdir(parents=True, exist_ok=True)
        yaml.dump(dict(config), open(os.path.join(config.output_dir, 'global_config.yaml'), 'w'))

    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    log_file = 'train.log'
    head = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(head)
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(output_path, log_file))
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logging_level = getattr(logging, config.logging_level, logging.DEBUG)
    ch.setLevel(logging_level)
    fh.setLevel(logging_level)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.setLevel(logging_level)

    setup_for_distributed_logger(logger, is_master=is_main_process())
    return logger

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, logging=None, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logging = logging if logging else print

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def summary(self, mode="avg"):
        """
        Determine how to count the variables in meters by specifying the mode.
        My annotation is of a kind of very exhautive numpydoc format docstring. 
        See: https://zhuanlan.zhihu.com/p/344543685

        args:
        --------
            mode -> ["avg", "global_avg", "max", "median", "total", "value"], Optional, by default "avg"
                    
        return:
        --------
            A summary string for multiple variables in MetricLogger.
        """
        summary_str = []
        for name, meter in self.meters.items():
            summary_str.append(
                "{}: {:.4f}".format(name, getattr(meter, mode))
            )
        return self.delimiter.join(summary_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logging(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    self.logging(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logging('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

@torch.no_grad()
def concat_all_gather(tensor):
    if is_dist_avail_and_initialized():
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output        
    else :
        return tensor
        
def setup_seed(seed=3407):
    # fix the seed for reproducibility, more details see:
    # https://www.zhihu.com/search?type=content&q=3407
    # https://blog.csdn.net/zxyhhjs2017/article/details/91348108
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # seed = seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    ...