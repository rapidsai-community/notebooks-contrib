from datetime import datetime
import json
import glob
import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
import torch
from torch import nn
from torch.utils.data import DataLoader


ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ

def set_seed(seed=6750):
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    
def gmean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).agg(lambda x: gmean(list(x)))


def mean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).mean()


def load_model(model: nn.Module, path: Path, multi2single=False) -> Tuple:
    state = torch.load(str(path))
    if multi2single:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state, state['best_valid_loss']
    
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
       param_group['lr'] = lr

    return True


class ThreadingDataLoader(DataLoader):
    def __iter__(self):
        sample_iter = iter(self.batch_sampler)
        if self.num_workers == 0:
            for indices in sample_iter:
                yield self.collate_fn([self._get_item(i) for i in indices])
        else:
            prefetch = 1
            with ThreadPool(processes=self.num_workers) as pool:
                futures = []
                for indices in sample_iter:
                    futures.append([pool.apply_async(self._get_item, args=(i,))
                                    for i in indices])
                    if len(futures) > prefetch:
                        yield self.collate_fn([f.get() for f in futures.pop(0)])
                    # items = pool.map(lambda i: self.dataset[i], indices)
                    # yield self.collate_fn(items)
                for batch_futures in futures:
                    yield self.collate_fn([f.get() for f in batch_futures])

    def _get_item(self, i):
        return self.dataset[i]


def write_event(log, step: int, epoch=None, **data):
    data['step'] = step
    data['dt'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['epoch']=epoch
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def plot(*args, ymin=None, ymax=None, xmin=None, xmax=None, params=False,
         max_points=200, legend=True, title=None,
         print_keys=False, print_paths=False, plt=None, newfigure=True,
         x_scale=1):
    """
    Use in the notebook like this::

        %matplotlib inline
        from imet.utils import plot
        plot('./runs/oc2', './runs/oc1', 'loss', 'valid_loss')

    """
    import json_lines  # no available on Kaggle

    if plt is None:
        from matplotlib import pyplot as plt
    paths, keys = [], []
    for x in args:
        if x.startswith('.') or '/' in x:
            if '*' in x:
                paths.extend(glob.glob(x))
            else:
                paths.append(x)
        else:
            keys.append(x)
    if print_paths:
        print('Found paths: {}'.format(' '.join(sorted(paths))))
    if newfigure:
        plt.figure(figsize=(12, 8))
    keys = keys or ['loss', 'valid_loss']

    ylim_kw = {}
    if ymin is not None:
        ylim_kw['bottom'] = ymin
    if ymax is not None:
        ylim_kw['top'] = ymax
    if ylim_kw:
        plt.ylim(**ylim_kw)

    xlim_kw = {}
    if xmin is not None:
        xlim_kw['left'] = xmin
    if xmax is not None:
        xlim_kw['right'] = xmax
    if xlim_kw:
        plt.xlim(**xlim_kw)
    all_keys = set()
    for path in sorted(paths):
        path = Path(path)
        with json_lines.open(path / 'train.log', broken=True) as f:
            events = list(f)
        all_keys.update(k for e in events for k in e)
        for key in sorted(keys):
            xs, ys, ys_err = [], [], []
            for e in events:
                if key in e:
                    xs.append(e['step'] * x_scale)
                    ys.append(e[key])
                    std_key = key + '_std'
                    if std_key in e:
                        ys_err.append(e[std_key])
            if xs:
                if np.isnan(ys).any():
                    print('Warning: NaN {} for {}'.format(key, path))
                if len(xs) > 2 * max_points:
                    indices = (np.arange(0, len(xs) - 1, len(xs) / max_points)
                               .astype(np.int32))
                    xs = np.array(xs)[indices[1:]]
                    ys = _smooth(ys, indices)
                    if ys_err:
                        ys_err = _smooth(ys_err, indices)
                label = '{}: {}'.format(path, key)
                if label.startswith('_'):
                    label = ' ' + label
                if ys_err:
                    ys_err = 1.96 * np.array(ys_err)
                    plt.errorbar(xs, ys, yerr=ys_err,
                                 fmt='-o', capsize=5, capthick=2,
                                 label=label)
                else:
                    plt.plot(xs, ys, label=label)
                plt.legend()
    if newfigure:
        plt.grid()
    if legend:
        plt.legend()
    if title:
        plt.title(title)
    if print_keys:
        print('Found keys: {}'
              .format(', '.join(sorted(all_keys - {'step', 'dt'}))))


def _smooth(ys, indices):
    return [np.mean(ys[idx: indices[i + 1]])
            for i, idx in enumerate(indices[:-1])]
