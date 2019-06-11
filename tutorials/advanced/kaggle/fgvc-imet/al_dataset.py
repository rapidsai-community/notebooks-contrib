from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import ON_KAGGLE
import numpy as np
from tqdm import tqdm

N_CLASSES = 1103
DATA_ROOT = Path('../input/imet-2019-fgvc6' if ON_KAGGLE else '../input/imet-2019-fgvc6')

RESIZE = True
CROP = False

class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 transform: Callable,
                 over_sample = False,
                 debug: bool = True):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = transform
        self._debug = debug

        self._label = np.zeros((len(self._df),1103))
        attributes = self._df.attribute_ids.tolist()
        for i in range(len(self._df)):
            for c in attributes[i].split():
                self._label[i, int(c)] = 1

        if over_sample:
            self.oversample()
    
    def oversample(self):
        print('over sampling..')
        new_df = self._df.copy()
        new_label = self._label.copy()
        for c in tqdm(range(1103)):
            idx = np.where(new_label[:,c]==1)[0]
            if len(idx)>100: continue
            idx = np.where(self._label[:,c]==1)[0]
            df_data = self._df.iloc[idx]
            label_data = self._label[idx]
            if len(idx)==0:
                continue
            new_df = pd.concat([new_df]+[df_data]*int((100-len(idx))//len(idx)+1), ignore_index=True)
            new_label = np.vstack([new_label]+[label_data]*int((100-len(idx))//len(idx)+1))

        self._df = new_df
        self._label = new_label

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(item, self._root,
                                     transform=self._image_transform,
                                     debug=self._debug)
        return image, torch.FloatTensor(self._label[idx])


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame,
                 transform: Callable, tta: int):
        self._root = root
        self._df = df
        self._image_transform = transform
        self._tta = tta

    def __len__(self):
        return len(self._df) * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root,
                                     transform=self._image_transform)
        return image, item.id


def load_transform_image(
        item, root: Path,
        transform: Callable,
        debug: bool = False):
    image = load_image(item, root)
    image = transform(image = image)
    if debug:
        image.save('_debug.png')
    return image['image']


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})
