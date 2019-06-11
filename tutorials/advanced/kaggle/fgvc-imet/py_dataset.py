from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from py_transforms import tensor_transform
from utils import ON_KAGGLE


N_CLASSES = 1103
DATA_ROOT = Path('../input/imet-2019-fgvc6' if ON_KAGGLE else '../input/imet-2019-fgvc6')


class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 transform: Callable, debug: bool = True, over_sample=False,):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        target = torch.zeros(N_CLASSES)
        for cls in item.attribute_ids.split():
            target[int(cls)] = 1
        return image, target


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame,
                 transform: Callable, tta: int,
                 valid=False):
        self._root = root
        self._df = df
        self._image_transform = transform
        self._tta = tta
        self.valid = valid

    def __len__(self):
        return len(self._df) * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root, self._image_transform)
        
        if self.valid:
            target = torch.zeros(N_CLASSES)
            for cls in item.attribute_ids.split():
                target[int(cls)] = 1
            return image, target
        else:
            return image, item.id


def load_transform_image(
        item, root: Path, transform: Callable, debug: bool = False):
    image = load_image(item, root)
    image = transform(image)
    if debug:
        image.save('_debug.png')
    return image


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ##return image
    return Image.fromarray(image)

def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})