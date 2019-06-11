from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# from al_transforms import tensor_transform, IMG_SIZE
from utils import ON_KAGGLE
import numpy as np
from tqdm import tqdm

N_CLASSES = 1103
DATA_ROOT = Path('../input/imet-2019-fgvc6' if ON_KAGGLE else '../input')

RESIZE = True
CROP = False


class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 # transform_resize: Callable,
                 transform_crop: Callable,
                 over_sample=False,
                 debug: bool = True):
        super().__init__()
        self._root = root
        self._df = df
        # self._transform_resize = transform_resize
        self._transform_crop = transform_crop
        self._debug = debug

        self._label = np.zeros((len(self._df), 1103))
        attributes = self._df.attribute_ids.tolist()
        for i in range(len(self._df)):
            for c in attributes[i].split():
                self._label[i, int(c)] = 1

        if over_sample:
            self.oversample()

    def get_valid_mask(self):
        mask = []
        for i in range(N_CLASSES):
            if self._label[:, i].sum() > 0:
                mask.append(i)
        return mask

    def get_ignore_class(self):
        ignore_class = []
        for i in range(N_CLASSES):
            if self._label[:, i].sum() == 0:
                ignore_class.append(i)
        print(ignore_class)

    def oversample(self):
        print('over sampling..')
        new_df = self._df.copy()
        new_label = self._label.copy()
        for c in tqdm(range(1103)):
            idx = np.where(new_label[:, c] == 1)[0]
            if len(idx) > 100: continue
            idx = np.where(self._label[:, c] == 1)[0]
            df_data = self._df.iloc[idx]
            label_data = self._label[idx]
            if len(idx) == 0:
                continue
            new_df = pd.concat([new_df] + [df_data] * int((100 - len(idx)) // len(idx) + 1), ignore_index=True)
            new_label = np.vstack([new_label] + [label_data] * int((100 - len(idx)) // len(idx) + 1))

        self._df = new_df
        self._label = new_label

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(item, self._root,
                                     # image_transform_resize=self._transform_resize,
                                     image_transform_crop=self._transform_crop,
                                     debug=self._debug)
        return image, torch.FloatTensor(self._label[idx])


def load_transform_image(
        item, root: Path,
        # image_transform_resize: Callable,
        image_transform_crop: Callable,
        debug: bool = False):
    # if debug:
    # image, resize = Image.fromarray(np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)), True
    image = load_image(item, root)
    # if resize==RESIZE:
    #     image = image_transform_resize(image = image)
    # else:
    image = image_transform_crop(image=image)
    if debug:
        image.save('_debug.png')
    return image['image']


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # old_size = image.shape[0:2]  # old_size is in (height, width) format
    # ratio = old_size[1]/old_size[0]
    # if ratio<0.5 and ratio>2:
    #     new_size = max(old_size)
    #     delta_top = (new_size-old_size[0])//2
    #     delta_bottom = new_size - delta_top-old_size[0]
    #     delta_left = (new_size-old_size[1])//2
    #     delta_right = new_size-delta_left-old_size[1]
    #     image = cv2.copyMakeBorder(image, delta_top, delta_bottom, delta_left, delta_right, cv2.BORDER_REFLECT)
    #     return image#, RESIZE
    # else:
    return image  # , CROP


def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})
