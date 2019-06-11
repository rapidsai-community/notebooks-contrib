from torchvision.transforms import Compose,FiveCrop,Resize,Lambda,ToTensor,Normalize
import torch
import cv2
from PIL import Image
from typing import Callable, List
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TTADataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable,
                 valid = True):
        super().__init__()        
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self.valid = valid

        self._label = np.zeros((len(self._df), 1103))
        attributes = self._df.attribute_ids.tolist()
        for i in range(len(self._df)):
            for c in attributes[i].split():
                self._label[i, int(c)] = 1

    def __len__(self):
        return len(self._df)# * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root,
                                     image_transform_crop=self._image_transform)
        if self.valid:
            return image, torch.FloatTensor(self._label[idx])
        else:
            return image, item.id

def load_transform_image(
        item, root: Path,
        image_transform_crop: Callable,
        debug: bool = False):
    image = load_image(item, root)
    image = image_transform_crop(image)
    if debug:
        image.save('_debug.png')
    return image

def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

fivecrop_transform = Compose([
    Resize(400),
    FiveCrop(384),  # this is a list of PIL Images
    Lambda(lambda crops: torch.stack([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(ToTensor()(crop)) for crop in crops]))  # returns a 4D tensor
])

test_transform = Compose([
    # PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE),
    # SmallestMaxSize(max_size=340),
    Resize(384, 384),
    ToTensor(),
    # RandomSizedCrop(min_max_height=(300, 340), height=IMG_SIZE, width=IMG_SIZE),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

