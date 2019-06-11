from albumentations import (Compose, RandomCrop, Normalize, Resize, Flip, RandomRotate90, PadIfNeeded, ShiftScaleRotate,
                            HueSaturationValue, Transpose, RandomBrightnessContrast,SmallestMaxSize,HorizontalFlip, ToGray,
                            Blur, GaussNoise, RandomSizedCrop, IAAPerspective, ElasticTransform,JpegCompression,OneOf,
                            MedianBlur,RandomGamma,CLAHE,InvertImg,IAASharpen,
                            CenterCrop)
from albumentations.pytorch import ToTensor
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class Solarize(ImageOnlyTransform):
    def __init__(self, threshold=128, always_apply=False, p=1.0):
        super(Solarize, self).__init__(always_apply, p)
        self.threshold = threshold

    def apply(self, img, **params):
        lut = []
        for i in range(256):
            if i < self.threshold:
                lut.append(i)
            else:
                lut.append(255 - i)
        return cv2.LUT(img, np.array(lut, dtype='uint8'))

class Posterize(ImageOnlyTransform):
    def __init__(self, bits=4, always_apply=False, p=1.0):
        super(Posterize, self).__init__(always_apply, p)
        self.bits = bits

    def apply(self, img, **params):
        lut = []
        mask = ~(2 ** (8 - self.bits) - 1)
        for i in range(256):
            lut.append(i & mask)
        return cv2.LUT(img, np.array(lut, dtype='uint8'))

IMG_SIZE = 384

# train_transform_resize = Compose([
#     # Resize(340, 340),
#     # RandomCrop(IMG_SIZE, IMG_SIZE),
#     SmallestMaxSize(max_size=340),
#     RandomSizedCrop(min_max_height=(250,340), height=IMG_SIZE, width=IMG_SIZE),
#     HorizontalFlip(),
#     # Flip(),
#     # RandomRotate90(p=0.1),
#     Transpose(),
#     ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
#     Blur(p=0.2),
#     GaussNoise(p=0.2),
#     HueSaturationValue(p=0.2),
#     RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
#     Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
#     ToTensor()
# ])

train_transform_crop = Compose([
    # PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE),
    # SmallestMaxSize(max_size=420),

    Resize(400, 400),
    RandomRotate90(),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.02, rotate_limit=15, p=0.3),
    # RandomSizedCrop(min_max_height=(300, 384), height=IMG_SIZE, width=IMG_SIZE),
    RandomCrop(height=IMG_SIZE, width=IMG_SIZE),
    # IAAPerspective(p=0.2),
    # ElasticTransform(p=0.2),
    HorizontalFlip(),
    # Flip(),
    # Transpose(),

    # Solarize(p=0.2),
    # Posterize(p=0.2),
    # InvertImg(p=0.2),
    # IAASharpen(p=0.2),
    Blur(p=0.2),
    # MedianBlur(p=0.3),
    GaussNoise(p=0.2),
    HueSaturationValue(p=0.2),
    CLAHE(p=0.2),

    # RandomGamma(p=0.2),
    # RandomGamma(),
    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    # JpegCompression(quality_lower=80, quality_upper=100,p=0.3),
    ToTensor(normalize={'mean':[0.485, 0.456, 0.406],'std':[0.229, 0.224, 0.225]})
    # ToTensor(normalize={'mean':[0.5,0.5,0.5],'std':[0.5,0.5,0.5]})
])

# test_transform_resize = Compose([
#     Resize(IMG_SIZE, IMG_SIZE),
#     Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
#     ToTensor()
# ])

test_transform_crop = Compose([
    # PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE),
    # SmallestMaxSize(max_size=420),
    # CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
    Resize(IMG_SIZE, IMG_SIZE),
    # RandomCrop(IMG_SIZE, IMG_SIZE),
    # RandomSizedCrop(min_max_height=(300, 340), height=IMG_SIZE, width=IMG_SIZE),
    ToTensor(normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
])

