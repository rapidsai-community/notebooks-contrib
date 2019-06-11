from albumentations import (Compose, RandomCrop, Normalize, Resize, Flip, RandomRotate90, PadIfNeeded, ShiftScaleRotate,
                            HueSaturationValue, Transpose, RandomBrightnessContrast,SmallestMaxSize,HorizontalFlip, ToGray,
                            Blur, GaussNoise, RandomSizedCrop, IAAPerspective, ElasticTransform)
from albumentations.pytorch import ToTensor
import cv2

IMG_SIZE = 320
def get_train_transform(size):
    train_transform = Compose([
        Resize(int(size*1.05), int(size*1.05)),
        RandomSizedCrop(min_max_height=(int(size*0.95), int(size*1.05)), height=size, width=size),
        HorizontalFlip(),
        Blur(p=0.2, blur_limit=5),
        GaussNoise(p=0.2, var_limit=(5.,30.)),
        HueSaturationValue(p=0.2),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        ToTensor(normalize={'mean':[0.485, 0.456, 0.406],'std':[0.229, 0.224, 0.225]})
    ])
    return train_transform
    
def get_test_transform(size):
    test_transform = Compose([
        Resize(size, size),
        ToTensor(normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    ])
    return test_transform

def get_test_tta_transform(size):
    test_transform = Compose([
        Resize(int(size*1.05), int(size*1.05)),
        RandomSizedCrop(min_max_height=(int(size*0.95), int(size*1.05)), height=size, width=size),
        HorizontalFlip(),
        ToTensor(normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    ])
    return test_transform