import random
import math
import numpy as np

from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomResizedCrop, ColorJitter, RandomApply)


def get_train_transform(size):
    return Compose([
            #SquarePadding(),
            #Resize(size + 32), # Resize keeping aspect ratio
            #RandomCrop(size),
            RandomResizedCrop(size, scale=(0.08, 1.0)), #scale range? default(0.08, 1)
            #RandomHorizontalFlip(),
            #RandomApply([ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            #             ],
            #              p=0.2),
            #Resize((size, size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    
def get_test_tta_transform(size):
    return Compose([
            #SquarePadding(),
            #RandomResizedCrop(size, scale=(0.9, 1.0)), #cv 0.5996
#            RandomHorizontalFlip(),                    #cv 0.6176
            RandomApply([ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                         ],
                          p=0.2),                       #cv 0.6178
            Resize((size, size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    
def get_test_transform(size):
    return Compose([
            #SquarePadding(),
            #HorizontalFlip(),
            Resize((size, size)),
            #Resize(np.random.randint(size+20, 2*size+40)), # Resize keeping aspect ratio
            #RandomCrop(size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

def get_test_trainform_NAS(size):
    return Compose([
            Resize((size, size)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),                    
            ])

tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
