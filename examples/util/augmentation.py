import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path


def augment_image(image, N, M):
    augmenter = v2.Compose([
        v2.ToImage(),
        v2.RandomInvert(0.2),
        # v2.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 1.)),
        v2.RandAugment(N, M),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.1926, 0.1926, 0.1926], std=[0.3343, 0.3343, 0.3343]),
    ])
    return(augmenter(image))

def augment_dataset(data_dict, num_images, N, M, pathname=None):
    # dir — the name of directory where data is stored
    # num_images — the number of augmented versions of each image that needs to be created.
    # N, M — parameters for augment_image's randAugment, number of layers and magnitude
    for filename, transcription in data_dict.copy().items():
        image = Image.open(pathname+filename+'.jpg') if pathname != None else Image.open(filename)
        for i in range(num_images):
            idx = len(filename) if pathname != None else filename.index('.png')
            aug_name =  filename[:idx] + '-' + str(i) + filename[idx:]
            augmented_image = augment_image(image, N, M)
            save_image(augmented_image, pathname+aug_name+'.jpg') if pathname != None else save_image(augmented_image, aug_name)
            data_dict[aug_name] = transcription
            
                
    