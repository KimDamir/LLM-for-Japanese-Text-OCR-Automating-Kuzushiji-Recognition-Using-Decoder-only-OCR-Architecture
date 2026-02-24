import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import tqdm
import re


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
    for filename, transcription in tqdm.tqdm(data_dict.copy().items(), desc='Augmenting the dataset'):
        image = Image.open(pathname+filename+'.jpg') if pathname != None else Image.open(filename)
        for i in range(num_images):
            idx = len(filename) if pathname != None else filename.index('.png')
            idx2 = 0 if pathname != None else filename.index('/')
            pattern = re.compile("\/kkanji|\/K49")
            if pathname != None:
                idx3 = 0
            else:
                match = pattern.search(filename)
                idx3 = match.start() + len(match.group())
            aug_name =  'augmented/' + filename[:idx] + '-' + str(i) + filename[idx:] if pathname != None \
                else filename[:idx2] + '/' + filename[idx2:idx3] + '/augmented/' + filename[idx3:idx] + '-' + str(i) + filename[idx:]
            augmented_image = augment_image(image, N, M)
            save_image(augmented_image, pathname+aug_name+'.jpg') if pathname != None else save_image(augmented_image, aug_name)
            data_dict[aug_name] = transcription
            
def augment_kuzushiji_kaggle(data_dict, num_images, N, M):
    # dir — the name of directory where data is stored
    # num_images — the number of augmented versions of each image that needs to be created.
    # N, M — parameters for augment_image's randAugment, number of layers and magnitude
    for filename, transcription in tqdm.tqdm(data_dict.copy().items(), desc='Augmenting the dataset'):
        image = Image.open(filename)
        for i in range(num_images):
            idx = filename.index('.jpg')
            idx2 = filename.index('images/')
            aug_name =  filename[:idx2+7] + 'augmented/' + filename[idx2+7:idx] + '-' + str(i) + filename[idx:]
            augmented_image = augment_image(image, N, M)
            save_image(augmented_image, aug_name)
            data_dict[aug_name] = transcription
            
                
    