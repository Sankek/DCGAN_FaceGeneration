import os
import shutil

import numpy as np
import gdown
import cv2 as cv
import pathlib

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tt

download_url = 'https://drive.google.com/uc?id=18X8g6DH5rZ1P9Ty0Yxyge7Pb7aWB21Ji'
image_size = 128


class ImagesRAM(Dataset):
    def __init__(self, data, dataset_mean=[0.5]*3, dataset_std=[0.5]*3):
        """
        Accept images array with shape B x W x H x C
        """
        super().__init__()
        self.mean = dataset_mean
        self.std = dataset_std

        self.normalize = tt.Compose([
                                     tt.Lambda(lambda images: torch.stack([tt.ToTensor()(image) for image in images])),
                                     tt.Normalize(self.mean, self.std) 
                                     ])

        self.data = self.normalize(data)

        
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        return self.data[index]
    
    
class FacesDataset(ImagesRAM):
    def __init__(self, path, download=True, dataset_mean=[0.5]*3, dataset_std=[0.5]*3):
        self.path = pathlib.Path(path)
        self.image_size = image_size
        if download:
            self._download_dataset()
        
        data = self._load_images()
        
        super().__init__(data, dataset_mean=dataset_mean, dataset_std=dataset_std)
         
            
    def _download_dataset(self):
        archive = f'{self.path.name}.zip'

        gdown.download(download_url, archive, quiet=False)
        shutil.unpack_archive(archive, os.path.curdir)
        os.remove(archive) 
        
        
    def _load_images(self):
        # загружаем весь датасет в оперативную память
        print('Loading images...')
        files = os.listdir(self.path)
        dataset_images = []
        for file in files:
            im = cv.imread(os.path.join(self.path, file))
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            dataset_images.append(im)
        dataset_images = np.array(dataset_images)
        print('Images are loaded!')
        
        return dataset_images
