import pickle
from pathlib import Path
import torch
from PIL import Image, ImageOps

import numpy as np
from torch.utils.data import Dataset


class PersonalityDataset(Dataset):
    def __init__(self, pickle_dir: Path):
        """
        :param pickle_dir: directory with pickle files
        """
        super().__init__()
        self.file_list = pickle_dir.glob("*.pickle")
        self.faces_list = dict(X=[],Y=[])
        for file in self.file_list:
            print('file:', file)
            pic = pickle.load(open(file, "rb"))
            self.faces_list['X'].extend(np.array(pic['X'], dtype='f'))
            self.faces_list['Y'].extend(np.array(pic['Y']))
                

    def __getitem__(self, index):
        image = self.faces_list['X'][index]
        label = self.faces_list['Y'][index]
        normalized = PersonalityDataset._normalize(image.copy())
        return {
            "original": image, 
            "normalized": normalized,
            "label": label
        }

    def __len__(self):
        return len(self.faces_list['Y'])
      
    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        image -= image.min()
        if (image.max() - image.min()) == 0:
            return np.zeros_like(image)
        else:
            image /= image.max()
            return image
