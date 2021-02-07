import torch.utils.data as data
from utils.util import read_path
import numpy as np
import torch
from skimage import io
import cv2


class HornetDataset(data.Dataset):
    def __init__(self):
        super(HornetDataset, self).__init__()
        self.data = []
        positive_paths = read_path('Positive ID')
        negative_paths = read_path('Negative ID')
        self.target = [1] * len(positive_paths) + [0] * len(negative_paths)
        img_paths = positive_paths + negative_paths
        for img_path in img_paths:
            image = io.imread(img_path) / 255.
            image = cv2.resize(image, (256, 256),
                               interpolation=cv2.INTER_AREA)
            if image.shape == (256, 256, 3):
                self.data.append(image)
        self.data = np.asarray(self.data)

    def __len__(self):
        return list(self.data.shape)[0]

    def __getitem__(self, index):
        return torch.from_numpy(np.asarray(self.data[index, :, :, :], dtype=np.float16)).float(), \
            torch.from_numpy(np.asarray(
                self.target[index], dtype=np.int16)).float()


class HornetTestDataset(data.Dataset):
    def __init__(self):
        super(HornetTestDataset, self).__init__()
        self.data = []
        positive_paths = read_path('Unprocessed')
        negative_paths = read_path('Unverified')
        self.target = [-1] * (len(positive_paths) + len(negative_paths))
        self.img_paths = positive_paths + negative_paths
        for img_path in self.img_paths:
            image = io.imread(img_path) / 255.
            image = cv2.resize(image, (256, 256),
                               interpolation=cv2.INTER_AREA)
            if image.shape == (256, 256, 3):
                self.data.append(image)
        self.data = np.asarray(self.data)

    def __len__(self):
        return list(self.data.shape)[0]

    def __getitem__(self, index):
        return torch.from_numpy(np.asarray(self.data[index, :, :, :], dtype=np.float16)).float(), \
            torch.from_numpy(np.asarray(
                self.target[index], dtype=np.int16)).float()
