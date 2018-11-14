from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from imgaug import augmenters as iaa


class DataSetBase(Dataset):
    def __init__(self, dir_with_images, dst_size, csv_file, mode='train'):
        self.dir_with_images = dir_with_images
        self.dst_size = dst_size
        self.mode = mode
        self.df = pd.read_csv(csv_file)
        print('init dataset')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        current_id = self.df['Id'][index]

        img = DataSetBase.open_rgb(self.dir_with_images, current_id)
        if img.shape[0] != self.dst_size:
            img = cv2.resize(img, (self.dst_size, self.dst_size), cv2.INTER_CUBIC)

        img = DataSetBase.augment(img) / 255
        # means = np.array([0.0808, 0.0530, 0.0550, 0.0830], dtype=np.float32)
        # stds = np.array([0.394, 0.321, 0.327, 0.399], dtype=np.float32)
        # img = (img - means)/stds
        if self.mode == 'train':
            y = np.array(self.df.values[index][3:], dtype=np.float32)
        else:
            y = current_id
        # print(index, img.dtype, y.dtype)

        img = img.transpose((2, 0, 1))
        # print(img.shape, img.max(), img.min())

        return img, y

    def get_image_size(self):
        return self.dst_size

    @staticmethod
    def open_rgb(path, id_):
        img = cv2.imread(os.path.join(path, id_ + '.png')).astype(np.float32)
        return img

    @staticmethod
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.Affine(rotate=(0, 360), translate_percent=(0, 0.1), scale=(0.8, 1.2), mode='reflect'),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sharpen(alpha=(0.0, 0.05), lightness=(0.8, 1.2)),
            iaa.ContrastNormalization(alpha=(0.9, 1.1))
            ],
            random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug