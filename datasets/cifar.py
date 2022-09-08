from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np


class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    def __init__(self, *args, **kwargs):

        super(CIFAR10Instance, self).__init__(*args, **kwargs)
        self.labels = self.targets

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
            # if self.train:
            #     img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # if self.train:
        #
        #     return [img1, img2], target, index

        return img1, target, index


class CIFAR100Instance(CIFAR10Instance):
    """CIFAR100Instance Dataset.

    This is a subclass of the `CIFAR10Instance` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

# A = CIFAR10Instance
