"""Forked from https://github.com/facebookresearch/barlowtwins/blob/main/main.py"""

import random
import matplotlib
import torchvision
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class MNISTTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            GaussianBlur(p=1.0),
            Solarization(p=1.0),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return transforms.ToTensor()(x), y1, y2


if __name__ == "__main__":
    dataset = torchvision.datasets.MNIST(
        root="/fs/data/tejasj",
        download=True,
        transform=MNISTTransform(),  
    )
    data, _ = dataset[0]
    assert data[0].shape == (1, 28, 28)
    assert data[1].shape == (1, 28, 28)
    assert data[2].shape == (1, 28, 28)

    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(data[0].permute(1, 2, 0))
    ax[2].imshow(data[2].permute(1, 2, 0))
    ax[1].imshow(data[1].permute(1, 2, 0))
    print(F.mse_loss(data[0], data[1]))
    print(F.mse_loss(data[0], data[2]))
    plt.savefig("augmentation_sample.png")


    