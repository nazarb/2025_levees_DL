import random
from torchvision.transforms import functional
from torchvision.transforms import v2

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
        return image, target


class RandomVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.vflip(image)
            target = functional.vflip(target)
        return image, target

class RandomRotate:
    def __init__(self, rot_prob=0.5, degrees=(0, 180)):
        self.rot_prob = rot_prob
        self.rotater = v2.RandomRotation(degrees=degrees)

    def __call__(self, image, target):
        if random.random() < self.rot_prob:
            image = self.rotater(image)
            target = self.rotater(target)
        return image, target