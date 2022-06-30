from __future__ import annotations

from torchvision import transforms as T


def get_train_transform(size: int = 512):
    # noqa :  Composes several transforms together. This transform does not support torchscript
    train_transform = T.Compose(
        [
            T.Resize([size, size]),
            T.RandomInvert(),
            T.RandomAutocontrast(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.445], std=[0.269]),
        ]
    )
    return train_transform
