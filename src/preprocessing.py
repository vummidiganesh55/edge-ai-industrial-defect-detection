import torch
from torchvision import transforms  # type: ignore[import-not-found]


IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def get_train_transform():
    """
    Preprocessing for normal training images.
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_test_transform():
    """
    Preprocessing for test images.
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def denormalize(image_tensor):
    """
    Convert normalized tensor back to displayable range.
    """

    mean = torch.tensor(
        IMAGENET_MEAN,
        device=image_tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        device=image_tensor.device
    ).view(3, 1, 1)

    image = image_tensor * std + mean

    return torch.clamp(image, 0.0, 1.0)