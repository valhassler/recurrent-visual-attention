import numpy as np
from rva.utils import plot_images

import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_train_valid_loader_old(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):
    """Train and validation data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 9x9 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test datalaoder.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


def get_train_valid_loader_MNIST(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
    mod_MNIST=False
):
    """Train and validation data loaders for (modified) MNIST.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 3x3 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
        mod_MNIST: whether to apply affine transformations to the MNIST dataset.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if mod_MNIST:
        trans = transforms.Compose([
            transforms.RandomAffine(
                degrees=20,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=20
            ),
            transforms.Resize(60),
            transforms.ToTensor(),
            normalize
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        images = images.numpy().transpose(0, 2, 3, 1)
        plot_images(images, labels)

    return train_loader, valid_loader

# Example usage:

# Define a function to plot images with denormalization
def plot_images_CIFAR(images, labels, mean, std):
    images = images * std + mean  # Denormalize the images
    images = np.clip(images, 0, 1)  # Clip values to be between 0 and 1
    fig, axes = plt.subplots(3, 3, figsize=(4, 4))  # Create a 3x3 grid for 9 images
    for img, label, ax in zip(images, labels, axes.flatten()):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_train_valid_loader_CIFAR10(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # Define transforms for normalization
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), np.array([0.2023, 0.1994, 0.2010]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load the CIFAR-10 dataset
    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Visualize some images if show_sample is True
    if show_sample:
        sample_loader = DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        images = images.numpy().transpose(0, 2, 3, 1)  # Change shape to (batch_size, height, width, channels)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        plot_images_CIFAR(images, labels, mean, std)

    return train_loader, valid_loader

