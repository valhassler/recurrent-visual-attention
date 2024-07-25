import numpy as np
from rva.utils import plot_images

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

#Try to avoid instability problems in Training throwing away to small batches
class CompleteBatchDataLoader(DataLoader):
    def __iter__(self):
        for batch in super().__iter__():
            if len(batch[0]) == self.batch_size:
                yield batch

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

    train_loader = torch.utils.data.CompleteBatchDataLoader(
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

def plot_images_extra(images, labels):
    fig, axes = plt.subplots(3, 3, figsize=(4, 4))
    for img, label, ax in zip(images, labels, axes.flatten()):
        ax.imshow(img.squeeze(), cmap='gray' if img.shape[2] == 1 else None)
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def create_dataset_subset(dataset_class, data_dir, mean_std, random_seed, percentage, custom_transform=None):
    """
    Create a subset of the dataset.
    
    Args:
        dataset_class: the dataset class from torchvision.datasets (e.g., datasets.ImageNet).
        data_dir: path directory to the dataset.
        mean_std: list with mean and std for normalization.
        random_seed: fix seed for reproducibility.
        percentage: percentage of the dataset to use.
        custom_transform: custom transform to apply to the dataset.
        
    Returns:
        subset_dataset: a subset of the original dataset.
    """
    mean, std = mean_std[0], mean_std[1]
    normalize = transforms.Normalize(mean, std)

    if custom_transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            custom_transform,
            transforms.ToTensor(),
            normalize
        ])

    full_dataset = dataset_class(root=data_dir, split='train', download=True, transform=transform)
    
    # Get the subset of the dataset
    num_samples = int(percentage * len(full_dataset))
    indices = list(range(len(full_dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    subset_indices = indices[:num_samples]
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    
    return subset_dataset

def create_dataloaders(subset_dataset, mean_std, batch_size, random_seed, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=False, show_sample=False):
    """
    Create DataLoaders for the subset dataset.
    
    Args:
        subset_dataset: the subset of the original dataset.
        mean_std: list with mean and std for normalization.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for the validation set.
        shuffle: whether to shuffle the train/validation indices.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory.
        show_sample: plot 3x3 sample grid of the dataset.
        
    Returns:
        train_loader: DataLoader for the training set.
        valid_loader: DataLoader for the validation set.
    """
    num_train = len(subset_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if show_sample:
        sample_loader = DataLoader(
            subset_dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        images = images.numpy().transpose(0, 2, 3, 1)
        mean = np.array(mean_std[0])
        std = np.array(mean_std[1])
        plot_images(images, labels, mean, std)
    
    return train_loader, valid_loader
