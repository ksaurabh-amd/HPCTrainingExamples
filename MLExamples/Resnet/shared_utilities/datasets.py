"""
Shared dataset utilities for ResNet workshop versions.

This module provides consistent dataset loading and preprocessing
across all workshop versions to ensure fair performance comparisons.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional
import os


class DatasetConfig:
    """Configuration for dataset loading."""
    
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    @staticmethod
    def get_transforms(dataset_name: str, is_training: bool = True) -> transforms.Compose:
        """Get appropriate transforms for the dataset."""
        
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            if is_training:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        DatasetConfig.CIFAR10_MEAN if dataset_name.lower() == 'cifar10' else DatasetConfig.CIFAR100_MEAN,
                        DatasetConfig.CIFAR10_STD if dataset_name.lower() == 'cifar10' else DatasetConfig.CIFAR100_STD
                    ),
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        DatasetConfig.CIFAR10_MEAN if dataset_name.lower() == 'cifar10' else DatasetConfig.CIFAR100_MEAN,
                        DatasetConfig.CIFAR10_STD if dataset_name.lower() == 'cifar10' else DatasetConfig.CIFAR100_STD
                    ),
                ])
        
        elif dataset_name.lower() == 'imagenet':
            if is_training:
                return transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(DatasetConfig.IMAGENET_MEAN, DatasetConfig.IMAGENET_STD),
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(DatasetConfig.IMAGENET_MEAN, DatasetConfig.IMAGENET_STD),
                ])
        
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_info(dataset_name: str) -> dict:
    """Get dataset information."""
    
    info = {
        'cifar10': {
            'num_classes': 10,
            'input_size': (3, 32, 32),
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        },
        'cifar100': {
            'num_classes': 100,
            'input_size': (3, 32, 32),
            'class_names': None  # Too many to list
        },
        'imagenet': {
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'class_names': None  # Too many to list
        }
    }
    
    return info.get(dataset_name.lower(), {})


def create_data_loaders(
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    data_root: str = './data',
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    dataset_name = dataset_name.lower()
    
    # Get transforms
    train_transform = DatasetConfig.get_transforms(dataset_name, is_training=True)
    val_transform = DatasetConfig.get_transforms(dataset_name, is_training=False)
    
    # Create datasets
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=download, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=download, transform=val_transform
        )
    
    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=download, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=download, transform=val_transform
        )
    
    elif dataset_name == 'imagenet':
        # For ImageNet, assume data is already downloaded
        train_dataset = torchvision.datasets.ImageNet(
            root=data_root, split='train', transform=train_transform
        )
        val_dataset = torchvision.datasets.ImageNet(
            root=data_root, split='val', transform=val_transform
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


class SubsetDataLoader:
    """Create data loaders with limited samples for quick testing."""
    
    def __init__(self, dataset_name: str, max_samples: int = 5000, **kwargs):
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.kwargs = kwargs
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get subset data loaders."""
        
        # Create full loaders first
        train_loader, val_loader = create_data_loaders(self.dataset_name, **self.kwargs)
        
        # Create subset datasets
        train_indices = torch.randperm(len(train_loader.dataset))[:self.max_samples]
        val_indices = torch.randperm(len(val_loader.dataset))[:min(self.max_samples//5, 1000)]
        
        train_subset = torch.utils.data.Subset(train_loader.dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_loader.dataset, val_indices)
        
        # Create subset loaders
        train_subset_loader = DataLoader(
            train_subset,
            batch_size=self.kwargs.get('batch_size', 32),
            shuffle=True,
            num_workers=self.kwargs.get('num_workers', 4),
            pin_memory=self.kwargs.get('pin_memory', True),
            drop_last=True
        )
        
        val_subset_loader = DataLoader(
            val_subset,
            batch_size=self.kwargs.get('batch_size', 32),
            shuffle=False,
            num_workers=self.kwargs.get('num_workers', 4),
            pin_memory=self.kwargs.get('pin_memory', True),
            drop_last=False
        )
        
        return train_subset_loader, val_subset_loader


def calculate_dataset_stats(dataset_name: str, data_root: str = './data') -> dict:
    """Calculate dataset statistics for normalization."""
    
    # Create basic transform (no normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
    elif dataset_name.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Statistics calculation not supported for {dataset_name}")
    
    # Calculate mean and std
    loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': total_samples
    }


# Convenience function for quick dataset setup
def get_quick_loaders(dataset_name: str, batch_size: int = 32, max_samples: int = 5000):
    """Get data loaders for quick testing with limited samples."""
    
    subset_loader = SubsetDataLoader(
        dataset_name=dataset_name,
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=2,  # Reduced for quick setup
        pin_memory=True
    )
    
    return subset_loader.get_loaders()


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    # Test CIFAR-10
    try:
        train_loader, val_loader = create_data_loaders('cifar10', batch_size=32)
        print(f"CIFAR-10: Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        
        # Test a batch
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Batch shape: {data.shape}, Target shape: {target.shape}")
            break
            
    except Exception as e:
        print(f"CIFAR-10 loading failed: {e}")
    
    # Test dataset info
    info = get_dataset_info('cifar10')
    print(f"CIFAR-10 info: {info}")
    
    print("Dataset loading test completed!")