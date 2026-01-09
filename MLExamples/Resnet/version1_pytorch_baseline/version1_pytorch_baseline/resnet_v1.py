#!/usr/bin/env python3
"""
ResNet V1: PyTorch Baseline with Comprehensive Profiling Integration

Enhanced version of the baseline ResNet implementation for the AI Workshop.
This version integrates PyTorch Profiler and comprehensive performance analysis 
capabilities while maintaining deterministic execution.

Features:
- PyTorch Profiler integration with GPU/CPU timeline analysis
- Memory profiling and bandwidth analysis
- Operator-level performance characterization
- Bottleneck identification and analysis
- Comprehensive performance reporting

Usage:
    # Basic training
    python resnet_v1.py --model resnet18 --dataset cifar10 --batch-size 32 --epochs 5

    # Complete profiling suite
    python resnet_v1.py --enable-pytorch-profiler --profile-memory \
            --profile-with-stack --profile-with-flops --profile-dir ./profiles \
            --model resnet18 --dataset cifar10 --batch-size 32 --epochs 5

    # Using AMD profiler rocprofv3
    rocprofv3 --kernel-trace --marker-trace --summary --summary-per-domain \
              --summary-output-file=profile.out -- python3 resnet_v1.py    \
              --model resnet18 --dataset cifar10 --batch-size 32 --epochs 5
    
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import time
import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
import warnings
from datetime import datetime
from analyze_trace import TraceAnalyzer
# Performance monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False




@dataclass
class ResNetConfig:
    """Configuration for ResNet model - optimized for profiling."""
    model_name: str = "resnet18"
    num_classes: int = 10
    input_channels: int = 3
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    epochs: int = 5
    # Model specific
    inplanes: int = 64
    groups: int = 1
    width_per_group: int = 64
    replace_stride_with_dilation: Optional[List[bool]] = None
    norm_layer: Optional[nn.Module] = None


@dataclass 
class ProfilerConfig:
    """Configuration for profiling options."""
    enable_pytorch_profiler: bool = False
    profile_memory: bool = False
    profile_with_stack: bool = False
    profile_with_flops: bool = False
    profile_dir: str = "./profiles"
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1


class PerformanceMonitor:
    """Comprehensive performance monitoring and analysis."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all performance counters."""
        self.batch_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.losses = []
        self.accuracies = []
        self.memory_usage = []
        self.gpu_utilization = []
        
    def record_batch_time(self, batch_time: float):
        """Record batch processing time."""
        self.batch_times.append(batch_time)
    
    def record_forward_time(self, forward_time: float):
        """Record forward pass time."""
        self.forward_times.append(forward_time)
    
    def record_backward_time(self, backward_time: float):
        """Record backward pass time."""
        self.backward_times.append(backward_time)
    
    def record_optimizer_time(self, opt_time: float):
        """Record optimizer step time."""
        self.optimizer_times.append(opt_time)
    
    def record_loss(self, loss: float):
        """Record training loss."""
        self.losses.append(loss)
    
    def record_accuracy(self, accuracy: float):
        """Record training accuracy."""
        self.accuracies.append(accuracy)
    
    def record_memory_usage(self):
        """Record current memory usage."""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024**2)
            self.memory_usage.append(memory_mb)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.batch_times:
            return {}
        
        summary = {}
        
        if NUMPY_AVAILABLE:
            summary.update({
                'avg_batch_time_ms': float(np.mean(self.batch_times) * 1000),
                'avg_forward_time_ms': float(np.mean(self.forward_times) * 1000) if self.forward_times else 0,
                'avg_backward_time_ms': float(np.mean(self.backward_times) * 1000) if self.backward_times else 0,
                'avg_optimizer_time_ms': float(np.mean(self.optimizer_times) * 1000) if self.optimizer_times else 0,
                'final_loss': float(np.mean(self.losses[-5:])) if len(self.losses) >= 5 else (self.losses[-1] if self.losses else 0),
                'final_accuracy': float(np.mean(self.accuracies[-5:])) if len(self.accuracies) >= 5 else (self.accuracies[-1] if self.accuracies else 0),
                'peak_memory_mb': float(max(self.memory_usage)) if self.memory_usage else 0,
                'avg_memory_mb': float(np.mean(self.memory_usage)) if self.memory_usage else 0,
            })
        else:
            # Fallback without numpy
            summary.update({
                'avg_batch_time_ms': sum(self.batch_times) / len(self.batch_times) * 1000,
                'final_loss': self.losses[-1] if self.losses else 0,
                'final_accuracy': self.accuracies[-1] if self.accuracies else 0,
                'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            })
        
        return summary


def setup_deterministic_environment():
    """Configure PyTorch for deterministic execution."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Enable deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        with record_function("basic_block"):
            identity = x

            with record_function("conv1_bn1_relu"):
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

            with record_function("conv2_bn2"):
                out = self.conv2(out)
                out = self.bn2(out)

            with record_function("shortcut"):
                if self.downsample is not None:
                    identity = self.downsample(x)

            with record_function("residual_add_relu"):
                out += identity
                out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50, ResNet-101, and ResNet-152."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups, dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        with record_function("bottleneck_block"):
            identity = x

            with record_function("conv1_bn1_relu"):
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

            with record_function("conv2_bn2_relu"):
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

            with record_function("conv3_bn3"):
                out = self.conv3(out)
                out = self.bn3(out)

            with record_function("shortcut"):
                if self.downsample is not None:
                    identity = self.downsample(x)

            with record_function("residual_add_relu"):
                out += identity
                out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model for profiling and demonstration."""

    def __init__(self, block, layers, config: ResNetConfig):
        super(ResNet, self).__init__()
        self.config = config
        
        if config.norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = config.norm_layer
        self._norm_layer = norm_layer

        self.inplanes = config.inplanes
        self.dilation = 1
        if config.replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            config.replace_stride_with_dilation = [False, False, False]
        if len(config.replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(config.replace_stride_with_dilation))
        self.groups = config.groups
        self.base_width = config.width_per_group

        # CIFAR-10 specific modifications
        if config.num_classes == 10:  # CIFAR-10
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:  # ImageNet
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        if config.num_classes == 10:  # CIFAR-10 - no maxpool
            self.maxpool = nn.Identity()
        else:  # ImageNet
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=config.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=config.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=config.replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, config.num_classes)

        # Initialize weights
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        with record_function("resnet_forward"):
            with record_function("stem"):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

            with record_function("layer1"):
                x = self.layer1(x)

            with record_function("layer2"):
                x = self.layer2(x)

            with record_function("layer3"):
                x = self.layer3(x)

            with record_function("layer4"):
                x = self.layer4(x)

            with record_function("classifier"):
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

        return x


def get_resnet_model(config: ResNetConfig) -> ResNet:
    """Create ResNet model based on configuration."""
    
    model_configs = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3]),
    }
    
    if config.model_name not in model_configs:
        raise ValueError(f"Unsupported model: {config.model_name}")
    
    block, layers = model_configs[config.model_name]
    return ResNet(block, layers, config)


class SimpleImageDataset:
    """Simple image dataset for training demonstration."""
    
    def __init__(self, dataset_name: str = "cifar10", batch_size: int = 32, num_workers: int = 4):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 10 if dataset_name == "cifar10" else 100
        
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get training and validation data loaders."""
        
        if self.dataset_name == "cifar10":
            # CIFAR-10 dataset
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
        
        elif self.dataset_name == "cifar100":
            # CIFAR-100 dataset
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)
        
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        train_loader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True)
        
        test_loader = DataLoader(
            testset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)
        
        return train_loader, test_loader


def setup_pytorch_profiler(profiler_config: ProfilerConfig) -> Optional[profile]:
    """Setup PyTorch profiler with comprehensive configuration."""
    if not profiler_config.enable_pytorch_profiler:
        return None
    
    # Create profile directory
    profile_dir = Path(profiler_config.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    trace_files = []
    
    def trace_handler(prof):
        """Custom trace handler to capture filenames and run analysis."""
        # Get the output filename
        output_path = prof.export_chrome_trace(str(profile_dir / f"trace.pt.json"))
        trace_files.append(output_path)

    profiler = profile(
        activities=activities,
        record_shapes=True,
        profile_memory=profiler_config.profile_memory,
        with_stack=profiler_config.profile_with_stack,
        with_flops=profiler_config.profile_with_flops,
        schedule=torch.profiler.schedule(
            wait=profiler_config.wait_steps,
            warmup=profiler_config.warmup_steps,
            active=profiler_config.active_steps,
            repeat=profiler_config.repeat
        ),
        on_trace_ready=trace_handler
    )
    
    return profiler


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return 100. * correct / total


def train_resnet_v1(
    config: ResNetConfig,
    profiler_config: ProfilerConfig,
    dataset_name: str = "cifar10",
    device: str = "cuda"
):
    """Train the ResNet model with comprehensive profiling."""
    
    print("=" * 80)
    print("AI WORKSHOP - RESNET VERSION 1: PYTORCH BASELINE")
    print("     Comprehensive Profiling and Performance Analysis")
    print("=" * 80)
    
    # Setup deterministic environment
    setup_deterministic_environment()
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device.type.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Update config based on dataset
    if dataset_name == "cifar100":
        config.num_classes = 100
    
    # Create model
    model = get_resnet_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Configuration:")
    print(f"   Architecture: {config.model_name}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Number of classes: {config.num_classes}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Create dataset
    dataset = SimpleImageDataset(dataset_name, config.batch_size)
    train_loader, test_loader = dataset.get_data_loaders()
    
    print(f"\nDataset Configuration:")
    print(f"   Training samples: {len(train_loader.dataset):,}")
    print(f"   Validation samples: {len(test_loader.dataset):,}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Batches per epoch: {len(train_loader):,}")
    
    # Setup optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, 
                         momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print(f"\nTraining Configuration:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Momentum: {config.momentum}")
    print(f"   Weight decay: {config.weight_decay}")
    
    # Setup profiler
    profiler = setup_pytorch_profiler(profiler_config)
    performance_monitor = PerformanceMonitor()
    
    print(f"\nProfiling Configuration:")
    print(f"   PyTorch Profiler: {'Enabled' if profiler else 'Disabled'}")
    print(f"   Memory Profiling: {'Enabled' if profiler_config.profile_memory else 'Disabled'}")
    print(f"   Profile Directory: {profiler_config.profile_dir}")
    
    print(f"\nStarting training...")
    print("=" * 80)
    
    # Training loop
    model.train()
    total_batches = 0
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            forward_start = time.time()
            with record_function("forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            
            # Backward pass
            backward_start = time.time()
            optimizer.zero_grad()
            with record_function("backward_pass"):
                loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            
            # Optimizer step
            opt_start = time.time()
            with record_function("optimizer_step"):
                optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            opt_time = time.time() - opt_start
            
            batch_time = time.time() - batch_start_time
            
            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, targets)
            
            # Record performance metrics
            performance_monitor.record_batch_time(batch_time)
            performance_monitor.record_forward_time(forward_time)
            performance_monitor.record_backward_time(backward_time)
            performance_monitor.record_optimizer_time(opt_time)
            performance_monitor.record_loss(loss.item())
            performance_monitor.record_accuracy(accuracy)
            performance_monitor.record_memory_usage()
            
            # Update running statistics
            running_loss += loss.item()
            running_accuracy += accuracy
            num_batches += 1
            total_batches += 1
            
            # Profiler step
            if profiler:
                profiler.step()
            
            # Progress reporting
            if batch_idx % 50 == 0:
                samples_per_sec = config.batch_size / batch_time if batch_time > 0 else 0
                memory_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                
                print(f"Epoch {epoch+1}/{config.epochs} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {accuracy:.2f}% | "
                      f"Speed: {samples_per_sec:.1f} samples/sec | "
                      f"Memory: {memory_mb:.1f} MB")
            
            # Limit batches for quick demonstration
            if batch_idx >= 100:  # Process only first 100 batches
                break
        
        # Update learning rate
        scheduler.step()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / num_batches
        avg_accuracy = running_accuracy / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Average Accuracy: {avg_accuracy:.2f}%") 
        print(f"   Epoch Time: {epoch_time:.2f} seconds")
        print("-" * 60)
    
    print("=" * 80)
    
    # Performance analysis
    summary_stats = performance_monitor.get_summary_stats()
    
    print(f"\nPerformance Summary:")
    print(f"   Total batches processed: {total_batches:,}")
    if summary_stats:
        print(f"   Average batch time: {summary_stats.get('avg_batch_time_ms', 0):.1f} ms")
        print(f"   Average forward time: {summary_stats.get('avg_forward_time_ms', 0):.1f} ms")
        print(f"   Average backward time: {summary_stats.get('avg_backward_time_ms', 0):.1f} ms")
        print(f"   Average optimizer time: {summary_stats.get('avg_optimizer_time_ms', 0):.1f} ms")
        print(f"   Final loss: {summary_stats.get('final_loss', 0):.4f}")
        print(f"   Final accuracy: {summary_stats.get('final_accuracy', 0):.2f}%")
        print(f"   Peak memory usage: {summary_stats.get('peak_memory_mb', 0):.1f} MB")
        
        # Calculate throughput
        if summary_stats.get('avg_batch_time_ms', 0) > 0:
            throughput = config.batch_size / (summary_stats['avg_batch_time_ms'] / 1000)
            print(f"   Throughput: {throughput:.1f} samples/sec")
    
    # Save performance data
    save_performance_data(config, profiler_config, summary_stats, dataset_name)
    
    # Profiler cleanup and analysis
    if profiler:
        profiler.stop()
        
        print(f"\n{'='*80}")
        print("Record Function Timing Analysis:")
        print(f"{'='*80}")

        events = profiler.key_averages()
        target_functions = ['forward_pass','backward_pass','optimizer_step']
        
        # Aggregate events by key to handle duplicates from forward/backward passes
        event_map = {}
        for evt in events:
     
            if evt.key in target_functions:
                if evt.key not in event_map:
                    event_map[evt.key] = {
                        'total_time': 0,
                        'count': 0
                    }
                # Use CPU time as total time (includes kernel launch overhead)
                event_map[evt.key]['total_time'] += evt.cpu_time_total
                event_map[evt.key]['count'] += evt.count
        
        # Print aggregated results
        for key in target_functions:
            if key in event_map and event_map[key]['total_time'] > 0:
                total_time_ms = event_map[key]['total_time'] / 1000
                count = event_map[key]['count']
                avg_time_ms = total_time_ms / count if count > 0 else 0.0
                
                print(f"{key:30s} | "
                      f"Total: {total_time_ms:8.2f}ms | "
                      f"Calls: {count:6d} | "
                      f"Avg: {avg_time_ms:6.3f}ms")
        
        print(f"\nProfiler traces saved to: {profiler_config.profile_dir}")
        print("Use 'tensorboard --logdir ./profiles' to view the traces")
        # Analyze trace files
            
        profile_dir = Path(profiler_config.profile_dir)
        analyzer = TraceAnalyzer(profile_dir / f"trace.pt.json")
        stats = analyzer.run()
        analyzer.print_summary(1000)
        
    print(f"\nTraining completed successfully!")
    return model, summary_stats


def save_performance_data(config: ResNetConfig, profiler_config: ProfilerConfig, 
                         summary_stats: Dict[str, Any], dataset_name: str):
    """Save performance data to JSON file."""
    
    from datetime import datetime
    
    # Create profile directory if it doesn't exist
    profile_dir = Path(profiler_config.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    performance_data = {
        'version': 'v1_baseline',
        'timestamp': timestamp_str,
        'model_config': {
            'model_name': config.model_name,
            'dataset': dataset_name,
            'num_classes': config.num_classes,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
        },
        'performance_summary': summary_stats,
        'system_info': {
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'timestamp_iso': datetime.now().isoformat()
        }
    }
    
    profile_path = profile_dir / f"performance_summary_v1_{timestamp_str}.json"
    with open(profile_path, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"\nPerformance data saved to: {profile_path}")


def main():
    """Main entry point for Version 1 training."""
    
    parser = argparse.ArgumentParser(description='ResNet V1: PyTorch Baseline Training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18', 
                      choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                      help='ResNet model variant')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                      choices=['cifar10', 'cifar100'],
                      help='Dataset to use')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # Profiling arguments
    parser.add_argument('--enable-pytorch-profiler', action='store_true',
                      help='Enable PyTorch profiler')
    parser.add_argument('--profile-memory', action='store_true',
                      help='Enable memory profiling')
    parser.add_argument('--profile-with-stack', action='store_true',
                      help='Enable stack trace profiling')
    parser.add_argument('--profile-with-flops', action='store_true',
                      help='Enable FLOPS profiling')
    parser.add_argument('--profile-dir', type=str, default='./profiles',
                      help='Directory for profiler output')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create configurations
    config = ResNetConfig(
        model_name=args.model,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    profiler_config = ProfilerConfig(
        enable_pytorch_profiler=args.enable_pytorch_profiler,
        profile_memory=args.profile_memory,
        profile_with_stack=args.profile_with_stack,
        profile_with_flops=args.profile_with_flops,
        profile_dir=args.profile_dir,
    )
    
    # Run training
    model, stats = train_resnet_v1(config, profiler_config, args.dataset, args.device)


if __name__ == "__main__":
    main()