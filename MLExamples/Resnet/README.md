# AI Workshop: ROCm Tools for PyTorch ResNet Training and Optimization

## Workshop Overview

This workshop provides a comprehensive hands-on introduction to PyTorch ResNet optimization and ROCm profiling tools through progressive implementation versions. Learn to identify bottlenecks, apply optimizations, and achieve significant performance improvements on AMD GPUs.

## Learning Objectives

By the end of this workshop, you will be able to:

- **Understand ResNet Architecture**: Core concepts of residual networks and their computational patterns
- **Profile PyTorch Models**: Use ROCm profiling tools to identify performance bottlenecks
- **Apply Optimization Techniques**: Implement fusion, mixed precision, and custom kernels
- **Measure Performance Gains**: Quantify improvements in training speed and memory usage
- **Debug Performance Issues**: Troubleshoot common optimization problems

## Quick Start

### 1. Verify Environment
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Run Version 1 (Baseline) - 5 minutes
```bash
cd version1_pytorch_baseline/
python resnet_v1.py --model resnet18 --dataset cifar10 --batch-size 32 --epochs 5

# Expected output:
# Throughput: ~1,023 samples/sec
# Peak memory: ~282 MB
# Batch time: ~31.3 ms
```

### 3. Run with Profiling
```bash
python resnet_v1.py --enable-pytorch-profiler --profile-memory
tensorboard --logdir ./profiles
```

## Expected Performance Progression

| Version | Speedup | Memory Reduction | Key Optimization |
|---------|---------|------------------|------------------|
| V1 Baseline | 1.0x | 0% | Profiling foundation |
| V2 Fused | 1.5x | 15% | Operator fusion |
| V3 Mixed Precision | 2.4x | 30% | AMP training |
| V4 Ultra Optimized | 3.2x | 39% | Custom kernels |

## Directory Structure

```
Resnet/
├── README.md                           # This overview
├── version1_pytorch_baseline/          # V1: PyTorch baseline
│   ├── README.md                       # V1 documentation
│   └── resnet_v1.py                    # Main implementation
├── shared_utilities/                   # Common utilities
│   ├── datasets.py                     # Dataset loading
│   └── metrics.py                      # Performance monitoring
└── test_resnet_tutorial.py             # Test suite
```

Ready to optimize ResNet training on AMD GPUs? Let's begin with Version 1!
