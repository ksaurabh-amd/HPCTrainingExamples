# Version 1: PyTorch Baseline - Profiling Foundation

README.md from `HPCTrainingExamples/MLExamples/Resnet/version1_pytorch_baseline` in the Training Examples repository

## Overview

This version establishes the baseline ResNet implementation with comprehensive profiling capabilities. It provides a clean, well-instrumented foundation for understanding ResNet architecture, identifying performance bottlenecks, and measuring optimization improvements in subsequent versions.

## Learning Objectives

By completing this version, you will:

- **Understand ResNet Architecture**: Core concepts of residual connections and their implementation
- **Establish Performance Baseline**: Measure initial throughput, memory usage, and accuracy
- **Master PyTorch Profiler**: Use profiling tools to identify computational bottlenecks
- **Analyze Memory Patterns**: Understand memory allocation and peak usage patterns
- **Identify Optimization Opportunities**: Discover areas for improvement in subsequent versions

## Architecture Overview

This implementation uses the standard ResNet architecture with:

- **Residual Blocks**: Skip connections to enable training of deep networks
- **Batch Normalization**: Normalization for improved training stability
- **ReLU Activation**: Standard rectified linear unit activation
- **Global Average Pooling**: Efficient spatial dimension reduction

### Model Configuration

```python
# Default ResNet-18 Configuration for CIFAR-10
model_name = "resnet18"         # ResNet variant
num_classes = 10               # CIFAR-10 classes
input_channels = 3             # RGB images
batch_size = 32               # Training batch size
learning_rate = 0.1           # Initial learning rate
epochs = 5                    # Training epochs
```

### Supported Models

| Model | Layers | Parameters | CIFAR-10 Memory | ImageNet Memory |
|-------|--------|------------|-----------------|-----------------|
| ResNet-18 | 18 | 11.7M | ~890 MB | ~2.1 GB |
| ResNet-34 | 34 | 21.8M | ~1.2 GB | ~3.8 GB |


## Implementation Details

### Key Components

#### 1. ResNet Blocks
```python
class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18/34."""
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Skip connection
        out = self.relu(out)
        return out
```

#### 2. Performance Monitoring
```python
class PerformanceMonitor:
    """Comprehensive performance tracking."""
    def record_batch_time(self, batch_time)
    def record_forward_time(self, forward_time)
    def record_backward_time(self, backward_time)
    def get_summary_stats(self)
```

#### 3. Profiling Integration
- **PyTorch Profiler**: Operator-level performance analysis
- **Memory Profiling**: Peak and average memory usage tracking
- **Timeline Analysis**: CPU/GPU execution timeline
- **TensorBoard Integration**: Visual profiling results

### Deterministic Execution

The implementation ensures reproducible results through:
- Fixed random seeds (torch.manual_seed(42))
- Deterministic CUDA operations
- Consistent data loading order
- Suppressed non-deterministic optimizations

## Workshop Exercises

### Exercise 1: Baseline Performance Measurement

**Objective**: Establish baseline performance metrics

**Steps**:
1. Run basic training without profiling
2. Record throughput and memory usage
3. Measure training accuracy

```bash
# Basic training run
python resnet_v1.py --model resnet18 --dataset cifar10 --batch-size 32 --epochs 5

# Expected output:
# Throughput: ~1,023 samples/sec
# Peak memory: ~282 MB
# Batch time: ~31.3 ms
```

**Analysis Questions**:
- What is the training throughput (samples/sec)?
- How much GPU memory is being used?
- What is the final validation accuracy?

### Exercise 2: PyTorch Profiler Analysis

**Objective**: Use PyTorch Profiler to identify bottlenecks

**Steps**:
1. Enable PyTorch profiler
2. Generate profiling traces
3. Analyze results in TensorBoard

```bash
# Run with profiling enabled
python resnet_v1.py --model resnet18 --dataset cifar10 --batch-size 32 --epochs 2 \
    --enable-pytorch-profiler --profile-memory --profile-dir ./profiles

# View results in TensorBoard
tensorboard --logdir ./profiles
```

**Analysis Tasks**:
- Identify the top 10 most time-consuming operations
- Find memory allocation hotspots
- Analyze GPU kernel launch patterns
- Measure CPU vs GPU time distribution

**Expected Findings**:
- Convolution operations: 65-75% of compute time
- Batch normalization: 10-15% of compute time  
- Memory transfers: 5-10% of total time
- Kernel launch overhead: Significant opportunity

### Exercise 3: Model Scaling Analysis

**Objective**: Understand performance scaling with model size

**Steps**:
1. Test different ResNet variants
2. Compare performance characteristics
3. Analyze memory and compute scaling

```bash
# Test ResNet-18
python resnet_v1.py --model resnet18 --batch-size 32 --epochs 2

# Test ResNet-34
python resnet_v1.py --model resnet34 --batch-size 32 --epochs 2
```

**Comparison Matrix**:

| Model | Throughput (samples/sec) | Memory (MB) | Batch Time | Parameters |
|-------|-------------------------|-------------|------------|------------|
| ResNet-18 | ~1,023 | 282 | 31.3 ms | 11.7M |
| ResNet-34 | ~804 | 400.7 | 42.7 ms | 21.8M |

### Exercise 4: Batch Size Optimization

**Objective**: Find optimal batch size for throughput

**Steps**:
1. Test different batch sizes
2. Monitor memory usage and throughput
3. Identify optimal configuration

```bash
# Test different batch sizes
for bs in 16 32 64 128 256; do
    echo "Testing batch size: $bs"
    python resnet_v1.py --model resnet18 --batch-size $bs --epochs 1
done
```

**Expected Results**:
- Small batches (16-32): Underutilize GPU, lower throughput
- Medium batches (64-128): Optimal throughput/memory balance
- Large batches (256+): May exceed memory or show diminishing returns

## Profiling Tools Integration

### PyTorch Profiler Features

1. **Operator Profiling**
   - Time breakdown by operation type
   - CPU vs GPU execution analysis
   - Memory allocation tracking

2. **Timeline Analysis**
   - Kernel execution timeline
   - CPU-GPU synchronization points
   - Memory transfer visualization

3. **Memory Profiling**
   - Peak memory usage
   - Memory allocation/deallocation patterns
   - Memory fragmentation analysis

### TensorBoard Visualization

Access profiling results through TensorBoard:
```bash
tensorboard --logdir ./profiles --port 6006
```

**Key Views**:
- **Overview**: High-level performance summary
- **Operator**: Detailed operator-level analysis
- **Trace**: Timeline view of execution
- **Memory**: Memory usage patterns

## Key Performance Metrics

### Expected Baseline Results (ResNet-18, CIFAR-10, Batch=32)

| Metric | Value | Notes |
|--------|-------|-------|
| **Throughput** | 1,023 samples/sec | Training speed |
| **Batch Time** | 31.3 ms | Per-batch processing |
| **Forward Time** | 9.5 ms | Forward pass only |
| **Backward Time** | 20.8 ms | Backward pass only |
| **Optimizer Time** | 0.7 ms | Optimizer step |
| **Memory Usage** | 282 MB | Peak GPU memory |
| **Final Accuracy** | 24.4% | After 5 epochs (101 batches/epoch) |


In order to get the GPU kernel names and their corresponding time, we can use rocprofv3:

```bash
rocprofv3 --kernel-trace --marker-trace --summary --summary-per-domain \
              --summary-output-file=profile.out -- python3 resnet_v1.py    \
              --model resnet18 --dataset cifar10 --batch-size 32 --epochs 5
```

### Performance Characteristics

1. **Compute Bound Operations**
   - Convolution layers: High arithmetic intensity
   - Matrix multiplications in FC layer
   - Good GPU utilization (>80%)

2. **Memory Bound Operations**
   - Batch normalization: Low arithmetic intensity
   - Activation functions: Memory bandwidth limited
   - Optimization opportunity for fusion

3. **Communication Patterns**
   - Frequent CPU-GPU synchronization
   - Small tensor transfers
   - Kernel launch overhead



## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce batch size: `--batch-size 16`
- Use smaller model: `--model resnet18`
- Enable gradient checkpointing (V2+)

#### Slow Performance
```
Warning: Training speed below expected
```
**Diagnostics**:
- Check GPU utilization: `rocm-smi`
- Verify data loading: `--num-workers 0`
- Profile memory bandwidth: Enable memory profiling

#### Profiler Issues
```
Warning: Profiler traces not generated
```
**Solutions**:
- Check disk space in profile directory
- Verify PyTorch version supports profiling
- Use basic profiling first: `--enable-pytorch-profiler`

### Performance Debugging

#### Low GPU Utilization
1. **Check Data Loading**: Ensure data loading isn't a bottleneck
2. **Verify Batch Size**: Use larger batches if memory allows
3. **Profile Timeline**: Look for CPU-GPU synchronization issues

#### Memory Issues
1. **Monitor Peak Usage**: Track memory growth patterns
2. **Check Fragmentation**: Look for frequent alloc/dealloc
3. **Analyze Tensor Sizes**: Identify largest allocations

## Next Steps

After completing Version 1:

1. **Review Performance Baseline**: Document current performance metrics
2. **Identify Bottlenecks**: Focus on highest-impact optimization opportunities
3. **Prepare for Version 2**: Understand operator fusion concepts

Version 2 will introduce operator fusion to reduce kernel launch overhead and improve memory efficiency, targeting 1.5x speedup over this baseline.

## Performance Summary Template

Use this template to record your baseline results:

```
=== ResNet V1 Baseline Results ===
Model: ResNet-18
Dataset: CIFAR-10
Batch Size: 32

Performance Metrics:
- Throughput: _____ samples/sec
- Batch Time: _____ ms
- Forward Time: _____ ms  
- Backward Time: _____ ms
- Peak Memory: _____ MB
- Final Accuracy: _____%

Top Operations (from profiler):
1. aten::conv2d: _____%
2. aten::batch_norm: _____%
3. aten::relu: _____%

Optimization Opportunities:
- [ ] Operator fusion potential
- [ ] Memory optimization needed
- [ ] Custom kernel opportunities

Notes:
_________________________________
_________________________________
```

Ready to establish your ResNet baseline? Start with Exercise 1!