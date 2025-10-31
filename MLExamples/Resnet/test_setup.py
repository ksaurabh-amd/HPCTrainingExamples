#!/usr/bin/env python3
"""
Simple test script to verify ResNet tutorial setup.
"""

import torch
import torchvision
import sys

def test_environment():
    """Test PyTorch and environment setup."""
    print("ResNet Tutorial Setup Test")
    print("=" * 40)
    
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test model creation
    try:
        model = torchvision.models.resnet18(pretrained=False)
        print(f"✓ ResNet-18 model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False
    
    # Test dataset loading
    try:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Try to create dataset (won't download, just test API)
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform
        )
        print(f"✓ Dataset API accessible")
        
    except Exception as e:
        print(f"Note: Dataset not downloaded yet (this is normal): {e}")
    
    print("\n" + "=" * 40)
    print("Setup test completed!")
    print("Next steps:")
    print("1. Create the full implementation files")
    print("2. Run: python version1_pytorch_baseline/resnet_v1.py")
    
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
