#!/usr/bin/env python3
"""
ResNet Tutorial Setup Checker - No dependencies required
"""

import os
import sys

def check_directory_structure():
    """Check if the tutorial directory structure is set up correctly."""
    
    print("ResNet Tutorial Directory Structure Check")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Expected structure
    expected_structure = [
        "README.md",
        "version1_pytorch_baseline/",
        "shared_utilities/",
        "test_setup.py"
    ]
    
    print("\nChecking directory structure:")
    all_good = True
    
    for item in expected_structure:
        if os.path.exists(item):
            print(f"✓ {item}")
        else:
            print(f"✗ {item} (missing)")
            all_good = False
    
    print("\nDirectory contents:")
    for item in sorted(os.listdir(".")):
        if os.path.isdir(item):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("✅ Directory structure looks good!")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch torchvision")
        print("2. Create the ResNet implementation files")
        print("3. Run training: cd version1_pytorch_baseline && python resnet_v1.py")
    else:
        print("⚠️  Some files/directories are missing")
        print("Please create the missing items")
    
    return all_good

def show_tutorial_info():
    """Show tutorial information."""
    print("\n" + "=" * 50)
    print("ResNet Tutorial Information")
    print("=" * 50)
    
    print("\nThis tutorial teaches ResNet optimization using:")
    print("• PyTorch baseline implementation")
    print("• ROCm profiling tools")
    print("• Progressive optimization techniques")
    print("• Performance measurement and analysis")
    
    print("\nTutorial Structure:")
    print("📁 version1_pytorch_baseline/    - Baseline ResNet with profiling")
    print("📁 shared_utilities/             - Common dataset and metrics code")
    print("📄 README.md                     - Main tutorial documentation")
    print("📄 test_setup.py                 - Environment verification")
    
    print("\nTo get started:")
    print("1. Ensure PyTorch is installed")
    print("2. Create the implementation files (coming next)")
    print("3. Follow the tutorial exercises")

if __name__ == "__main__":
    structure_ok = check_directory_structure()
    show_tutorial_info()
    
    if structure_ok:
        print("\n🎉 Ready to proceed with creating the ResNet implementation!")
    else:
        print("\n⚠️  Please fix the directory structure first")
    
    sys.exit(0 if structure_ok else 1)
