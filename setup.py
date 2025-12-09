#!/usr/bin/env python3
"""
Arabic Digit Recognition System - Setup Script
"""

import os
import sys
import subprocess


def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def create_directory_structure():
    """Create necessary directories"""
    directories = ['data', 'models', 'training_results']

    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")
        else:
            print(f"📁 Directory exists: {dir_name}")


def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")

    requirements = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'PyQt5>=5.15.0',
        'Pillow>=10.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.3.0',
        'tqdm>=4.65.0',
        'seaborn>=0.12.0',
        'opencv-python>=4.8.0'
    ]

    for package in requirements:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


def check_data_structure():
    """Check data folder structure"""
    print("\n📂 Checking data structure...")

    if os.path.exists('data'):
        # Check for class folders
        class_folders = [d for d in os.listdir('data')
                         if os.path.isdir(os.path.join('data', d)) and d.isdigit()]

        if class_folders:
            print(f"✅ Found class folders: {sorted(class_folders)}")

            # Count images
            total_images = 0
            for folder in class_folders:
                folder_path = os.path.join('data', folder)
                images = [f for f in os.listdir(folder_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
                print(f"   📁 {folder}: {len(images)} images")
                total_images += len(images)

            print(f"\n📊 Total images: {total_images}")

            if total_images < 100:
                print("⚠️  Warning: Less than 100 images found. Consider adding more data.")
        else:
            print("⚠️  No class folders found in 'data/'")
            print("   Expected: data/0/, data/1/, ..., data/9/")
    else:
        print("⚠️  'data' folder not found")


def print_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)

    print("\n📋 NEXT STEPS:")
    print("1. Prepare your data:")
    print("   Create folder structure: data/0/, data/1/, ..., data/9/")
    print("   Put images of Arabic digits in corresponding folders")

    print("\n2. Train the model:")
    print("   python train.py --mode train")
    print("   or run.bat and select option 2")

    print("\n3. Run the GUI:")
    print("   python main.py")
    print("   or run.bat and select option 1")

    print("\n4. Quick test with sample data:")
    print("   python train.py --mode sample  # Creates sample data")
    print("   python train.py --mode train   # Train on sample data")
    print("   python main.py                 # Test the GUI")

    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("Arabic Digit Recognition System - Setup")
    print("=" * 60)

    # Check Python
    check_python_version()

    # Create directories
    create_directory_structure()

    # Install dependencies
    install_dependencies()

    # Check data
    check_data_structure()

    # Print instructions
    print_instructions()


if __name__ == "__main__":
    main()