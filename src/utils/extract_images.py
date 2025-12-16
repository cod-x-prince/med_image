import tarfile
import os
import glob

print("🔓 Extracting NIH dataset images...")

# Find all tar.gz files
tar_files = glob.glob("images_*.tar.gz")

if not tar_files:
    print("❌ No images_*.tar.gz files found in current directory")
    print("Please ensure the downloaded tar.gz files are in the same directory as this script")
    exit(1)

# Create images directory
os.makedirs('data/images', exist_ok=True)

print(f"Found {len(tar_files)} archive files to extract\n")

for idx, tar_path in enumerate(sorted(tar_files), 1):
    print(f"[{idx}/{len(tar_files)}] Extracting {tar_path}...")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Extract all files
            tar.extractall(path='data/images')
        
        # Get file size for confirmation
        size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        print(f"  ✓ Extracted ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"  ❌ Error extracting {tar_path}: {e}")
        continue

# Count extracted files
try:
    image_files = glob.glob('data/images/*.png')
    print(f"\n{'='*50}")
    print(f"✅ Extraction Complete!")
    print(f"   Total PNG files extracted: {len(image_files)}")
    print(f"   Location: data/images/")
    print(f"{'='*50}")
except Exception as e:
    print(f"\n⚠️ Could not count extracted files: {e}")

print("\nYou can now run: python train_real_model.py")
