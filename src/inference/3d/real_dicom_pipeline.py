import os
import pydicom
import numpy as np
import trimesh
from skimage.measure import marching_cubes

def load_dicom_series(directory):
    """
    Loads a DICOM series from a directory and converts it to a 3D numpy array (HU).
    """
    print(f"📂 Scanning directory: {directory}")
    
    # 1. Find all DICOM files
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        raise FileNotFoundError("No .dcm files found in the specified directory.")
        
    print(f"   Found {len(dicom_files)} DICOM files.")
    
    # 2. Read files and sort by Instance Number or Z-position
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception as e:
            print(f"   ⚠️ Skipping {f}: {e}")
            
    # Sort by ImagePositionPatient Z coordinate (reliable for 3D)
    # If missing, fall back to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        print("   ⚠️ ImagePositionPatient not found, sorting by InstanceNumber.")
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
    # 3. Convert to 3D Numpy Array (Hounsfield Units)
    # Get pixel data
    image_shape = slices[0].pixel_array.shape
    volume = np.stack([s.pixel_array for s in slices])
    volume = volume.astype(np.float64)
    
    # Apply Rescale Slope/Intercept to get HU
    slope = getattr(slices[0], 'RescaleSlope', 1)
    intercept = getattr(slices[0], 'RescaleIntercept', 0)
    
    volume = volume * slope + intercept
    
    print(f"✅ Loaded 3D Volume. Shape: {volume.shape}, Range: {volume.min()} to {volume.max()} HU")
    return volume

def reconstruct_mesh(volume, output_path, threshold=200):
    """
    Reconstructs a 3D mesh from a volume using Marching Cubes.
    """
    print(f"⚙️ Running Marching Cubes (Threshold: {threshold})...")
    
    try:
        verts, faces, normals, values = marching_cubes(volume, level=threshold)
        print(f"   Vertices: {len(verts)}")
        print(f"   Faces: {len(faces)}")
        
        # Create Trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        print(f"💾 Exporting mesh to: {output_path}")
        mesh.export(output_path)
        print("✨ Done!")
        return True
    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")
        return False

if __name__ == "__main__":
    print("--- High-Precision 3D Bone Reconstructor (Real Data) ---")
    
    # Ask for input
    dicom_dir = input("👉 Enter the full path to your DICOM folder: ").strip()
    
    # Remove quotes if user added them (common when copying paths)
    dicom_dir = dicom_dir.replace('"', '').replace("'", "")
    
    if not os.path.exists(dicom_dir):
        print(f"❌ Directory not found: {dicom_dir}")
    else:
        try:
            # 1. Load Data
            volume = load_dicom_series(dicom_dir)
            
            # 2. Define Output
            output_dir = "prototypes/3d_reconstruction/data"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "reconstructed_bone.obj")
            
            # 3. Reconstruct
            # Bone threshold is typically > 200 HU
            reconstruct_mesh(volume, output_file, threshold=300)
            
        except Exception as e:
            print(f"❌ Error: {e}")
