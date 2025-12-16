import numpy as np
import os

def create_fractured_bone_phantom(shape=(100, 100, 100), radius=30, fracture_gap=5):
    """
    Creates a 3D numpy array representing a bone with a fracture.
    HU values: Air = -1000, Bone = 1000.
    """
    # Initialize with Air
    volume = np.full(shape, -1000, dtype=np.int16)
    
    center = np.array(shape) // 2
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Create a Cylinder (Bone) along Z-axis
    # bone_mask is 2D (Y, X), needs to be broadcast to 3D (Z, Y, X)
    dist_from_center = np.sqrt((x - center[2])**2 + (y - center[1])**2)
    bone_mask_2d = dist_from_center <= radius
    # Broadcast to 3D: (1, Y, X) -> (Z, Y, X)
    bone_mask_3d = np.broadcast_to(bone_mask_2d, shape)
    
    volume[bone_mask_3d] = 1000
    
    # Create a Fracture (Air gap) in the middle
    z_center = center[0]
    # fracture_mask is 1D (Z), needs to be broadcast
    fracture_mask_1d = (z >= z_center - fracture_gap) & (z <= z_center + fracture_gap)
    fracture_mask_3d = np.broadcast_to(fracture_mask_1d, shape)
    
    # Apply fracture (set back to Air)
    # Intersection of bone and fracture zone
    volume[bone_mask_3d & fracture_mask_3d] = -1000
    
    return volume

if __name__ == "__main__":
    output_dir = "prototypes/3d_reconstruction/data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔨 Generating mock fractured bone data...")
    volume = create_fractured_bone_phantom()
    
    output_path = os.path.join(output_dir, "mock_bone_volume.npy")
    np.save(output_path, volume)
    
    print(f"✅ Saved mock volume to: {output_path}")
    print(f"   Shape: {volume.shape}")
    print(f"   Min/Max HU: {volume.min()}/{volume.max()}")
