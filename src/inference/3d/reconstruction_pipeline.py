import numpy as np
import trimesh
from skimage.measure import marching_cubes
import os

def reconstruct_bone_mesh(volume_path, output_path, threshold=200):
    """
    Reconstructs a 3D mesh from a volume using Marching Cubes.
    """
    print(f"📂 Loading volume from: {volume_path}")
    volume = np.load(volume_path)
    
    print(f"⚙️ Running Marching Cubes (Threshold: {threshold})...")
    # Marching Cubes: Extract isosurface at the given threshold
    verts, faces, normals, values = marching_cubes(volume, level=threshold)
    
    print(f"   Vertices: {len(verts)}")
    print(f"   Faces: {len(faces)}")
    
    # Create Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Optional: Smooth the mesh slightly to reduce pixelation artifacts
    # mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3)
    
    print(f"💾 Exporting mesh to: {output_path}")
    mesh.export(output_path)
    print("✨ Done!")

if __name__ == "__main__":
    base_dir = "prototypes/3d_reconstruction/data"
    input_file = os.path.join(base_dir, "mock_bone_volume.npy")
    output_file = os.path.join(base_dir, "reconstructed_bone.obj")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("   Run mock_data_gen.py first!")
    else:
        reconstruct_bone_mesh(input_file, output_file)
