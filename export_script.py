import os
import sys
import argparse
import pickle
import numpy as np
import torch
import o_voxel
import trimesh
from pygltflib import GLTF2

def patch_glb(filename):
    try:
        gltf = GLTF2().load(filename)
        vertex_views, index_views, image_views = set(), set(), set()
        for mesh in gltf.meshes:
            for primitive in mesh.primitives:
                for attr in primitive.attributes.__dict__.values():
                    if attr is not None:
                        vertex_views.add(gltf.accessors[attr].bufferView)
                if primitive.indices is not None:
                    index_views.add(gltf.accessors[primitive.indices].bufferView)
        for image in gltf.images:
            if image.bufferView is not None:
                image_views.add(image.bufferView)
        for i, bv in enumerate(gltf.bufferViews):
            if i in vertex_views: bv.target = 34962
            elif i in index_views: bv.target = 34963
            elif i in image_views: bv.target = None
            else: bv.target = None
        gltf.save(filename)
        print("[Subprocess] Header repair complete.")
    except Exception as e:
        print(f"[Subprocess Error] Patching failed: {e}")

def run_export(input_path, output_path):
    print(f"[Subprocess] Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print("[Subprocess] Converting data to CUDA...")
    # Wir schieben es hier direkt auf CUDA, da du 16GB hast und wir in app.py nicht swappen
    v = torch.from_numpy(data['vertices']).cuda().float() 
    f = torch.from_numpy(data['faces']).cuda().int()
    a = torch.from_numpy(data['attr_volume']).cuda().float()
    c = torch.from_numpy(data['coords']).cuda().float()

    print("[Subprocess] Running o_voxel postprocessing...")
    print(f"[Subprocess] UV params: cone_angle={data.get('uv_cone_angle', 90.0)}, refine={data.get('uv_refine_iterations', 0)}, global={data.get('uv_global_iterations', 1)}, smooth={data.get('uv_smooth_strength', 1)}")
    glb = o_voxel.postprocess.to_glb(
        vertices=v, faces=f, attr_volume=a, coords=c,
        attr_layout=data['attr_layout'], grid_size=data['grid_size'],
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=data['decimation_target'],
        texture_size=data['texture_size'],
        remesh=True, use_tqdm=True,
        # UV unwrap parameters (PozzettiAndrea/ComfyUI-TRELLIS2 nodes/nodes_unwrap.py:157-160)
        # Defaults match cumesh.CuMesh.compute_charts() defaults
        mesh_cluster_threshold_cone_half_angle_rad=np.radians(data.get('uv_cone_angle', 90.0)),
        mesh_cluster_refine_iterations=data.get('uv_refine_iterations', 100),
        mesh_cluster_global_iterations=data.get('uv_global_iterations', 3),
        mesh_cluster_smooth_strength=data.get('uv_smooth_strength', 1),
    )

    # --- DER ROBUSTE FIX ---
    print("[Subprocess] Fixing Geometry Normals...")
    # Wir prüfen, ob es eine Scene oder ein einzelnes Mesh ist
    if isinstance(glb, trimesh.Scene):
        for geometry_name in glb.geometry:
            m = glb.geometry[geometry_name]
            m.update_faces(m.nondegenerate_faces())
            m.fix_normals()
    else:
        # Es ist direkt ein Trimesh Objekt
        glb.update_faces(glb.nondegenerate_faces())
        glb.fix_normals()
    # --- FIX ENDE ---

    print(f"[Subprocess] Exporting to {output_path}...")
    glb.export(output_path, extension_webp=False)
    
    patch_glb(output_path)
    print("[Subprocess] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    try:
        run_export(args.input, args.output)
    except Exception as e:
        print(f"[Subprocess Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)