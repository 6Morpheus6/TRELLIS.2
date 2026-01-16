# export_script.py
import os
import sys
import argparse
import pickle
import torch
import o_voxel

def run_export(input_path, output_path):
    print(f"[Subprocess] Loading data from {input_path}...")
    
    # Daten laden (Numpy Arrays)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print("[Subprocess] Converting data back to CUDA Tensors...")
    
    # WICHTIG: Numpy Arrays wieder in PyTorch Tensors auf der GPU umwandeln
    # o_voxel benötigt zwingend CUDA Tensors für die Berechnungen
    vertices = torch.from_numpy(data['vertices']).cuda()
    faces = torch.from_numpy(data['faces']).cuda()
    attr_volume = torch.from_numpy(data['attr_volume']).cuda()
    coords = torch.from_numpy(data['coords']).cuda()

    print("[Subprocess] Running o_voxel postprocessing...")
    
    glb = o_voxel.postprocess.to_glb(
        vertices=vertices,
        faces=faces,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=data['attr_layout'],
        grid_size=data['grid_size'],
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=data['decimation_target'],
        texture_size=data['texture_size'],
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    print(f"[Subprocess] Exporting to {output_path}...")
    glb.export(output_path, extension_webp=True)
    print("[Subprocess] Done. Exiting and releasing memory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    try:
        run_export(args.input, args.output)
    except Exception as e:
        # Gibt den detaillierten Fehler an den Hauptprozess zurück
        print(f"[Subprocess Error] {e}")
        # Traceback auch drucken, falls nötig
        import traceback
        traceback.print_exc()
        sys.exit(1)