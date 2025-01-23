import trimesh
import glob
import os
import time
from color_watermark import create_colored_obj
from join_obj_texture import merge_obj_models
import numpy as np
import json
import pandas as pd
import argparse
import distutils
from bpy_tools import clean_scene, trimesh_to_blender, save_all_objects, get_distance_variance, adjust_watermark_core, add_random_color, blender_to_trimesh
import shutil
pd.set_option('display.max_columns', None)


ap = argparse.ArgumentParser("Watermarking script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--input_folder", type=str, default="test_inputs", help="Input folder")
ap.add_argument("--ours_folder", type=str, default="test_outputs", help="Folder with watermarks before embossing")
ap.add_argument("--output_folder", type=str, default="test_outputs-curve", help="Output folder")

args = ap.parse_args()

debug = False
input_folder = args.input_folder
output_folder = args.output_folder
ours_folder = args.ours_folder

shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)
start_time = time.time()
# text_3d, box_3d, area_long_face = helper.load_text(watermark_text, scale=wm_size, initial_height=thickness)
all_model_folders = sorted(glob.glob(f'{input_folder}/*'))
for midx, input_model_folder in enumerate(all_model_folders):
    start_time_ = time.time()
    model_name = os.path.basename(input_model_folder)
    print(f'\n\n\n Starting model {model_name} ({midx+1}/ {len(all_model_folders)})')
    output_model_folder = f'{output_folder}/{model_name}'
    os.makedirs(output_model_folder, exist_ok=True)
    ours_model_folder = f'{ours_folder}/{model_name}'

    try:
        distutils.dir_util.copy_tree(ours_model_folder, output_model_folder)
        output_obj_filepath = [file for file in glob.glob(f'{output_model_folder}/{model_name}*.obj') if 'watermark' not in file][0]
        print(f'loading output_obj_filepath {output_obj_filepath}')
        mesh_tri = trimesh.load(output_obj_filepath, force='mesh')
        waters_path_pat = os.path.join(output_model_folder, 'untextured_watermarks/*_watermarks_only_*.obj')
        filter_waters = [trimesh.load(water_file, force="mesh") for water_file in sorted(glob.glob(waters_path_pat))]

        # trimesh.Scene([mesh_tri] + filter_waters).show()
        # utils.plot([mesh_tri] + filter_waters)

        ''' curve following codes '''
        uneven_score_before = 0.0
        uneven_score_after = 0.0        
        clean_scene()
        orig_bpy_mesh = trimesh_to_blender(mesh_tri, model_name)
        all_bpy_waters = []
        all_dist_var_old = []
        all_dist_var_new = []
        for idx, water in enumerate(filter_waters):
            water_name = f'watermark_{idx:03}'
            bpy_water = trimesh_to_blender(water, water_name)
            dist_var_old = get_distance_variance(orig_bpy_mesh, bpy_water)
            bpy_water_new = adjust_watermark_core(bpy_water, orig_bpy_mesh, shift_z=0.05)
            dist_var_new = get_distance_variance(orig_bpy_mesh, bpy_water_new)
            # add_random_color(bpy_water_new)
            # print(f'dist_var_old is {dist_var_old:.06f} and dist_var_new is {dist_var_new:0.6f}')
            all_bpy_waters.append(bpy_water_new)
            all_dist_var_old.append(dist_var_old)
            all_dist_var_new.append(dist_var_new)

            print(f'dist_var_old is {np.mean(np.array(all_dist_var_old)):.06f} and dist_var_new is {np.mean(np.array(dist_var_new)):0.6f}')
            # save_all_objects(f"{model_name}_watermarked.obj")

            filter_waters = [blender_to_trimesh(bpy_water) for bpy_water in all_bpy_waters]
            uneven_score_before = np.mean(np.array(all_dist_var_old))
            uneven_score_after = np.mean(np.array(all_dist_var_new))

        # trimesh.Scene([mesh_tri] + filter_waters).show()
        # utils.plot([mesh_tri] + filter_waters)

        ''' text 3d watermarks save '''
        untextured_folder = f'{output_model_folder}/untextured_watermarks'
        [box.export(f'{untextured_folder}/{model_name}_watermarks_only_{idx}.obj') for idx, box in enumerate(filter_waters)]

        ''' color meshes '''
        textured_folder = f'{output_model_folder}/colored_watermarks/'
        os.makedirs(textured_folder, exist_ok=True)
        for uncolor_water_file in sorted(glob.glob(f'{untextured_folder}/{model_name}_watermarks_only_*.obj')):
            create_colored_obj(uncolor_water_file, [255,0,0], textured_folder)
        output_obj_filepath_colored = output_obj_filepath

        ''' EXECUTE BLENDER MERGING '''
        obj_files =  [output_obj_filepath_colored] + sorted(glob.glob(os.path.join(textured_folder, "*_clr.obj")))
        output_path = os.path.join(output_model_folder, f"{model_name}_watermarked.obj")
        merge_obj_models(obj_files, output_path, boolean=False)
        print(f"Watermarked model exported to {output_path}\n")
        output_boolean_dir = os.path.join(output_model_folder, 'boolean')
        os.makedirs(output_boolean_dir, exist_ok=True)
        output_path_boolean = os.path.join(output_boolean_dir, f"{model_name}_watermarked_boolean.obj")
        merge_obj_models(obj_files, output_path_boolean, boolean=True)
        print(f"Boolean watermarked model exported to {output_path_boolean}\n")
        total_runtime = time.time() - start_time_

        # watermark boxes and watermarks
        [box.export(f'{untextured_folder}/{model_name}_watermarks_only_{idx}.obj') for idx, box in enumerate(filter_waters)]

        print(f'\n DONE model {os.path.basename(input_model_folder)} - {midx+1}/{len(all_model_folders)}. Time taken {time.time()-start_time_}s')

    except Exception as e:
        print(f'Exception in {os.path.basename(input_model_folder)} - {midx+1}/{len(all_model_folders)}')
        print(e)

print(f'\n\n\nALL DONE. Time Taken {round(time.time()-start_time,2)}s\n\n\n')
