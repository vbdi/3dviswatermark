import pickle
import trimesh
import glob
import os, sys
import watermarker
import filtering
import utils
import helper
import time
from color_watermark import create_colored_obj
from join_obj_texture import merge_obj_models
import torch
import roughness
import numpy as np
import json
import pandas as pd
pd.set_option('display.max_columns', None)
import argparse
import random
import shutil

from bpy_tools import clean_scene, trimesh_to_blender, save_all_objects, get_distance_variance, adjust_watermark_core, add_random_color, blender_to_trimesh

ap = argparse.ArgumentParser("Watermarking script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--input_folder", type=str, default="test_inputs", help="Input folder")
ap.add_argument("--output_folder", type=str, default="test_outputs", help="Output folder")
ap.add_argument("--watermark", type=str, default="watermark", help="watermark text")
ap.add_argument("--wm_size", type=float, default=0.13, help="ratio of wm size relative to model size")
ap.add_argument("--wm_thickness", type=float, default=0.5, help="Watermark thickness")
ap.add_argument("--fix_num_waters", type=int, default=0, help="0 for adaptive number of watermark, 1 for fixing the number of watermarks")
ap.add_argument("--num_waters", type=int, default=8, help="number of watermark if not adaptive")
ap.add_argument("--num_per_octant", type=int, default=1, help="max number of watermark per octant if not adaptive")
args = ap.parse_args()

debug = False
model_size = 30
max_water_samples = 300
vertices_threshold = 80000
watermark_text = args.watermark
input_folder = args.input_folder
output_folder = args.output_folder
wm_size = args.wm_size*model_size
thickness = args.wm_thickness
num_waters = args.num_waters
num_per_octant = args.num_per_octant
fix_num_waters = args.fix_num_waters

shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)
start_time = time.time()
text_3d, box_3d, area_long_face = helper.load_text(watermark_text, scale=wm_size, initial_height=thickness)
all_model_folders = sorted(glob.glob(f'{input_folder}/*'))
for midx, input_model_folder in enumerate(all_model_folders):
    start_time_ = time.time()
    model_name = os.path.basename(input_model_folder)
    print(f'\n\n\n Starting model {model_name} ({midx+1}/ {len(all_model_folders)})')
    output_model_folder = f'{output_folder}/{model_name}'
    os.makedirs(output_model_folder, exist_ok=True)
    obj_filepath = glob.glob(f'{input_model_folder}/*.obj')[0]
    # shutil.copyfile(obj_filepath, f'{output_model_folder}/{os.path.basename(obj_filepath)}')
    mesh_, mesh_tri_ = helper.load_mesh(obj_filepath, size=model_size)
    print(f'{os.path.basename(obj_filepath)}. Vertices - {mesh_tri_.vertices.__len__()}. Faces - {mesh_tri_.faces.__len__()}.')
    if len(mesh_tri_.vertices) > vertices_threshold:
        mesh = mesh_.decimate_pro(1- vertices_threshold/mesh_.points.__len__())
        mesh_tri = utils.pv_to_tri(mesh)
        print(f'Mesh decimated. Vertices - {mesh_tri.vertices.__len__()}. Faces - {mesh_tri.faces.__len__()}.')
    else:
        mesh = mesh_
        mesh_tri = mesh_tri_

    output_obj_filepath = f'{output_model_folder}/{os.path.basename(obj_filepath)}'
    mesh_tri.export(output_obj_filepath)

    try:
        ''' watermark sampling '''
        # start_time_model = time.time()
        sampled_points, sampled_normals, _, _ =  helper.sample_mesh(mesh_tri, count=max_water_samples, radius=1)
        if debug:
            trimesh.Scene([mesh_tri, trimesh.points.PointCloud(sampled_points)]).show(flags={'wireframe': False})
        box3d_verts_trans, text3d_verts_trans = helper.create_wateramrks_vectorized(text_3d, box_3d, sampled_normals, sampled_points)
        print(f'Sampled points {box3d_verts_trans.__len__()}.')

        ''' OPTIMIZATION CODE  '''
        box3d_verts_trans_, text3d_verts_trans_, losses, box3d_verts_trans, mesh_py3d = (
            watermarker.optimize_candidates(mesh_tri, box3d_verts_trans, text3d_verts_trans))
        text3d_verts_trans_ = text3d_verts_trans_.cpu()
        box3d_verts_trans_ = box3d_verts_trans_.cpu()

        optimized_boxes = [trimesh.Trimesh(vertices=verts.numpy(), faces=box_3d.faces) for verts in box3d_verts_trans_.cpu()]
        initial_boxes = [trimesh.Trimesh(vertices=verts.numpy(), faces=box_3d.faces) for verts in box3d_verts_trans.cpu()]
        if debug:
            helper.plot_multi([[mesh_tri] + initial_boxes, [mesh_tri] + optimized_boxes])

        ''' compute the area and roughness scores '''
        box3d_normals_trans_ = torch.from_numpy(filtering.compute_normals(optimized_boxes)[0]).double()
        inside_area, inside_faces, intersect_faces, inside_verts = helper.compute_inside_area_vectorized(box3d_verts_trans_.double().cpu(), box3d_normals_trans_, mesh_tri)
        avg_cos_sims = np.array([np.mean(roughness.get_cossims(mesh_tri.vertex_normals[verts])) for verts in inside_verts])
        final_losses = losses[-1]

        ''' box data preparation '''
        for idx in range(len(optimized_boxes)):
            optimized_boxes[idx].metadata = {
                'loss': final_losses[idx].numpy(),
                'inside_area': inside_area[idx].numpy()/ area_long_face,
                'cos_sim': avg_cos_sims[idx],
                'inside_verts': inside_verts[idx],
                'water_mesh': trimesh.Trimesh(vertices= text3d_verts_trans_[idx].numpy(), faces=text_3d.faces)
            }

        # ''' playground '''
        # metric_name = 'ray_score'
        # metric = np.array([box.metadata[metric_name] for box in optimized_boxes])
        # high_waters = [optimized_boxes[idx] for idx in np.argsort(metric)[::-1][:5]]
        # low_waters = [optimized_boxes[idx] for idx in np.argsort(metric)[:5]]
        # helper.plot_multi([[mesh_tri] + high_waters, [mesh_tri] + low_waters])

        ''' FILTERING CODE  '''
        filter_boxes_rb = filtering.metric_thresh(optimized_boxes, metric='cos_sim', thresh=0.80, order='+')
        if debug:
            helper.plot_multi([[mesh_tri] + optimized_boxes, [mesh_tri] + filter_boxes_rb])

        filter_boxes_lt = filtering.metric_thresh(filter_boxes_rb, metric ='loss', thresh=0.005, order='-')
        if debug:
            helper.plot_multi([[mesh_tri] + filter_boxes_rb, [mesh_tri] + filter_boxes_lt])

        filter_boxes_no = filtering.non_overlapping(filter_boxes_lt)
        if debug:
            helper.plot_multi([[mesh_tri] + filter_boxes_lt, [mesh_tri] + filter_boxes_no])

        start_time__ = time.time() # most likely we can improve this time's step using vectorization
        decision_results, percent_results = helper.find_ray_intersections(mesh_tri, filter_boxes_no)
        print(f'Time ray intersections {time.time()-start_time__}') # takes ~40s alone
        filter_bool = percent_results<0.0001
        filter_boxes_rt = np.array(filter_boxes_no)[filter_bool].tolist()
        print(f'Filter ray test yielded {np.mean(filter_bool)} which are {np.sum(filter_bool)} boxes')
        if debug:
            helper.plot_multi([[mesh_tri] + filter_boxes_lt, [mesh_tri] + filter_boxes_rt])

        filter_boxes_oc, filtered_idx_oc = filtering.octant_based(mesh_tri, filter_boxes_rt, num_per_octant)
        if debug:
            helper.plot_multi([[mesh_tri] + filter_boxes_rt, [mesh_tri] + filter_boxes_oc])

        filter_boxes_vb = filtering.view_based(filter_boxes_rt, filtered_idx_oc, angle_cutoff=45)
        if debug:
            helper.plot_multi([[mesh_tri] + filter_boxes_oc, [mesh_tri] + filter_boxes_vb + filter_boxes_oc])

        filter_boxes = filter_boxes_oc + filter_boxes_vb
        if fix_num_waters==1 and num_waters<=len(filter_boxes):
            random.seed(42)
            filter_boxes = random.sample(filter_boxes, num_waters)

        assert(len(filtering.non_overlapping(filter_boxes)) == len(filter_boxes))
        if len(filter_boxes) == 0:
            print(f'\n Skipping {model_name} ({midx+1}/ {len(all_model_folders)}). No watermark found.')
            continue

        # filter_boxes = filter_boxes_oc
        filter_waters = [box.metadata['water_mesh'] for box in filter_boxes]

        ''' removed overlapped watermarks '''
        # filter_boxes, filter_waters = filtering.remove_overlapped(filter_boxes, filter_waters)        

        ''' calculate the uneven score '''
        clean_scene()
        orig_bpy_mesh = trimesh_to_blender(mesh_tri, model_name)
        all_dist_var = []
        for idx, water in enumerate(filter_waters):
            print(f"calculate the uneven score for wm {idx}")
            water_name = f'watermark_{idx:03}'
            bpy_water = trimesh_to_blender(water, water_name)
            all_dist_var.append(get_distance_variance(orig_bpy_mesh, bpy_water))

        uneven_score = np.mean(np.array(all_dist_var))
        print(f"uneven_score = {uneven_score}")


        ''' display stats of watermarks '''
        data = [box.metadata for box in filter_boxes]
        data = pd.DataFrame(data)[['loss', 'inside_area', 'cos_sim']]
        if debug:
            print(data)

        ''' text 3d watermarks save '''
        print('text 3d watermarks save')
        untextured_folder = f'{output_model_folder}/untextured_watermarks'
        os.makedirs(untextured_folder, exist_ok=True)
        filter_waters = [utils.tri_translate(mesh_water, mesh_box.centroid - mesh_water.bounding_box_oriented.centroid) for mesh_box, mesh_water in zip(filter_boxes, filter_waters)]
        # [utils.plot([filter_boxes[idx], filter_waters[idx]], wireframe=True) for idx in range(len(filter_boxes))]
        [box.export(f'{untextured_folder}/{model_name}_watermarks_only_{idx}.obj') for idx, box in enumerate(filter_waters)]

        ''' adjust watermarks for visualization'''
        # print('adjust watermarks for visualization')
        output_obj_filepath_colored, _ = create_colored_obj(output_obj_filepath, [211, 211, 211], output_model_folder, 'orig_colored')

        ''' color watermark meshes '''
        print('color watermark meshes')
        textured_folder = f'{output_model_folder}/colored_watermarks/'
        os.makedirs(textured_folder, exist_ok=True)
        for uncolor_water_file in sorted(glob.glob(f'{untextured_folder}/{model_name}_watermarks_only_*.obj')):
            create_colored_obj(uncolor_water_file, [255,0,0], textured_folder)


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
        [box.export(f'{untextured_folder}/{model_name}_boxes_{idx}.obj') for idx, box in enumerate(filter_boxes)]
        [box.export(f'{untextured_folder}/{model_name}_watermarks_only_{idx}.obj') for idx, box in enumerate(filter_waters)]

        # normals
        normals, lf_dirs, sf_dirs = filtering.compute_normals(filter_boxes)
        [pickle.dump(normal, open(f'{untextured_folder}/{model_name}_normal_{idx}.obj', 'wb')) for idx, normal in enumerate(normals)]

        # partitions
        partition_folder = f'{output_model_folder}/partitions/'
        os.makedirs(partition_folder, exist_ok=True)
        _,_,_,_,octants = helper.split_model_into_octants(mesh_tri)
        [octant.export(f'{partition_folder}/sliced_{model_name}_octant_{idx}.obj') for idx, octant in enumerate(octants)]

        print(f'\n DONE model {os.path.basename(input_model_folder)} - {midx+1}/{len(all_model_folders)}. Time taken {time.time()-start_time_}s')

    except Exception as e:
        print(f'Exception in {os.path.basename(input_model_folder)} - {midx+1}/{len(all_model_folders)}')
        print(e)

print(f'\n\n\nALL DONE. Time Taken {round(time.time()-start_time,2)}s\n\n\n')
