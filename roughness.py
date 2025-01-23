import numpy as np
import utils, helper
from collections import defaultdict
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
device=torch.device("cuda:0")

def get_faces_area(faces, areamap):
    faces = list(set(faces))
    areas = [areamap[face] for face in faces]
    return np.sum(areas)

def update_maps(id, vertices, faces, vertexmap, facemap, areamap, target_area):
    vertices.update(vertexmap[id])
    faces.update(facemap[id])
    area = get_faces_area(faces, areamap)
    if area > target_area:
        return -1
    else:
        return 1

def find_points_bounded_by_area(ver_id, vertexmap, facemap, areamap, target_area):
    sel_vertices = set()
    sel_faces = set()
    processed = set()
    queue = list(vertexmap[ver_id])
    while(1):
        if len(queue)>0:
            id = queue.pop(0)
            if id not in processed:
                processed.add(id)
                queue = queue + list(vertexmap[id])
                res = update_maps(id, sel_vertices, sel_faces, vertexmap, facemap, areamap, target_area)
                if res == -1:
                    break
        else:
            break
    return np.array(list(sel_vertices)), np.array(list(sel_faces))

def get_cossims(vector_list, max_sample_size=200):
    if len(vector_list) == 0:
        return [0]
    if len(vector_list.shape) == 1:
        vector_list = np.expand_dims(vector_list,0)
    idx = np.random.choice(len(vector_list), min(max_sample_size, len(vector_list)), replace=False)
    vector_list = vector_list[idx]
    cos_sims = cosine_similarity(vector_list, vector_list)
    return cos_sims

def estimate_roughness_area(mesh, target_area):
    avg_sims = []
    std_sims = []
    min_sims = []
    vertexmap, facemap = get_onehop_maps(mesh)
    areamap = utils.pv_to_tri(mesh).area_faces
    for point_idx in range(mesh.points.__len__()):
        nei_points, nei_faces = find_points_bounded_by_area(point_idx, vertexmap, facemap, areamap, target_area)
        nei_normals = np.array(mesh.point_data['Normals'])[nei_points]
        cos_sims = get_cossims(nei_normals)
        avg_sims.append(np.mean(cos_sims))
        std_sims.append(np.std(cos_sims))
        min_sims.append(np.std(cos_sims))
    return np.array(avg_sims), np.array(std_sims)

def get_onehop_maps(mesh):
    # IMPLEMENT HASH MAP IDEA
    mesh_tri = utils.pv_to_tri(mesh)
    facemap = defaultdict(list)
    vertexmap = defaultdict(set)
    for idx, face in enumerate(mesh_tri.faces):
        v1,v2,v3 = face
        facemap[v1].append(idx)
        facemap[v2].append(idx)
        facemap[v3].append(idx)
        vertexmap[v1].update(face)
        vertexmap[v2].update(face)
        vertexmap[v3].update(face)
    return vertexmap, facemap


