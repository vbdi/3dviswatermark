import helper
import numpy as np
import torch
import trimesh

def remove_overlapped(watermark_boxes, watermark_meshes):
    filtered_meshes = []
    filtered_boxes = []

    for box, water in zip(watermark_boxes, watermark_meshes):
        if len(filtered_meshes) == 0:
            filtered_meshes.append(water)
            filtered_boxes.append(box)
            continue

        is_intersect = False
        for filter_mesh in filtered_meshes:
            if water.intersection(filter_mesh).volume > 0.0001:
                is_intersect = True
            if is_intersect:
                break

        if is_intersect:
            continue
        else:
            filtered_meshes.append(water)
            filtered_boxes.append(box)

    return filtered_boxes, filtered_meshes

def non_overlapping(boxes):
    if len(boxes) == 0:
        return boxes
    filtered_boxes = []
    selected_verts = np.array([])
    inside_verts = [box.metadata['inside_verts'] for box in boxes]
    final_losses = [box.metadata['loss'] for box in boxes]
    sorted_idx = np.argsort(final_losses)
    for idx in sorted_idx:
        inside_verts_box = inside_verts[idx]
        if len(np.intersect1d(selected_verts, inside_verts_box)) == 0:
            selected_verts = np.append(selected_verts, inside_verts_box)
            filtered_boxes.append(boxes[idx])
    print(f'Filtering by non overlapping yielded {len(filtered_boxes)/len(boxes)} boxes which are {len(filtered_boxes)} boxes')
    return filtered_boxes

def metric_thresh(boxes, metric, thresh, order='-'):
    if len(boxes) == 0:
        return boxes
    final_losses = np.array([box.metadata[metric] for box in boxes])
    if order == '-':
        filter_decisions = final_losses < thresh
    else:
        filter_decisions = final_losses > thresh
    filtered_boxes = np.array(boxes)[filter_decisions]
    print(f'Filtering by {metric}{order} threshold yieleded {np.mean(filter_decisions)} boxes which are {np.sum(filter_decisions)} boxes')
    return filtered_boxes.tolist()

def compute_normals(final_boxes):
    box3d_verts = torch.cat([torch.Tensor(box.vertices).unsqueeze(0) for box in final_boxes], 0)
    b1_vecs = box3d_verts[:, 0, :]
    b2_vecs = box3d_verts[:, 4, :]
    b3_vecs = box3d_verts[:, 6, :]
    b4_vecs = box3d_verts[:, 2, :]
    t1_vecs = box3d_verts[:, 1, :]
    t2_vecs = box3d_verts[:, 5, :]
    t3_vecs = box3d_verts[:, 7, :]
    t4_vecs = box3d_verts[:, 3, :]

    lf_sizes = torch.linalg.norm(b2_vecs - b1_vecs, dim=1)
    sf_sizes = torch.linalg.norm(b4_vecs - b1_vecs, dim=1)
    hf_sizes = torch.linalg.norm(t1_vecs - b1_vecs, dim=1)
    lf_dirs = (b2_vecs-b1_vecs)/lf_sizes.unsqueeze(1)
    sf_dirs = (b4_vecs-b1_vecs)/sf_sizes.unsqueeze(1)
    hf_dirs = (t1_vecs - b1_vecs)/hf_sizes.unsqueeze(1)
    return hf_dirs.numpy(), lf_dirs.numpy(), sf_dirs.numpy()

def octant_based(mesh_tri, boxes, num_per_octant=1):
    if len(boxes) == 0:
        return boxes, []
    # mesh_tri = trimesh.load('temp_data/dolphin.obj', force='mesh')
    # filter_boxes = np.array([trimesh.load(f'temp_data/water_{idx}.obj', force='mesh') for idx in range(30)])
    # box3d_normals,_,_ = compute_normals(filter_boxes)
    # utils.plot([mesh_tri]+filter_boxes.tolist(), use_colors=True, wireframe=False)
    box_centroids=np.vstack([box.centroid for box in boxes])
    _,_,_,_,octants = helper.split_model_into_octants(mesh_tri)
    octants = [octant for octant in octants if len(octant.vertices)>0]
    # [octant.show() for octant in octants]
    octant_dists = [trimesh.proximity.closest_point(octant, box_centroids)[1] for octant in octants]
    closest_octant = np.argmin(np.vstack(octant_dists),0)
    # [utils.plot([octants[idx]]+filter_boxes[closest_octant==idx].tolist(), use_colors=True, wireframe=False) for idx in range(len(octants))]
    watermark_count = {idx:np.sum(closest_octant==idx) for idx in np.sort(np.unique(closest_octant))}
    print(f'Filtering octants - watermarks found per octant {watermark_count}')
    pair_wise_dists = lambda boxes1,boxes2: np.abs(np.expand_dims(np.array([box.centroid for box in boxes1]), 1) -
                               np.expand_dims(np.array([box.centroid for box in boxes2]), 0)).mean(2).mean(1)

    ''' OCTANT FILTERING CODE '''
    filtered_boxes = []
    filtered_idx = []
    for oidx in range(len(octants)):
        octant_waters_idx = np.where(closest_octant==oidx)[0]
        if len(octant_waters_idx) >= num_per_octant:
            octant_waters = np.array(boxes)[octant_waters_idx]
            if len(filtered_boxes) == 0:
                filter_idxes = np.random.choice(octant_waters_idx, num_per_octant,replace=False)
            else:
                dists = pair_wise_dists(octant_waters, filtered_boxes)
                filter_idxes = octant_waters_idx[np.argpartition(dists, 0-num_per_octant)[0-num_per_octant:]]
                # filter_idx = np.random.choice(octant_waters_idx, 1)

            for filter_idx in filter_idxes:
                filtered_box = boxes[int(filter_idx)]
                filtered_boxes.append(filtered_box)
                filtered_idx.append(int(filter_idx))
        else:
            print(f'Not enough watermark for octant {oidx}. Adding all watermarks in this octant.')
            for filter_idx in octant_waters_idx:
                filtered_box = boxes[int(filter_idx)]
                filtered_boxes.append(filtered_box)
                filtered_idx.append(int(filter_idx))

    print(f'Avg box distance {np.mean(pair_wise_dists(filtered_boxes, filtered_boxes))}')
    print(f'Filtering octant based yielded {len(filtered_boxes)/ len(boxes)} which are {len(filtered_boxes)} boxes')
    return filtered_boxes, filtered_idx

def view_based(all_boxes, filter_idx=None, angle_cutoff=45):
    if len(all_boxes) == 0:
        return all_boxes

    angle_cutoff1 = angle_cutoff
    angle_cutoff2 = angle_cutoff
    added_boxes = []
    for axis in [[0, 1, 0], [1, 0, 0]]:
        for angle in range(0, 360, 30):
            # print(f'Angle {axis} {angle}')
            R = trimesh.transformations.rotation_matrix(np.deg2rad(angle), axis)
            final_boxes_ = [box.copy().apply_transform(R) for box in all_boxes]
            box3d_normals_, lf_dirs_, sf_dirs_ = compute_normals(final_boxes_)
            normal_scores_ = np.dot(box3d_normals_, [0, 0, 1])
            normal_angles_ = np.rad2deg(np.arccos(normal_scores_))
            if any(normal_angles_[filter_idx] < angle_cutoff1):
                continue  # if a watermark is visible already just ignore this view
            order_index = np.argsort(normal_angles_)
            for idx in order_index:
                box_angle = normal_angles_[idx]
                if box_angle < angle_cutoff2:
                    # np.append(filter_idx, idx)
                    added_boxes.append(all_boxes[idx])
                    filter_idx = filter_idx + [idx]
                    break
    # print(len(filter_idx))
    # utils.plot([mesh_tri]+filter_boxes[filter_idx].tolist(), use_colors=True, wireframe=False)
    print(f'Filtering view based. More watermarks added {len(added_boxes)}')
    return added_boxes
