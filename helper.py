from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import point_mesh_distance
import torch
import pyvista as pv
import utils
import trimesh
import numpy as np
import pytorch3d
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch3d import _C
import time, os
import glob



def find_ray_intersections(mesh_tri, boxes, visualize=False):
    decision_results = []
    percent_results = []
    index_results = []
    ray_data = []
    for box in boxes:
        [b1, t1, b4, t4, b2, t2, b3, t3] = box.vertices
        lf_size = np.linalg.norm(t2 - t1)
        sf_size = np.linalg.norm(t4 - t1)
        hf_size = np.linalg.norm(t1 - b1)
        lf_dir = (t2 - t1) / lf_size
        sf_dir = (t4 - t1) / sf_size
        hf_dir = (t1 - b1) / hf_size
        lf_count_sample = 10
        sf_count_sample = 4
        lf_samples3 = [t1 + sf_dir * (sf_size / sf_count_sample * i) for i in range(sf_count_sample + 1)]
        lf_samples4 = [t2 + sf_dir * (sf_size / sf_count_sample * i) for i in range(sf_count_sample + 1)]
        lf_samples1 = [x + lf_dir * (lf_size / lf_count_sample * i) for i in range(lf_count_sample + 1) for x in lf_samples3]
        lf_samples2 = [x + lf_dir * (lf_size / lf_count_sample * i) for i in range(lf_count_sample + 1) for x in lf_samples3]

        ray_origins = np.array(lf_samples1 + lf_samples2+ lf_samples3+ lf_samples4)
        ray_origins = ray_origins + hf_dir*0.001 # to prevent ray origins to strike the letters where they start
        ray_directions = np.array([hf_dir for i in range(len(ray_origins))])

        locations, index_ray, index_tri = mesh_tri.ray.intersects_location(
                                            ray_origins=ray_origins,
                                            ray_directions=ray_directions)

        decisions = np.array([True if i in np.unique(index_ray) else False for i in range(len(ray_directions))])

        index_results.append(index_tri)
        decision_results.append(decisions)
        percent_results.append(np.mean(decisions))
        ray_visualize = trimesh.load_path(np.hstack((ray_origins, ray_origins + ray_directions * 0.1)).reshape(-1, 2, 3))
        ray_data.append(ray_visualize)

    # print(percent_results)
    if visualize:
        for ray_visualize, index_tri, box in zip(ray_data, index_results, boxes):
            mesh_tri.unmerge_vertices()
            # mesh_tri.visual.face_colors = [255,255,255,255]
            mesh_tri.visual.face_colors = np.ones([mesh_tri.faces.__len__(), 4]) * [255, 255, 255, 255]
            mesh_tri.visual.face_colors[index_tri] = [255, 0, 0, 255]
            # scene = trimesh.Scene([mesh_tri, ray_visualize])
            # scene = trimesh.Scene([mesh_tri, box])
            scene = trimesh.Scene([mesh_tri, box, ray_visualize])
            scene.show()
    return np.array(decision_results), np.array(percent_results)

def mesh_boxes_loss(mesh_py3d, box3d_verts):
    b1_vecs = box3d_verts[:, 0, :]
    b2_vecs = box3d_verts[:, 4, :]
    b3_vecs = box3d_verts[:, 6, :]
    b4_vecs = box3d_verts[:, 2, :]
    t1_vecs = box3d_verts[:, 1, :]
    t2_vecs = box3d_verts[:, 5, :]
    t3_vecs = box3d_verts[:, 7, :]
    t4_vecs = box3d_verts[:, 3, :]
    m1_vecs = (b1_vecs + t1_vecs) / 2
    m2_vecs = (b2_vecs + t2_vecs) / 2
    m3_vecs = (b3_vecs + t3_vecs) / 2
    m4_vecs = (b4_vecs + t4_vecs) / 2

    lf_count_sample = 10
    sf_count_sample = 5
    lf_sizes = torch.linalg.norm(m2_vecs - m1_vecs, dim=1)
    sf_sizes = torch.linalg.norm(m4_vecs - m1_vecs, dim=1)
    lf_dirs = (m2_vecs-m1_vecs)/lf_sizes.unsqueeze(1)
    sf_dirs = (m4_vecs-m1_vecs)/sf_sizes.unsqueeze(1)
    lf_samples1 = [m1_vecs + lf_dirs*(lf_sizes/lf_count_sample*i).unsqueeze(1) for i in range(lf_count_sample+1)]
    lf_samples2 = [m4_vecs + lf_dirs*(lf_sizes/lf_count_sample*i).unsqueeze(1) for i in range(lf_count_sample+1)]
    sf_samples1 = [m1_vecs + sf_dirs*(sf_sizes/sf_count_sample*i).unsqueeze(1) for i in range(sf_count_sample+1)]
    sf_samples2 = [m2_vecs + sf_dirs*(sf_sizes/sf_count_sample*i).unsqueeze(1) for i in range(sf_count_sample + 1)]

    # make point clouds
    # m_pts = torch.cat(lf_samples1 + lf_samples2 + sf_samples1 + sf_samples2).unsqueeze(0)
    # m_ptcl = pytorch3d.structures.Pointclouds(m_pts)

    lf_samples1 = torch.cat([samp.unsqueeze(0) for samp in lf_samples1], 0).transpose(0, 1)
    lf_samples2 = torch.cat([samp.unsqueeze(0) for samp in lf_samples2], 0).transpose(0, 1)
    sf_samples1 = torch.cat([samp.unsqueeze(0) for samp in sf_samples1], 0).transpose(0, 1)
    sf_samples2 = torch.cat([samp.unsqueeze(0) for samp in sf_samples2], 0).transpose(0, 1)
    m_pts = torch.cat([lf_samples1, lf_samples2, sf_samples1, sf_samples2],1)
    m_ptcl = pytorch3d.structures.Pointclouds(m_pts)
    loss = point_face_distance(mesh_py3d.extend(len(m_ptcl)), m_ptcl)
    loss = loss.view(len(m_ptcl), -1).mean(1)
    return loss

def closest_face(meshes, pcls, min_triangle_area=point_mesh_distance._DEFAULT_MIN_TRIANGLE_AREA):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    dists, idxs = _C.point_face_dist_forward(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area,
    )
    return idxs, dists

def point_face_distance(meshes, pcls, min_triangle_area=point_mesh_distance._DEFAULT_MIN_TRIANGLE_AREA):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_mesh_distance._PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    return point_to_face

def plot_pointcloud(data, title=""):
    # Sample points uniformly from the surface of the mesh.
    if isinstance(data, torch.Tensor):
        points = data
    else:
        points = sample_points_from_meshes(data, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)

def py3d_to_trimesh(mesh_py3d):
    verts = mesh_py3d.verts_packed().numpy()
    faces = mesh_py3d.faces_packed().numpy()
    mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh_tri

def trimesh_to_py3d(mesh_tri):
    verts = torch.from_numpy(np.array(mesh_tri.vertices)).float()
    faces = torch.from_numpy(np.array(mesh_tri.faces)).float()
    mesh_py3d = Meshes(verts=[verts], faces=[faces])
    return mesh_py3d

def plot_py3d(mesh_py3d):
    mesh_tri = py3d_to_trimesh(mesh_py3d)
    scene = trimesh.Scene([mesh_tri])
    scene.show()

def plot_multi(meshes_arr, use_colors=True, wireframe=False):
    colors = ['lightblue', 'red', 'blue', 'orange', 'magenta', 'white', 'cyan', 'green']
    plotter = pv.Plotter(shape=(1, 2), border=False)

    for idx, meshes in enumerate(meshes_arr):
        plotter.subplot(0, idx)
        for idx in range(len(meshes)):
            mesh = meshes[idx]
            color = colors[idx%len(colors)]
            if use_colors:
                if wireframe and idx>0:
                    _ = plotter.add_mesh(mesh, color=color, style='wireframe')
                else:
                    _ = plotter.add_mesh(mesh, color=color)
            else:
                _ = plotter.add_mesh(mesh, color='blue')
    plotter.link_views()
    plotter.show()

def plot_waters(target_mesh, optimized_waters=[], orig_waters=[], use_colors=True, wireframe='waters_only'):
    style_target = 'wireframe' if wireframe == 'all' else 'surface'
    style_waters = 'wireframe' if wireframe != 'none' else 'surface'
    plotter = pv.Plotter()
    _ = plotter.add_mesh(target_mesh, color='lightblue', style=style_target)
    for idx in range(len(orig_waters)):
        mesh = orig_waters[idx]
        _ = plotter.add_mesh(mesh, color='blue', style=style_waters)
    for idx in range(len(optimized_waters)):
        mesh = optimized_waters[idx]
        _ = plotter.add_mesh(mesh, color='red', style=style_waters)
    # plotter.show(full_screen=True)
    plotter.show()




def split_model_into_octants(input_obj, output_prefix=None):
    mesh = trimesh.load(input_obj, file_type='obj', force="mesh")
    # assumes mesh is axis aligned - otherwise it's an approximation
    # print(mesh)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_extents = np.max(mesh_vertices, 0) - np.min(mesh_vertices, 0)
    max_dist = max(mesh_extents)
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.triangles)
    # print(f"Model size = {max_dist}. Vertices {num_vertices}. Faces {num_faces}")
    centroid = np.mean(mesh.vertices, axis=0)

    # Create eight partitions centered at the centroid
    split_plane_normals = [np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([0, -1.0, 0]),
                           np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])]

    split_meshes_2 = []
    for i in [np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]:
        split_meshes_2.append(mesh.slice_plane(centroid, i))

    split_meshes_4 = []
    for split_mesh in split_meshes_2:
        for j in [np.array([0, 1.0, 0]), np.array([0, -1.0, 0])]:
            split_meshes_4.append(split_mesh.slice_plane(centroid, j))

    split_meshes_8 = []
    for split_mesh in split_meshes_4:
        for k in [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])]:
            split_meshes_8.append(split_mesh.slice_plane(centroid, k))

    # Save each octant as a separate .obj file
    octant_vertices = []
    for i, octant_mesh in enumerate(split_meshes_8):
        octant_vertices.append(len(octant_mesh.vertices))
    return max_dist, num_vertices, num_faces, octant_vertices, split_meshes_8


def load_text(text, scale=5, initial_height=0.5):
    text = pv.Text3D(text, initial_height)
    text = utils.pv_to_tri(text)
    text = utils.tri_translate(utils.tri_scale(utils.tri_scale(text), scale))
    S = trimesh.transformations.scale_matrix(1, text.centroid, [0, 0, 1])
    text = text.apply_transform(S)
    box = trimesh.creation.box(bounds=text.bounds)
    area_long_face = np.prod(np.sort(text.extents)[-2:])
    # pl = pv.Plotter()
    # pl.add_mesh(text)
    # pl.add_mesh(box, color='red', style='wireframe')
    # pl.show()
    return text, box, area_long_face


def load_text2(text, scale=5, initial_height=0.5):
    text = pv.Text3D(text, initial_height)
    text = text.translate(-np.array(text.center))
    text = utils.pv_to_tri(text)
    text = utils.tri_scale(utils.tri_scale(text), scale)
    S = trimesh.transformations.scale_matrix(1, text.centroid, [0, 0, 1])
    text = text.apply_transform(S)
    box = trimesh.creation.box(bounds=text.bounds)
    area_long_face = np.prod(np.sort(text.extents)[-2:])
    # pl = pv.Plotter()
    # pl.add_mesh(text)
    # pl.add_mesh(box, color='red', style='wireframe')
    # pl.show()
    return text, box, area_long_face

def load_mesh(path=None, size=30, mesh=None):
    if mesh is None:
        mesh = pv.read(path)
    else:
        pass
    # mesh = pv.Cube().triangulate(inplace=True).subdivide(5, subfilter='butterfly')
    mesh = mesh.triangulate(inplace=True)
    mesh = utils.tri_to_pv(utils.pv_to_tri(mesh))
    mesh = mesh.translate(-np.array(mesh.center))
    scale = max(utils.pv_to_tri(mesh).extents)
    mesh = mesh.scale([1 / scale, 1 / scale, 1 / scale])
    if size > 0: 
        mesh = mesh.scale([size, size, size])
    mesh = mesh.compute_normals()
    mesh_tri = utils.pv_to_tri(mesh)
    # mesh.plot()
    return mesh, mesh_tri

def transform_mesh(path1, path2, paths, output_dir):
    mesh1 = trimesh.load(path1, file_type='obj', force="mesh")
    mesh2 = trimesh.load(path2, file_type='obj', force="mesh")
    mesh1 = utils.tri_to_pv(mesh1)
    mesh2 = utils.tri_to_pv(mesh2)
    # mesh1 = utils.tri_to_pv(utils.pv_to_tri(mesh1))
    # mesh2 = utils.tri_to_pv(utils.pv_to_tri(mesh2))
    distance = np.array(mesh2.center)-np.array(mesh1.center)
    size = max(utils.pv_to_tri(mesh2).extents)/max(utils.pv_to_tri(mesh1).extents)
    for wm_path in paths:
        wm_mesh = pv.read(wm_path)
        wm_mesh = wm_mesh.scale([size, size, size])
        wm_mesh = wm_mesh.translate(distance)
        pv.save_meshio(os.path.join(output_dir, os.path.basename(wm_path)), wm_mesh) 


def sample_mesh(mesh_tri, count=10, radius=None):
    # sampled_points = trimesh.sample.sample_surface(mesh_tri, count=30000)
    if radius is None:
        radius = np.sum(mesh_tri.area_faces)*0.0005
    sampled_points, sampled_faces_idx = trimesh.sample.sample_surface_even(mesh_tri, count=count, radius=radius)
    sampled_normals = mesh_tri.face_normals[sampled_faces_idx]
    sampled_points = torch.from_numpy(sampled_points)
    sampled_normals = torch.from_numpy(sampled_normals)
    sampled_faces_idx = torch.from_numpy(sampled_faces_idx)
    # mesh_tri.show()
    # pcd = trimesh.points.PointCloud(sampled_points)
    # pcd.show()
    return sampled_points, sampled_normals, sampled_faces_idx, radius

def create_wateramrks_vectorized(text_3d, box_3d, sampled_normals, sampled_points):
    from pytorch3d import transforms as py3d_transforms
    box3d_vertices = torch.tile(torch.cat([torch.from_numpy(box_3d.vertices), torch.ones(len(box_3d.vertices),1)], dim=1).unsqueeze(0), [len(sampled_points), 1, 1])
    text3d_vertices = torch.tile(torch.cat([torch.from_numpy(text_3d.vertices), torch.ones(len(text_3d.vertices),1)], dim=1).unsqueeze(0),[len(sampled_points),1,1])
    to_normals = sampled_normals
    from_normals = torch.mul(torch.ones_like(to_normals), torch.Tensor([0, 0, 1]))
    axis = torch.cross(from_normals, to_normals)
    axis = torch.div(axis, axis.norm(dim=1).unsqueeze(1))
    cosangles = torch.div(torch.bmm(from_normals.unsqueeze(1), to_normals.unsqueeze(2)).squeeze(), torch.multiply(to_normals.norm(dim=1), from_normals.norm(dim=1)))
    angles = torch.acos(cosangles)
    rot_mats = torch.tile(torch.diag(torch.DoubleTensor([1,1,1,1])).unsqueeze(0), [len(sampled_points),1,1])
    rot_mats[:, :3, :3] = py3d_transforms.axis_angle_to_matrix(torch.multiply(axis, angles.unsqueeze(1)))
    box3d_verts_trans = torch.matmul(box3d_vertices, rot_mats.transpose(1,2))[:,:,:3]
    text3d_verts_trans = torch.matmul(text3d_vertices, rot_mats.transpose(1, 2))[:,:,:3]
    box3d_verts_trans = box3d_verts_trans+(sampled_points-box3d_verts_trans.mean(dim=1)).unsqueeze(1)
    text3d_verts_trans = text3d_verts_trans+(sampled_points-text3d_verts_trans.mean(dim=1)).unsqueeze(1)
    # remove any potential nans (happens due to cross prod being small)
    nan_mask = torch.isnan(box3d_verts_trans.view(len(box3d_verts_trans), -1).sum(1))
    box3d_verts_trans = box3d_verts_trans[~nan_mask]
    text3d_verts_trans = text3d_verts_trans[~nan_mask]
    return box3d_verts_trans, text3d_verts_trans


def get_onehop_maps(mesh_tri):
    # IMPLEMENT HASH MAP IDEA
    facemap = defaultdict(list)
    vertexmap = defaultdict(set)
    for idx, face in enumerate(mesh_tri.faces):
        v1, v2, v3 = face
        facemap[v1].append(idx)
        facemap[v2].append(idx)
        facemap[v3].append(idx)
        vertexmap[v1].update(face)
        vertexmap[v2].update(face)
        vertexmap[v3].update(face)
    return vertexmap, facemap

def inside_verts_vectorized(boxes, points_vecs):
    # boxes = torch.from_numpy(vertices)
    # boxes = boxes.unsqueeze(0).repeat(2,1,1)
    b1_vecs = boxes[:,0,:]
    b2_vecs = boxes[:,4,:]
    b3_vecs = boxes[:,5,:]
    b4_vecs = boxes[:,1,:]
    t1_vecs = boxes[:,2,:]
    t2_vecs = boxes[:,6,:]
    t3_vecs = boxes[:,7,:]
    t4_vecs = boxes[:,3,:]
    dir1_vecs = t1_vecs - b1_vecs
    size1_vecs = dir1_vecs.norm(dim=1).unsqueeze(-1)
    dir1_vecs = dir1_vecs/size1_vecs
    dir2_vecs = b2_vecs - b1_vecs
    size2_vecs = dir2_vecs.norm(dim=1).unsqueeze(-1)
    dir2_vecs = dir2_vecs/size2_vecs
    dir3_vecs = b4_vecs - b1_vecs
    size3_vecs = dir3_vecs.norm(dim=1).unsqueeze(-1)
    dir3_vecs = dir3_vecs/size3_vecs
    cube3d_center_vecs = ((b1_vecs + t3_vecs) / 2.0).unsqueeze(1)
    # points_vecs = torch.from_numpy(points)
    point_center_vecs = points_vecs - cube3d_center_vecs
    dec1_vecs = torch.abs(torch.bmm(point_center_vecs, dir1_vecs.unsqueeze(2)))*2 <size1_vecs.unsqueeze(1)
    dec2_vecs = torch.abs(torch.bmm(point_center_vecs, dir2_vecs.unsqueeze(2)))*2 <size2_vecs.unsqueeze(1)
    dec3_vecs = torch.abs(torch.bmm(point_center_vecs, dir3_vecs.unsqueeze(2)))*2 <size3_vecs.unsqueeze(1)
    dec_vecs = torch.cat([dec1_vecs, dec2_vecs, dec3_vecs], 2)
    bool_decisions = torch.all(dec_vecs, dim=2)
    inside_verts = torch.nonzero(bool_decisions)
    outside_verts = torch.nonzero(~bool_decisions)
    return inside_verts, outside_verts, bool_decisions
    # torch.nonzero(dec_vecs)[torch.nonzero(dec_vecs)[:,0]==0][:,1]

def compute_inside_area_vectorized(boxes_vertices, boxes_normals, mesh_tri):
    USE_GPU='USE_GPU' in os.environ and os.environ['USE_GPU']=='1'
    if USE_GPU:
        print('USING GPU CUDA')
    start_time = time.time()
    vertexmap, facemap = get_onehop_maps(mesh_tri)
    # lengthmap = get_length_map(mesh_tri)
    print(f' Time taken computing maps {time.time() - start_time}')

    facemap_torch = {key: torch.LongTensor(val) for key, val in facemap.items()}
    # boxes_vertices = []
    # boxes_normals = []
    # for (box, _, normal) in candidate_watermarks:
    #     boxes_vertices.append(torch.from_numpy(box.vertices).unsqueeze(0))
    #     boxes_normals.append(torch.from_numpy(normal).unsqueeze(0))
    # boxes_vertices = torch.cat(boxes_vertices,0)
    # boxes_normals = torch.cat(boxes_normals, 0)
    mesh_vertices = torch.from_numpy(mesh_tri.vertices)
    mesh_faces = torch.from_numpy(mesh_tri.faces)
    mesh_normals = torch.from_numpy(mesh_tri.face_normals.copy())
    areamap = torch.Tensor(mesh_tri.area_faces.copy())
    if USE_GPU:
        mesh_vertices = mesh_vertices.cuda()
        boxes_vertices = boxes_vertices.cuda()
        mesh_faces = mesh_faces.cuda()
        areamap = areamap.cuda()

    inside_verts, outside_verts, bool_decisions = inside_verts_vectorized(boxes_vertices, mesh_vertices)
    inside_verts_flat = []
    all_relevant_faces = []
    for idx in range(len(boxes_vertices)):
        inside_verts_flat_idx = inside_verts[inside_verts[:,0]==idx][:,1]
        inside_verts_flat.append(inside_verts_flat_idx)
        relevant_faces_idx = [facemap_torch[int(vert)] for vert in inside_verts_flat[idx]]
        if len(relevant_faces_idx)>0:
            relevant_faces_idx = torch.unique(torch.cat(relevant_faces_idx))
        else:
            relevant_faces_idx = torch.LongTensor([])
        if USE_GPU:
            relevant_faces_idx = relevant_faces_idx.cuda()
        all_relevant_faces.append(relevant_faces_idx)

    # facemap_expanded = [facemap[idx] for idx in range(len(mesh_tri.vertices))]
    # facemap_bool = torch.BoolTensor(mesh_tri.vertices.__len__(), mesh_tri.faces.__len__())
    # all_relevant_faces = [torch.unique(torch.cat([facemap_torch[int(vert)] for vert in inside_verts_flat[idx]])) for idx in range(len(candidate_watermarks))]

    inside_faces = []
    intersect_faces = []
    inside_area = []
    intersect_area = []
    for idx in range(len(boxes_vertices)):
        relevant_faces_idx = all_relevant_faces[idx]
        relevant_faces = mesh_faces[relevant_faces_idx, :]
        all_decis = torch.sum(bool_decisions[idx][[relevant_faces]], 1)
        inside_faces.append(relevant_faces_idx[all_decis==3])
        # outside_faces.append(relevant_faces_idx[all_decis==0])
        intersect_faces.append(relevant_faces_idx[all_decis<3])
        face_areas = areamap[inside_faces[idx]]
        # inside_area.append(torch.sum(face_areas).unsqueeze(0))

        ''' projected area '''
        # https://math.stackexchange.com/questions/2578976/area-of-projected-parallelogram-onto-a-plane
        face_normals = mesh_normals[inside_faces[idx]]
        cos_angles = torch.bmm(face_normals.unsqueeze(1), boxes_normals[idx].unsqueeze(0).tile([len(face_normals),1]).unsqueeze(2)).squeeze()
        projected_area = torch.mul(face_areas, cos_angles)
        total_proj_area = torch.abs(torch.sum(projected_area).unsqueeze(0))
        inside_area.append(total_proj_area)

        ''' point projection area computation'''
        # inside_faces_xyz = mesh_vertices[mesh_faces[inside_faces[idx]]]
        # face_areas = torch.norm(torch.cross(inside_faces_xyz[:,2]-inside_faces_xyz[:,0], inside_faces_xyz[:,1]-inside_faces_xyz[:,0])/2, dim=1)
        # proj_mat = torch.from_numpy(trimesh.transformations.projection_matrix([10, 10, 10], box_normal)[:3, :3])
        # proj_points = torch.matmul(inside_faces_xyz, proj_mat)
        # proj_areas = torch.norm(torch.cross(proj_points[:, 2] - proj_points[:, 0], proj_points[:, 1] - proj_points[:, 0]) / 2, dim=1)

        ''' intersection area '''
        # box_vertices_idx = boxes_vertices[idx]
        # box_normal_idx = boxes_normals[idx]
        # intersect_faces_idx = intersect_faces[idx]
        #
        # intersect_area_idx = get_intersection_area(intersect_faces_idx, box_vertices_idx, box_normal_idx, mesh_vertices, mesh_faces)
        # total_area = intersect_area_idx + inside_area[idx]
        # inside_area[idx] = total_area
        # intersect_area.append(intersect_area_idx)

    inside_area = torch.cat(inside_area)
    if USE_GPU:
        inside_area = inside_area.cpu()
        inside_faces = [data.cpu() for data in inside_faces]
        intersect_faces = [data.cpu() for data in intersect_faces]
        inside_verts_flat = [data.cpu() for data in inside_verts_flat]
    return inside_area, inside_faces, intersect_faces, inside_verts_flat
