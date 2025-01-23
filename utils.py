import numpy as np
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt
import helper


def tri_scale(mesh, factor=None, origin=None):
    mesh = mesh.copy()
    if factor is None:
        max_extent = max(mesh.extents)
        factor = 1/max_extent
    if origin is None:
        scale_origin = mesh.centroid
    else:
        scale_origin = origin
    S = trimesh.transformations.scale_matrix(factor, scale_origin)
    new_mesh = mesh.apply_transform(S)
    return new_mesh

def tri_translate(mesh, new_origin=None):
    if new_origin is None:
        new_origin = -mesh.centroid
    mesh = mesh.copy()
    T = trimesh.transformations.translation_matrix(new_origin)
    new_mesh = mesh.apply_transform(T)
    return new_mesh

def tri_summary(mesh):
    print(f'Centroid {mesh.centroid}')
    print(f'Extents {mesh.extents}')
    print(f'Scale {mesh.scale}')

''' UTILITY FUNCTIONS '''
def pv_to_tri(mesh):
    ufaces = _unflat_faces(mesh.faces)
    tri_mesh = trimesh.Trimesh(vertices=mesh.points, faces=ufaces)
    return tri_mesh

def tri_to_pv(mesh):
    ufaces = _flat_faces(mesh.faces)
    pv_mesh = pv.PolyData(mesh.vertices, ufaces)
    return pv_mesh

def tri_center(mesh):
    T = trimesh.transformations.translation_matrix(-mesh.centroid)
    new_mesh = mesh.apply_transform(T)
    return new_mesh
#
# def tri_scale(mesh):
#     max_extent = max(mesh.extents)
#     scale_origin = mesh.centroid
#     S = trimesh.transformations.scale_matrix(1/max_extent, scale_origin)
#     new_mesh = mesh.apply_transform(S)
#     return new_mesh


def generate_tri_boxes(mesh_tri, box3d_faces, box3d_verts, filter_idx=None, show=False):
    if filter_idx is None:
        filter_idx = range(len(box3d_verts))
    box3d_verts_ = box3d_verts[filter_idx]
    initial_boxes = [trimesh.Trimesh(vertices=verts.numpy(), faces=box3d_faces) for verts in box3d_verts]
    filter_boxes = [trimesh.Trimesh(vertices=verts.numpy(), faces=box3d_faces) for verts in box3d_verts_]
    if show==True:
        helper.plot_multi([[mesh_tri] + initial_boxes, [mesh_tri] + filter_boxes])
    return filter_boxes, initial_boxes


''' PLOTTING FUNCTIONS '''
def plot(meshes, use_colors=True, wireframe=True):
    if not isinstance(meshes, list):
        meshes = [meshes]
    colors = ['lightblue', 'red', 'blue', 'orange', 'magenta', 'white', 'cyan', 'green']
    plotter = pv.Plotter()
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
    plotter.show()
    return plotter

def mpl_show(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces)
    plt.show()


''' HELPER FUNCTIONS '''
def _unflat_faces(faces):
    idx = 0
    unflat_faces = []
    while idx < len(faces):
        num = faces[idx]
        face = np.expand_dims(faces[idx + 1:idx + 1 + num], 0)
        unflat_faces.append(face)
        idx = idx + num + 1
    unflat_faces = np.concatenate(unflat_faces, 0)
    return unflat_faces

def _flat_faces(faces):
    lfaces = [[len(face)] + face.tolist() for face in faces]
    ufaces = [val for sublist in lfaces for val in sublist]
    return ufaces


