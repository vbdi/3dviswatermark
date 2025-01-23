import numpy as np
# import open3d as o3d
import time
import trimesh
import bpy

def remove_texture_coordinates(input_file, output_file):
    print("remove_texture_coordinates...")
    prefixes = ['v', 'vt', 'vn', 'f', 'mtlib', 'usemtl', 'g', 'o', '#']
    with open(input_file, "r") as input_obj:
        lines = input_obj.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts[0] == "vt" or parts[0] not in prefixes:
            continue  # Skip texture coordinate lines
        elif parts[0] == "f":
            parts = line.strip().split()
            pstr = 'f '
            vstr = ''
            for part in parts[1:]:
                    v = part.split("/")
                    vstr += '{}//{} '.format(int(v[0]), int(v[1]))
            pstr += vstr
            pstr += '\n'
            new_lines.append(pstr)
        else: 
            new_lines.append(line)

    with open(output_file, "w") as output_obj:
        output_obj.writelines(new_lines)

def split_model_into_octants(mesh, output_prefix):
    centroid = np.mean(mesh.vertices, axis=0)

    # Create eight partitions centered at the centroid
    split_plane_normals = [np.array([0,0,1.0]), np.array([0,0,-1.0]), np.array([0,1.0, 0]), np.array([0,-1.0, 0]), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])]

    split_meshes_2 = []
    for i in [np.array([0,0,1.0]), np.array([0,0,-1.0])]:
        split_meshes_2.append(mesh.slice_plane(centroid, i))

    split_meshes_4 = []
    for split_mesh in split_meshes_2:
        for j in [np.array([0,1.0, 0]), np.array([0,-1.0, 0])]:
            split_meshes_4.append(split_mesh.slice_plane(centroid, j))

    split_meshes_8 = []
    for split_mesh in split_meshes_4:
        for k in [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])]:
            split_meshes_8.append(split_mesh.slice_plane(centroid, k))

    # Save each octant as a separate .obj file
    octant_vertices = []
    for i, octant_mesh in enumerate(split_meshes_8):
        octant_vertices.append(len(octant_mesh.vertices))
        output_path = f"{output_prefix}_octant_{i}.obj"
        octant_mesh.export(output_path)
        print(f"Saved octant {i} to {output_path}")
    return octant_vertices

def subdivide_to_size_iter(vertices, faces, max_edge, return_index=False):
    # vertices ((n, 3) float) – Vertices in space
    # faces ((m, 3) int) – Indices of vertices which make up triangles
    # max_edge (float) – Maximum length of any edge in the result
    # return_index (bool) – If True, return index of original face for new faces
    
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    
    # Each face falls into one of 4 cases:
    #   1. Preserved: No edges need to be halved. Preserve this face.
    #   2. Split: One edge needs to be halved. Split this face into 2 new faces along the midpoint and the opposite vertex
    #   3. Triquad: Two edges need to be halved. Create a face from the shared vertex and the two midpoints, resulting in a tri and a quad
    #        Create two triangles from quad along the shortest diagonal
    #   4. Subdivide: Three edges need to be halved. Subdivide normally.
    
    # By construction, all faces must have 3 edges. -> shape (num_faces, 3)
    face_to_edge = np.arange(mesh.edges.shape[0]).reshape((-1, 3))
    face_to_edge_unique = mesh.edges_unique_inverse[face_to_edge] # which unique edge each of the ordered edges in each face corresponds to.
    
    edge_criterion_individual = (mesh.edges_unique_length > max_edge) # whether each edge is too long
    face_criterion = np.any(edge_criterion_individual[face_to_edge_unique], axis=-1) # whether the face has any edges that are too long
    edge_criterion = np.zeros(mesh.edges_unique.shape[0], dtype='bool') # whether each edge should be halved (based on self and adjacent faces)
    edge_criterion[np.unique(face_to_edge_unique[face_criterion])] = True
    face_edge_criterion = edge_criterion[face_to_edge_unique] # shape (num_faces, 3 faces_per_edge). Whether each edge of each face should be halved
    
    
    halved_midpoints = mesh.vertices[mesh.edges_unique[edge_criterion]].mean(1)
    
    # new_midpoint_inverse maps from unique edge index to new_vert idx of its midpoint
    new_midpoint_inverse = np.cumsum(edge_criterion) - 1
    new_midpoint_inverse[~edge_criterion] = len(edge_criterion) # this should an invalid index when referencing and is used as a sanity check
    new_midpoint_inverse += mesh.vertices.shape[0]
    
    
    new_verts = np.concatenate([mesh.vertices, halved_midpoints], axis=0)
    face_split_type = face_edge_criterion.sum(-1)
    
    preserve_inds = np.nonzero(face_split_type == 0)[0]
    split_inds = np.nonzero(face_split_type == 1)[0]
    triquad_inds = np.nonzero(face_split_type == 2)[0]
    subdivide_inds = np.nonzero(face_split_type == 3)[0]
    ########################
    preserve_faces = mesh.faces[preserve_inds]
    ########################
    # either 0, 1, or 2. Referrering to which edge is split in face_edge_criterion
    # split_edge[i] = j means that edge j (vertices split_faces[i, [j, (j+1)%3]]) is split and vertex (j + 2)%3 = 2 is opposite
    split_edge = np.argmax(face_edge_criterion[split_inds], axis=1)
    opposite_vertex = (split_edge + 2) % 3

    # new faces are [opposite vertex, (opp+1)%3, split midpoint]
    #           and [(opp+2)%3, opp, midpoint]
    face_to_edge_unique_split = face_to_edge_unique[split_inds]
    split_faces = np.array([
        [mesh.faces[split_inds, opposite_vertex], mesh.faces[split_inds, (opposite_vertex+1)%3], new_midpoint_inverse[face_to_edge_unique_split[np.arange(len(split_inds)), split_edge]]],
        [mesh.faces[split_inds, (opposite_vertex+2)%3], mesh.faces[split_inds, opposite_vertex], new_midpoint_inverse[face_to_edge_unique_split[np.arange(len(split_inds)), split_edge]]],
    ]).transpose(2, 0, 1).reshape((-1, 3))
    
    ########################
    # either 0, 1, or 2. Referrering to which edge is not split in face_edge_criterion for triquad faces
    # triquad_edge[i] = j means that edge j (vertices split_faces[i, [j, (j+1)%3]]) is split and vertex (j + 2)%3 = 2 is opposite
    triquad_edge = np.argmin(face_edge_criterion[triquad_inds], axis=1)
    triquad_opposite_vertex = (triquad_edge + 2) % 3
    face_to_edge_unique_triquad = face_to_edge_unique[triquad_inds]

    # triquad_tri is the triangle between the shared vertex and the two midpoints
    triquad_tri = np.array([
        [mesh.faces[triquad_inds, triquad_opposite_vertex], new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge-1)%3]], new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge+1)%3]]]
    ]).transpose(2, 0, 1).reshape((-1, 3))
    triquad_edge_diag1 = np.array([new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge-1)%3]], mesh.faces[triquad_inds, (triquad_opposite_vertex-1)%3]]).T
    triquad_edge_diag2 = np.array([new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge+1)%3]], mesh.faces[triquad_inds, (triquad_opposite_vertex+1)%3]]).T
    triquad_diag1_length = np.linalg.norm(np.diff(new_verts[triquad_edge_diag1], axis=1), axis=-1).flatten()
    triquad_diag2_length = np.linalg.norm(np.diff(new_verts[triquad_edge_diag2], axis=1), axis=-1).flatten()

    # faces created from drawing edge from midpoint following shared vertex with vertex preceding shared vertex 
    triquad_faces1 = np.array([
        [new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge-1)%3]], mesh.faces[triquad_inds, (triquad_opposite_vertex+1)%3], mesh.faces[triquad_inds, (triquad_opposite_vertex-1)%3]],
        [new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge-1)%3]], mesh.faces[triquad_inds, (triquad_opposite_vertex-1)%3], new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge+1)%3]]],
    ]).transpose(2, 0, 1)[triquad_diag1_length <= triquad_diag2_length].reshape((-1, 3))
    # faces created from drawing edge from midpoint preceding shared vertex with vertex following shared vertex
    triquad_faces2 = np.array([
        [new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge+1)%3]], mesh.faces[triquad_inds, (triquad_opposite_vertex+1)%3], mesh.faces[triquad_inds, (triquad_opposite_vertex-1)%3]],
        [new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge+1)%3]], new_midpoint_inverse[face_to_edge_unique_triquad[np.arange(len(triquad_inds)), (triquad_edge-1)%3]], mesh.faces[triquad_inds, (triquad_opposite_vertex+1)%3], ],
    ]).transpose(2, 0, 1)[triquad_diag1_length >  triquad_diag2_length].reshape((-1, 3))

    triquad_faces = np.concatenate([triquad_tri, triquad_faces1, triquad_faces2], axis=0)
    
    ########################
    # faces from subdivision
    subdivide_faces = np.array([
        [mesh.faces[subdivide_inds, 0], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 0]], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 2]]],
        [mesh.faces[subdivide_inds, 1], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 1]], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 0]]],
        [mesh.faces[subdivide_inds, 2], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 2]], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 1]]],
        [new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 0]], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 1]], new_midpoint_inverse[face_to_edge_unique[subdivide_inds, 2]]],
    ]).transpose(2, 0, 1).reshape((-1, 3))
    ########################
    new_faces = np.concatenate([preserve_faces, split_faces, triquad_faces, subdivide_faces], axis=0)
    
    if return_index:
        index = np.concatenate([
            preserve_inds,
            np.repeat(split_inds, 2),
            triquad_inds,
            np.repeat(triquad_inds[triquad_diag1_length >= triquad_diag2_length], 2),
            np.repeat(triquad_inds[triquad_diag1_length <  triquad_diag2_length], 2),
            np.repeat(subdivide_inds, 4),
        ])
        return new_verts, new_faces, index
    
    return new_verts, new_faces

def subdivide_to_size(vertices, faces, max_edge, max_iter=10, return_index=False):
    # vertices ((n, 3) float) – Vertices in space
    # faces ((m, 3) int) – Indices of vertices which make up triangles
    # max_edge (float) – Maximum length of any edge in the result
    # max_iter (int) – The maximum number of times to run subdivision. A non-positive value will use as many iterations as needed.
    # return_index (bool) – If True, return index of original face for new faces


    max_length = trimesh.Trimesh(vertices, faces, process=False).edges_unique_length.max()
    min_length = trimesh.Trimesh(vertices, faces, process=False).edges_unique_length.min()
    mean_length = trimesh.Trimesh(vertices, faces, process=False).edges_unique_length.mean()
    # max_edge = min(max_edge, mean_length)

    print(f'max_length = {max_length}')
    print(f'min_length = {min_length}')
    print(f'mean_length = {mean_length}')
    print(f'dividing to max_edge = {max_edge}')
    n_iter = int(np.ceil(np.log2(max_length/max_edge)))
    n_iter = min(max_iter, n_iter) if max_iter > 0 else n_iter
    index_maps = [np.arange(faces.shape[0])]
    for _ in range(n_iter):
        if not return_index:
            vertices, faces = subdivide_to_size_iter(vertices, faces, max_edge)
        else:
            vertices, faces, index_iter = subdivide_to_size_iter(vertices, faces, max_edge, return_index=True)
            index_maps.append(index_iter)
    if not return_index:
        return vertices, faces
    index = index_maps[-1]
    for index_prev in reversed(index_maps[0:-1]):
        index = index_prev[index]
    return vertices, faces, index

def subdivide_to_size_file(input_path, output_path, max_edge=0.5):
    # mesh: trimesh.Trimesh object
    # max_edge (float) – Maximum length of any edge in the result
    mesh = trimesh.load(input_path, force='mesh')
    new_mesh = trimesh.Trimesh(*subdivide_to_size(mesh.vertices, mesh.faces, max_edge, max_iter=-1), process=False)
    new_mesh.export(output_path)
    return new_mesh, len(new_mesh.vertices), len(new_mesh.triangles)

def subdivide_to_size_trimesh(mesh, output_path, max_edge=0.5):
    # mesh: trimesh.Trimesh object
    # max_edge (float) – Maximum length of any edge in the result
    new_mesh = trimesh.Trimesh(*subdivide_to_size(mesh.vertices, mesh.faces, max_edge, max_iter=-1), process=False)
    new_mesh.export(output_path)
    return new_mesh, len(new_mesh.vertices), len(new_mesh.triangles)

def get_model_size(input_obj_path):
    # Load the input .obj file
    mesh = trimesh.load(input_obj_path, force='mesh')
    # assumes mesh is axis aligned - otherwise it's an approximation
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_extents = np.max(mesh_vertices, 0) - np.min(mesh_vertices, 0)
    model_size = max(mesh_extents)
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.triangles)
    print(f"Trimesh model size = {model_size}. Vertices = {num_vertices}. Faces = {num_faces}")

    return model_size, num_vertices, num_faces

def count_loose_parts(input_obj_path, merge_doubles=True):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=input_obj_path)

    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.editmode_toggle()

    return len(bpy.context.scene.objects)
