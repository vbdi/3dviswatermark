import bpy
import math
from mathutils import Vector

import numpy as np
import random
import trimesh
import bmesh

def remove_materials(obj):
    bpy.context.view_layer.objects.active = obj
    for x in obj.material_slots: #For all of the materials in the selected object:
        obj.active_material_index = 0 #select the top material
        bpy.ops.object.material_slot_remove() #delete it

def delete_faces_by_material(mesh):
    
    faces_to_delete = []
    mesh.select_set(True)
    me = mesh.data
    bpy.context.view_layer.objects.active = mesh

    material_name = mesh.material_slots[0].name
    
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    
    for poly in bm.faces:
        if me.materials[poly.material_index].name != material_name:
            # print(me.materials[poly.material_index].name)
            faces_to_delete.append(poly)
            
    bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
    bmesh.update_edit_mesh(me)
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh.select_set(False)  

def read_obj(obj_name):
    clean_scene()
    bpy.ops.wm.obj_import(filepath=obj_name)
    obj = None
    for o in bpy.data.objects:
        if o.type == 'MESH':
            obj = o
    return obj

def clean_scene():
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.ops.object.delete()        

def add_random_color(obj):
    r, g, b = [random.random() for _ in range(3)]
    random_color = [r, g, b, 1]
    bpy.context.view_layer.objects.active = obj
    # bpy.ops.outliner.item_activate(deselect_all=True)
    mat = bpy.data.materials.new(name=f"Material")
    obj.data.materials.append(mat)
    mat.use_nodes = True
        # Access the Principled BSDF node
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes["Principled BSDF"]
    if principled_bsdf is None:
        principled_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        nodes.active = principled_bsdf

    # Generate random RGB values
    principled_bsdf.inputs["Base Color"].default_value = tuple(random_color) 

def trimesh_to_blender(trimesh, mesh_name):
    mesh = bpy.data.meshes.new("mesh")  # add a new mesh
    obj = bpy.data.objects.new(mesh_name, mesh)  # add a new object using the mesh
    col = bpy.data.collections["Collection"]
    col.objects.link(obj)
    # bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(trimesh.vertices, [], trimesh.faces)
    return obj

def blender_to_trimesh(bpy_mesh):
    """Converts a Blender mesh object to a trimesh object, handling potential errors.

    Args:
        mesh_object (bpy.types.Object): The mesh object to convert.

    Returns:
        trimesh.Trimesh: The converted trimesh object.

    Raises:
        ValueError: If the mesh has no vertices or faces.
        TypeError: If the input is not a mesh object.
    """

    if not isinstance(bpy_mesh, bpy.types.Object) or bpy_mesh.type != 'MESH':
        raise TypeError("Input must be a valid mesh object.")

    mesh = bpy_mesh.data
    if not mesh.vertices or not mesh.polygons:
        raise ValueError("Mesh has no vertices or faces.")

    # Create empty arrays to store vertices and faces
    vertices = []
    faces = []

    # Loop through each vertex and face, ensuring consistent indexing
    for i, vertex in enumerate(mesh.vertices):
        vertices.append([vertex.co.x, vertex.co.y, vertex.co.z])

    for i, polygon in enumerate(mesh.polygons):
        # Convert polygons to triangles to ensure compatibility with trimesh
        if len(polygon.vertices) > 3:
            # Handle non-triangular faces by splitting into triangles
            for j in range(2, len(polygon.vertices)):
                faces.append([polygon.vertices[0], polygon.vertices[j - 1], polygon.vertices[j]])
        else:
            # Add triangular faces directly
            faces.append(polygon.vertices)

    # Create and return the trimesh object
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def get_distance_variance(obj, water):
    verts   = [v.co for v in water.data.vertices]
    verts = [water.matrix_world @ vert for vert in verts]
    verts = [obj.matrix_world.inverted() @ vert for vert in verts]
    # true/false, location, normal, index of face
    locs = [obj.closest_point_on_mesh(vert)[1] for vert in verts]
    dists = [math.dist(p, q) for q, p in zip(locs, verts)]
    return np.mean(np.var(np.array(dists)))


def save_all_objects(save_name):
    bpy.ops.wm.obj_export(filepath=save_name)


def get_closest_face_normal(obj, water):
    verts   = [v.co for v in water.data.vertices]
    points = np.asarray(verts)
    loc = Vector(np.mean(points, axis=0))
    loc = water.matrix_world @ loc
    loc = obj.matrix_world.inverted() @ loc
    # true/false, location, normal, index of face
    cpom = obj.closest_point_on_mesh(loc)
    normal = obj.matrix_world @ cpom[2]
    normal = water.matrix_world.inverted() @ normal
    return normal

def triangulate_object(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()

def adjust_watermark_core(water_mesh, original_mesh, shift_z=0.0):
    bpy.context.view_layer.objects.active = water_mesh
    bpy.ops.object.mode_set(mode='OBJECT')

    # finding the mesh intersect
    bool_inter = water_mesh.modifiers.new(type="BOOLEAN", name="bool_inter")
    bool_inter.object = original_mesh
    bool_inter.operation = 'INTERSECT'
    bool_inter.material_mode = 'TRANSFER'
    bpy.ops.object.modifier_apply(modifier="bool_inter")
    bpy.ops.object.modifier_remove(modifier="bool_inter")

    if len(water_mesh.data.vertices) <= 0:
        raise Exception("The watermark mesh and original mesh doesn't intersect")

    face_normal = get_closest_face_normal(original_mesh, water_mesh)
    mesh = water_mesh.data
    for vertex in mesh.vertices:
        vertex.co += face_normal * shift_z
    mesh.update()

    # final cleanup to make sure all the faces are trianglized
    for o in bpy.context.scene.objects:
        triangulate_object(o)

    return water_mesh    