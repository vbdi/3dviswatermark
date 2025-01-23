import bpy
import bmesh
import os

def import_obj(obj_path):
    """Import a single OBJ file along with its MTL file."""
    bpy.ops.wm.obj_import(filepath=obj_path)

def create_mesh(verts, faces, color, name):
    new_mesh = bpy.data.meshes.new(name=name)
    new_mesh.from_pydata(verts, [], faces)
    new_mesh.update()
    new_object = bpy.data.object
    bpy.data.collections["Collection"].objects.link(new_object)

def triangulate_object(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()

def boolean_union_objects(output_path, texture):
    """Join all objects in the scene into a single object."""
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")
    # Select all objects
    bpy.ops.object.select_all(action='SELECT')

    for obj in bpy.context.scene.objects:
        if "_watermarks_only_" not in obj.name and '_clr' not in obj.name:
            bpy.context.view_layer.objects.active = obj
    objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers["Boolean"].operation = 'UNION'
    bpy.context.object.modifiers["Boolean"].operand_type = 'COLLECTION'
    bpy.context.object.modifiers["Boolean"].collection = bpy.data.collections["Collection"]
    bpy.context.object.modifiers["Boolean"].solver = 'EXACT'
    bpy.context.object.modifiers["Boolean"].material_mode = 'TRANSFER'
    bpy.ops.object.modifier_apply(modifier="Boolean")

    for obj in bpy.context.scene.objects:
        if "_watermarks_only_" in obj.name or '_clr' in obj.name:
            obj.select_set(True)
        else:
            obj.select_set(False)
    bpy.ops.object.delete()

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            triangulate_object(obj)

    bpy.ops.wm.obj_export(
        filepath=output_path, export_materials=texture, apply_modifiers=True
    )

def join_objects(output_path):
    """Join all objects in the scene into a single object."""
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            ob.select_set(True)
            triangulate_object(ob)
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()          

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.wm.obj_export(
        filepath=output_path, export_materials=True
    )

def merge_obj_models(obj_files, output_path, texture=True, boolean=True):
    """Merge multiple OBJ models into one."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for obj_path in obj_files:
        import_obj(obj_path)

    if boolean: 
        print("Using Boolean Union")
        boolean_union_objects(output_path, texture)
    else: 
        print("Using Join")
        join_objects(output_path)