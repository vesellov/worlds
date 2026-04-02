import os
import sys
import json
import time
import math
import random
import numpy as np

from kivy.graphics import (
    RenderContext, Callback, BindTexture,
    ChangeState, PushState, PopState,
    PushMatrix, PopMatrix,
    Color, Translate, Rotate, Mesh,
)
from kivy.graphics.transformation import Matrix  # @UnresolvedImport
from kivy.graphics.instructions import InstructionGroup  # @UnresolvedImport
from kivy.graphics.context_instructions import Transform  # @UnresolvedImport

import res
import mth
import dat


_Debug = True


_NextUnitID = 0
_NextObjectID = 0
_NextMeshID = 0


class Unit(object):

    def __init__(self, name, object_name):
        self.name = name
        self.object_name = object_name
        self.template = None
        self.w = None
        self.h = None
        self.shift_w = None
        self.shift_h = None
        self.area_w = None
        self.area_h = None
        self.static = None
        self.animations_list = []
        self.animation_playing = None
        self.animation_frame = 0
        self.meshes_transforms = {}
        self.root_mesh_center = None
        self.rotate_axis_x = None
        self.rotate_axis_z = None
        self.rotate_vertical = None
        self.translate_shift = None
        self.direction = 0.0
        self.acceleration = 0.0
        self.speed = 0.0
        self.max_speed = 0.0

    def run(self, scene):
        # return
        self.speed += self.acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        self.direction += 1.0
        if self.direction > 360.0:
            self.direction -= 360.0
        self.rotate_vertical.angle = self.direction + 90
        self.shift_w += self.speed * math.cos(math.radians(self.direction))
        self.shift_h += self.speed * math.sin(math.radians(self.direction))
        w_new = self.w
        h_new = self.h
        if self.shift_w > 1.0:
            w_new += int(self.shift_w)
            self.shift_w = float(self.shift_w) - float(int(self.shift_w))
        elif self.shift_w < 0.0:
            w_new += int(self.shift_w) - 1
            self.shift_w = float(self.shift_w) - float(int(self.shift_w)) + 1.0
        if self.shift_h > 1.0:
            h_new += int(self.shift_h)
            self.shift_h = float(self.shift_h) - float(int(self.shift_h))
        elif self.shift_h < 0.0:
            h_new += int(self.shift_h) - 1
            self.shift_h = float(self.shift_h) - float(int(self.shift_h)) + 1.0
        w_diff = w_new - self.w
        h_diff = h_new - self.h
        self.w = w_new
        self.h = h_new
        e_correction = 0
        if self.root_mesh_center:
            e_correction = self.root_mesh_center[0][2]
        shift_vector = scene.coords_map2xyz(self.w, self.h, self.shift_w, self.shift_h, elevation_correction=e_correction)
        self.translate_shift.xyz = shift_vector
        if w_diff != 0 or h_diff != 0:
            # if _Debug:
            #     print(f'  unit {self.name} at map {self.w},{self.h} shift:{self.shift_w},{self.shift_h} shift_vector:{shift_vector} direction:{self.direction} speed:{self.speed}')
            self.area_w += w_diff
            self.area_h += h_diff
            segment_angle_x, segment_angle_z = scene.coords_area2angles(self.area_w, self.area_h)
            self.rotate_axis_x.angle = segment_angle_x
            self.rotate_axis_z.angle = segment_angle_z


class Scene(object):

    SEGMENT_SIZE = 6.0
    PLANET_EQUATOR_SEGMENTS = 360
    PLANET_EQUATOR_LENGTH = SEGMENT_SIZE * PLANET_EQUATOR_SEGMENTS
    PLANET_RADIUS = PLANET_EQUATOR_LENGTH / (2.0 * math.pi)    
    SEGMENT_ANGLE = 360.0 / PLANET_EQUATOR_SEGMENTS
    SEGMENT_ANGLE_HALF = SEGMENT_ANGLE / 2.0
    SEGMENT_ANGLE_HALF_RADIANS = math.radians(SEGMENT_ANGLE_HALF)
    SEGMENT_HALF_SIN = math.sin(SEGMENT_ANGLE_HALF_RADIANS)
    SEGMENT_HALF_COS = math.cos(SEGMENT_ANGLE_HALF_RADIANS)  
    SEGMENT_ANGLE_RADIANS = math.radians(SEGMENT_ANGLE/1.414213562373095)
    SEGMENT_SIN = math.sin(SEGMENT_ANGLE_RADIANS)
    SEGMENT_COS = math.cos(SEGMENT_ANGLE_RADIANS)  
    PI_4_SIN = math.sin(math.pi / 4.0)
    PI_4_COS = math.cos(math.pi / 4.0)
    ELEVATION_FACTOR = PLANET_RADIUS / 6.0
    VISIBLE_AREA_SIZE_SEGMENTS = 48
    VISIBLE_AREA_SIZE_SEGMENTS_HALF = int(VISIBLE_AREA_SIZE_SEGMENTS / 2.0)
    LAND_MOVE_SPEED = 0.5

    def __init__(self, land):
        self.land = land
        self.renderer = None
        self.models = {}
        self.meshes = {}
        self.meshes_index = {}
        self.static_objects = {}
        self.animated_objects = {}
        self.units = {}
        self.animating_units = set()
        self.visible_animating_units = set()
        self.container = None
        self.container_land = None
        self.container_land_tiles = None
        self.container_static_objects = None
        self.container_animated_objects = None
        self.global_land_translate_before = None
        self.global_land_translate_after = None
        self.global_land_rotate_x = None
        self.global_land_rotate_y = None
        self.global_land_rotate_z = None
        self.map_width = None
        self.map_height = None
        self.area_center_w = None
        self.area_center_h = None
        self.segment_shift_w = None
        self.segment_shift_h = None
        # self.land_area_left = 0
        # self.land_area_top = 0
        self.land_area_mask = {}
        self.land_tiles_visible = {}

    def coords_area2angles(self, w, h, as_area=True):
        # if as_area:
        #     angle_z = mth.w2lat_degrees(float(w) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        #     angle_x = mth.h2lon_degrees(float(h) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        # else:
        angle_z = mth.w2lat_degrees(float(w), self.PLANET_EQUATOR_SEGMENTS)
        angle_x = mth.h2lon_degrees(float(h), self.PLANET_EQUATOR_SEGMENTS)
        return angle_x, angle_z

    def coords_map2xyz(self, map_w, map_h, shift_w, shift_h, elevation_correction=None):
        e, _, _ = self.calculate_elevation(map_w, map_h, shift_w, shift_h)
        c = e * self.SEGMENT_SIN
        e_correction = 0
        if elevation_correction is not None:
            e_correction = elevation_correction
        return (
            c * self.PI_4_SIN * ((0.5 - shift_w) * 2.0),
            e - e_correction,
            c * self.PI_4_COS * ((shift_h - 0.5) * 2.0),
        )

    def create_containers(self):
        self.container = InstructionGroup()
        self.container_land = InstructionGroup()

    def mesh_key(self, template, part_name, coefs):
        c = mth.quantize_coefs(coefs)
        return f'{template}_{part_name}_{c[0]}_{c[1]}_{c[2]}'

    def get_segment_elevation(self, map_w, map_h):
        e00 = self.PLANET_RADIUS + self.land.get_elevation(map_w, map_h) * self.ELEVATION_FACTOR
        e01 = self.PLANET_RADIUS + self.land.get_elevation(map_w, map_h + 1) * self.ELEVATION_FACTOR
        e10 = self.PLANET_RADIUS + self.land.get_elevation(map_w + 1, map_h) * self.ELEVATION_FACTOR
        e11 = self.PLANET_RADIUS + self.land.get_elevation(map_w + 1, map_h + 1) * self.ELEVATION_FACTOR
        return e00, e01, e10, e11

    def calculate_elevation(self, w_i, h_i, shift_w, shift_h):
        e00, e01, e10, e11 = self.get_segment_elevation(w_i, h_i)
        e_min = min(e00, e01, e10, e11)
        e_max = max(e00, e01, e10, e11)
        a = self.SEGMENT_ANGLE
        # p00 = (0, 0, e01)
        # p01 = (0, a, e11)
        # p10 = (a, 0, e00)
        # p11 = (a, a, e10)
        p00 = (0, 0, e00)
        p01 = (0, a, e01)
        p10 = (a, 0, e10)
        p11 = (a, a, e11)
        w_f = shift_w * a
        h_f = shift_h * a
        if mth.point_line_left_or_right(w_f, h_f, p00[0], p00[1], p11[0], p11[1]) == -1:
            e = mth.get_z_in_triangle(w_f, h_f, p00, p11, p01)
        else:
            e = mth.get_z_in_triangle(w_f, h_f, p11, p00, p10)
        return e, e_min, e_max

    def add_model_template(self, template, model):
        self.models[template] = model

    def create_mesh_from_fig_data(self, fig_data, prefix='', texture_filename=None, coefs=[0, 0, 0]):
        """
        fig_data fields list:
            0:"blocks",
            1:"vertex_count",
            2:"normal_count",
            3:"texcoord_count",
            4:"index_count",
            5:"vertex_component_count",
            6:"morph_component_count",
            7:"group",
            8:"texture_number",
            9:"center",
            10:"min",
            11:"max",
            12:"radius",
            13:"vertices",
            14:"normals",
            15:"texcoords",
            16:"indexes",
            17:"vertex_components",
            18:"morph_components"
        """
        global _NextMeshID
        _NextMeshID += 1
        name = prefix + '_' + str(_NextMeshID)
        mesh = dat.MeshData(
            name=name,
            material={'map_Kd': texture_filename} if texture_filename else None,
        )
        mesh.coefs = coefs
        vert_buf = []
        norm_buf = []
        tex_buf = []
        for i in range(fig_data[1]):
            for j in range(4):
                vert_buf.append(mth.ei2xyz_list([
                    mth.trilinear([fig_data[13][i][0][k][j] for k in range(8)], coefs),
                    mth.trilinear([fig_data[13][i][1][k][j] for k in range(8)], coefs),
                    mth.trilinear([fig_data[13][i][2][k][j] for k in range(8)], coefs),
                ]))
        for i in range(fig_data[2]):
            for j in range(4):
                norm_buf.append(mth.ei2xyz_list([
                    fig_data[14][i][0][j],
                    fig_data[14][i][1][j],
                    fig_data[14][i][2][j],
                ]))
        for i in range(fig_data[3]):
            tex_buf.append(fig_data[15][i])
        idx = 0
        d = fig_data[17]
        for i in fig_data[16]:
            for f in range(3):
                j = i[f]
                mesh.vertices.extend([
                    vert_buf[d[j][0] * 4 + d[j][1]][0],
                    vert_buf[d[j][0] * 4 + d[j][1]][1],
                    vert_buf[d[j][0] * 4 + d[j][1]][2],
                    norm_buf[d[j][2] * 4 + d[j][3]][0],
                    norm_buf[d[j][2] * 4 + d[j][3]][1],
                    norm_buf[d[j][2] * 4 + d[j][3]][2],
                    tex_buf[d[j][4]][0],
                    tex_buf[d[j][4]][1],
                ])
            mesh.indices.extend([idx, idx + 1, idx + 2])
            idx += 3
        mesh.center = fig_data[9]
        mesh.min = fig_data[10]
        mesh.max = fig_data[11]
        mesh.radius = fig_data[12]
        self.meshes[name] = mesh
        # if _Debug:
        #     print(f'  prepared mesh {name} with {idx} faces and texture {texture_filename}')
        return mesh
    
    def create_animated_object_from_model_data(self, template, coefs=[0, 0, 0], selected_parts=[], excluded_parts=[], selected_animations=None, textures={'*': 'default.png'}):
        global _NextObjectID
        _NextObjectID += 1
        if template not in self.models:
            m = dat.ModelData()
            m.unpack_figure_data('figures.res', 'models', template=template)
            if not os.path.isfile('textures/model/' + template + '.png'):
                m.unpack_texture('textures.res', 'textures/model', template)
            self.add_model_template(template, m)
        m = self.models[template]
        ao = dat.ObjectData(name=template+'#'+str(_NextObjectID), static=False)
        coefs = mth.quantize_coefs(coefs)
        ao.template = template
        ao.textures = textures
        ao.parts_tree_ordered = m.links[template]['ordered']
        ao.parts_tree = m.links[template]['tree']
        ao.parts_parents = m.links[template]['parents']
        ordered_parts_list = res.flat_tree(ao.parts_tree_ordered)
        ao.animations_loaded = selected_animations
        if not ao.animations_loaded:
            ao.animations_loaded = list(m.animations.keys())
        if not selected_parts:
            selected_parts = ordered_parts_list
        for exclude in excluded_parts:
            if exclude in selected_parts:
                selected_parts.remove(exclude)
        ao.root_part_name = selected_parts[0]
        # if _Debug:
        #     print(f'about to prepare unit ({ao.name}) with {len(selected_parts)} parts and {len(ao.animations_loaded)} animations from model {{{template}}}')
        t1 = time.time()
        for part_name in selected_parts:
            ao.parts.append(part_name)
            part_info = m.bones[part_name]
            ao.bones[part_name] = mth.ei2xyz_list([
                mth.trilinear([part_info[i][0] for i in range(8)], coefs),
                mth.trilinear([part_info[i][1] for i in range(8)], coefs),
                mth.trilinear([part_info[i][2] for i in range(8)], coefs),
            ])
            mesh_key = self.mesh_key(ao.template, part_name, coefs)
            if mesh_key in self.meshes_index:
                mesh = self.meshes[self.meshes_index[mesh_key]]
                if _Debug:
                    print(f'    reused mesh {mesh.name} for part {ao.name}:{part_name} with texture {mesh.material["map_Kd"]}')
            else:
                mesh = self.create_mesh_from_fig_data(
                    fig_data=m.figures[part_name],
                    prefix=ao.template + '_' + part_name,
                    texture_filename=ao.textures[part_name] if part_name in ao.textures else ao.textures['*'],
                    coefs=coefs,
                )
                mesh.object_name = ao.name
                mesh.object_part_name = part_name
                self.meshes_index[mesh_key] = mesh.name
            ao.meshes[part_name] = mesh.name
            for anim_name in ao.animations_loaded:
                if part_name not in m.animations[anim_name]:
                    continue
                animation_info = m.animations[anim_name][part_name]
                if anim_name not in ao.animations:
                    ao.animations[anim_name] = dat.ObjectAnimationData(template, anim_name)
                a = dat.ObjectPartAnimationData()
                a.rotation_frames_input = [mth.ei2quad_list(quad) for quad in animation_info[1]]
                a.translation_frames_input = [mth.ei2xyz_list(coord) for coord in animation_info[3]]
                morphing_frames = []
                if animation_info[4] != 0 and animation_info[5] != 0:
                    for value in animation_info[6]:
                        morphing_frames.append([])
                        for i in range(animation_info[5]):
                            morphing_frames[0].append(mth.ei2xyz_list(value[i]))
                    a.morphing_frames_input = morphing_frames
                a.frames = len(a.rotation_frames_input)
                ao.animations[anim_name].parts[part_name] = a
            if part_name == ao.root_part_name:
                ao.root_mesh_name = mesh.name
                ao.root_mesh_center = mesh.center
        ao.calculate_animations()
        self.animated_objects[ao.name] = ao
        t2 = time.time()
        if _Debug:
            print(f'  animated object {ao.name} with {len(selected_parts)} parts and {len(ao.animations_loaded)} animations created in {t2 - t1} sec from template {template}')
        return ao

    def create_static_object_from_model_data(self, template, coefs=[0, 0, 0], selected_parts=[], excluded_parts=[], textures={'*': 'default.png'}):
        global _NextObjectID
        _NextObjectID += 1
        if template not in self.models:
            m = dat.ModelData()
            m.unpack_figure_data('figures.res', 'models', template=template)
            if not os.path.isfile('textures/model/' + template + '.png'):
                m.unpack_texture('textures.res', 'textures/model', template)
            self.add_model_template(template, m)
        m = self.models[template]
        so = dat.ObjectData(name=template+'#'+str(_NextObjectID), static=True)
        coefs = mth.quantize_coefs(coefs)
        so.template = template
        so.textures = textures
        so.parts_tree_ordered = m.links[template]['ordered']
        so.parts_tree = m.links[template]['tree']
        so.parts_parents = m.links[template]['parents']
        ordered_parts_list = res.flat_tree(so.parts_tree_ordered)
        if not selected_parts:
            selected_parts = ordered_parts_list
        for exclude in excluded_parts:
            if exclude in selected_parts:
                selected_parts.remove(exclude)
        so.root_part_name = selected_parts[0]
        # if _Debug:
        #     print(f'about to prepare static object ({so.name}) with {len(selected_parts)} parts from model {{{template}}}')
        t1 = time.time()
        for part_name in selected_parts:
            so.parts.append(part_name)
            part_info = m.bones[part_name]
            so.bones[part_name] = mth.ei2xyz_list([
                mth.trilinear([part_info[i][0] for i in range(8)], coefs),
                mth.trilinear([part_info[i][1] for i in range(8)], coefs),
                mth.trilinear([part_info[i][2] for i in range(8)], coefs),
            ])
            mesh_key = self.mesh_key(so.template, part_name, coefs)
            if mesh_key in self.meshes_index:
                mesh = self.meshes[self.meshes_index[mesh_key]]
                if _Debug:
                    print(f'      reused mesh {mesh.name} for part {so.name}:{part_name} with texture {mesh.material["map_Kd"]}')
            else:
                mesh = self.create_mesh_from_fig_data(
                    fig_data=m.figures[part_name],
                    prefix=so.template + '_' + part_name,
                    texture_filename=so.textures[part_name] if part_name in so.textures else so.textures['*'],
                    coefs=coefs,
                )
                mesh.object_name = so.name
                mesh.object_part_name = part_name
                self.meshes_index[mesh_key] = mesh.name
            so.meshes[part_name] = mesh.name
            if part_name == so.root_part_name:
                so.root_mesh_name = mesh.name
                so.root_mesh_center = mesh.center
        self.static_objects[so.name] = so
        t2 = time.time()
        # if _Debug:
        #     print(f'    static object {so.name} with {len(selected_parts)} parts created in {t2 - t1} sec from template {template}')
        return so

    def construct_unit_from_object_data(self, container, object_name, angle_coords=None, shift_vector=None, direction=0, static=True):
        global _NextUnitID
        _source_dict = self.static_objects if static else self.animated_objects
        if object_name not in _source_dict:
            raise Exception(f'Model data object {object_name} was not prepared')
        if not shift_vector:
            shift_vector = [0.0, 0.0, 0.0]
        _NextUnitID += 1
        unit = Unit(name=object_name+'#'+str(_NextUnitID), object_name=object_name)
        unit.static = static
        if static:
            source_object = self.static_objects[object_name]
        else:
            source_object = self.animated_objects[object_name]
            unit.animations_list = source_object.animations_loaded.copy()
        unit.root_mesh_center = source_object.root_mesh_center

        def _visitor(part_name, parent_part_name):
            mesh_name = source_object.meshes[part_name]
            mesh = self.meshes[mesh_name]
            if part_name in unit.meshes_transforms:
                raise Exception(f'Mesh transform for part [{part_name}] of unit ({unit.name}) already exists')
            mesh_transform = dat.MeshTransformData()
            mesh_transform.part_translate = Transform(group=unit.name)
            mesh_transform.part_rotate = Transform(group=unit.name)
            unit.meshes_transforms[part_name] = mesh_transform
            container.add(PushMatrix(group=unit.name))
            container.add(mesh_transform.part_translate)
            container.add(PushMatrix(group=unit.name))
            container.add(mesh_transform.part_rotate)
            # TODO: check if we can pass texture data directly to Mesh instruction as a parameter
            container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=unit.name))
            container.add(Mesh(
                vertices=mesh.vertices,
                indices=mesh.indices,
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=unit.name,
                # texture=<already loaded Texture>,
            ))
            container.add(PopMatrix(group=unit.name))  # part_rotate
            container.add(PopMatrix(group=unit.name))  # part_translate

        unit.rotate_axis_x = Rotate(angle_coords[0], 1, 0, 0, group=unit.name)
        unit.rotate_axis_z = Rotate(angle_coords[1], 0, 0, 1, group=unit.name)
        unit.rotate_vertical = Rotate(direction + 90, 0, 1, 0, group=unit.name)
        unit.translate_shift = Translate(shift_vector[0], shift_vector[1], shift_vector[2], group=unit.name)
        container.add(PushMatrix(group=unit.name))  # unit
        container.add(unit.rotate_axis_x)
        container.add(unit.rotate_axis_z)
        container.add(PushMatrix(group=unit.name))  # unit shift
        container.add(unit.translate_shift)
        container.add(PushMatrix(group=unit.name))  # unit rotate
        container.add(unit.rotate_vertical)

        if False and _Debug:
            sz = 1.0
            # container.add(PushMatrix(group=unit.name))  # border
            # container.add(PushState(group=unit.name))
            # container.add(Translate(shift_vector[0], shift_vector[1], shift_vector[2], group=unit.name))
            container.add(ChangeState(line_color=(1., 0.5, 0.5, 1.), group=unit.name))
            container.add(Mesh(
                vertices=[
                    -1 * sz, -1 * sz, -1 * sz,
                    -1 * sz, -1 * sz, 1 * sz,
                    -1 * sz, 1 * sz, 1 * sz,
                    -1 * sz, 1 * sz, -1 * sz,
                    1 * sz, -1 * sz, -1 * sz,
                    1 * sz, -1 * sz, 1 * sz,
                    1 * sz, 1 * sz, 1 * sz,
                    1 * sz, 1 * sz, -1 * sz,
                ],
                indices=[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7],
                fmt=[(b'v_pos', 3, 'float'), ],
                mode='lines',
                group=unit.name,
            ))
            container.add(ChangeState(line_color=(1., 0., 0., 1.), group=unit.name))
            container.add(Mesh(
                vertices=[1 * sz, 0, 0, 0, 0, 0],
                indices=[0, 1],
                fmt=[(b'v_pos', 3, 'float'), ],
                mode='lines',
                group=unit.name,
            ))
            container.add(ChangeState(line_color=(0., 1., 0., 1.), group=unit.name))
            container.add(Mesh(
                vertices=[0, 1 * sz, 0, 0, 0, 0],
                indices=[0, 1],
                fmt=[(b'v_pos', 3, 'float'), ],
                mode='lines',
                group=unit.name,
            ))
            container.add(ChangeState(line_color=(0., 0., 1., 1.), group=unit.name))
            container.add(Mesh(
                vertices=[0, 0, 1 * sz, 0, 0, 0],
                indices=[0, 1],
                fmt=[(b'v_pos', 3, 'float'), ],
                mode='lines',
                group=unit.name,
            ))
            container.add(ChangeState(line_color=(1., 1., 1., 1.), group=unit.name))
            # container.add(PopState(group=unit.name))
            # container.add(PopMatrix(group=unit.name))  # border

        source_object.walk_parts_ordered(_visitor)

        container.add(PopMatrix(group=unit.name))  # unit rotate
        container.add(PopMatrix(group=unit.name))  # unit shift
        container.add(PopMatrix(group=unit.name))  # unit
        self.units[unit.name] = unit
        if static is False:
            if _Debug:
                print(f'created animated unit at {angle_coords} with shift {shift_vector} from object {object_name} and placed on scene')
        # if _Debug:
        #     print(f'  constructed unit ({unit.name}) from object {object_name} and placed on scene')
        return unit

    def remove_unit_from_stage(self, container, unit_name):
        if unit_name not in self.units:
            raise Exception(f'Unit {unit_name} is not on the stage at the moment')
        unit = self.units[unit_name]
        container.remove_group(unit.name)
        for part_name in unit.meshes_transforms.keys():
            unit.meshes_transforms[part_name].part_rotate = None
            unit.meshes_transforms[part_name].part_translate = None
        unit.meshes_transforms.clear()
        unit.rotate_x_axis = None
        unit.rotate_z_axis = None
        self.units.pop(unit_name)
        # if _Debug:
        #     print(f'  removed unit {unit.name} from scene')

    def prepare_land(self, map_center_w, map_center_h, map_width, map_height):
        for _w in range(-self.VISIBLE_AREA_SIZE_SEGMENTS_HALF, self.VISIBLE_AREA_SIZE_SEGMENTS_HALF):
            for _h in range(-self.VISIBLE_AREA_SIZE_SEGMENTS_HALF, self.VISIBLE_AREA_SIZE_SEGMENTS_HALF):
                dist = int(math.sqrt(_w * _w + _h * _h))
                if dist < self.VISIBLE_AREA_SIZE_SEGMENTS_HALF:
                    self.land_area_mask[(_w, _h)] = dist
        self.global_land_rotate_x = Rotate(0, 1, 0, 0, group='land')
        self.global_land_rotate_y = Rotate(0, 0, 1, 0, group='land')
        self.global_land_rotate_z = Rotate(0, 0, 0, 1, group='land')
        self.map_width = map_width
        self.map_height = map_height
        self.area_center_w = int(map_center_w)
        self.area_center_h = int(map_center_h)
        self.segment_shift_w = 0.5
        self.segment_shift_h = 0.5
        w = int(self.area_center_w)
        h = int(self.area_center_h)
        # self.land_area_left = w - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
        # self.land_area_top  = h - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
        camera_shift_angle_x, camera_shift_angle_z = self.coords_area2angles(0.5-self.segment_shift_w, 0.5-self.segment_shift_h, as_area=False)
        elevation_at_center = self.land.get_elevation(w, h)
        planet_shift_y = self.PLANET_RADIUS + elevation_at_center * self.ELEVATION_FACTOR
        self.global_land_translate_before = Translate(0, -planet_shift_y, 0, group='land')
        self.global_land_translate_after = Translate(0, planet_shift_y, 0, group='land')
        self.global_land_rotate_x.angle = camera_shift_angle_x
        self.global_land_rotate_z.angle = camera_shift_angle_z
        self.container_land.add(PushMatrix(group='land'))
        self.container_land.add(self.global_land_translate_before)
        self.container_land.add(self.global_land_rotate_x)
        self.container_land.add(self.global_land_rotate_y)
        self.container_land.add(self.global_land_rotate_z)
        self.container_animated_objects = InstructionGroup()
        self.container_static_objects = InstructionGroup()
        self.container_land_tiles = InstructionGroup()
        self.container_land.add(self.container_animated_objects)
        self.container_land.add(self.container_land_tiles)
        self.container_land.add(self.container_static_objects)
        self.container_land.add(self.global_land_translate_after)
        self.container_land.add(PopMatrix(group='land'))
        added = 0
        for _w, _h in self.land_area_mask.keys():
            w_t = w + _w
            h_t = h + _h
            if (w_t, h_t) not in self.land_tiles_visible:
                self.add_land_segment(w_t, h_t, _w, _h)
                added += 1
        # for _w in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
        #     for _h in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
        #         w_t = self.land_area_left + _w
        #         h_t = self.land_area_top + _h
        #         if (w_t, h_t) not in self.land_tiles_visible:
        #             self.add_land_segment(w_t, h_t, _w, _h)
        #             added += 1
        if _Debug:
            print(f'prepare land area at {w} {h} with {added} segments planet angle x:0 z:0')

    def update_land(self, new_position=None):
        if new_position:
            w_i, h_i, sh_w, sh_h = new_position
            wd = w_i - int(self.area_center_w)
            hd = h_i - int(self.area_center_h)
            self.segment_shift_w = sh_w
            self.segment_shift_h = sh_h
        else:
            w0 = int(self.area_center_w)
            h0 = int(self.area_center_h)
            w_i = w0
            h_i = h0
            if self.segment_shift_w > 1.0:
                w_i += int(self.segment_shift_w)
                self.segment_shift_w = float(self.segment_shift_w) - float(int(self.segment_shift_w))
            elif self.segment_shift_w < 0.0:
                w_i += int(self.segment_shift_w) - 1
                self.segment_shift_w = float(self.segment_shift_w) - float(int(self.segment_shift_w)) + 1.0
            if self.segment_shift_h > 1.0:
                h_i += int(self.segment_shift_h)
                self.segment_shift_h = float(self.segment_shift_h) - float(int(self.segment_shift_h))
            elif self.segment_shift_h < 0.0:
                h_i += int(self.segment_shift_h) - 1
                self.segment_shift_h = float(self.segment_shift_h) - float(int(self.segment_shift_h)) + 1.0
            wd = w_i - w0
            hd = h_i - h0
        self.area_center_w = w_i
        self.area_center_h = h_i
        e, _, _ = self.calculate_elevation(self.area_center_w, self.area_center_h, self.segment_shift_w, self.segment_shift_h)
        # if _Debug:
        #     print(f'  map from {w0},{h0} shift:{w0shift},{h0shift} to {w_i},{h_i} with e:{e} new shift is {self.segment_shift_w},{self.segment_shift_h}')
        planet_shift_y = e # self.PLANET_RADIUS + e * self.ELEVATION_FACTOR
        self.global_land_translate_before.y = -planet_shift_y
        self.global_land_translate_after.y = planet_shift_y
        camera_shift_angle_x, camera_shift_angle_z = self.coords_area2angles(0.5-self.segment_shift_w, 0.5-self.segment_shift_h, as_area=False)
        self.global_land_rotate_x.angle = camera_shift_angle_x
        self.global_land_rotate_z.angle = camera_shift_angle_z
        added = 0
        removed = 0
        if wd != 0 or hd != 0:
            # new_area_left = w_i - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
            # new_area_top = h_i - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
            # new_area_right = w_i + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF - 1
            # new_area_bottom = h_i + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF - 1
            for unit_name in self.units.keys():
                u = self.units[unit_name]
                if u.static:
                    continue
                u.area_w -= wd
                u.area_h -= hd
                segment_angle_x, segment_angle_z = self.coords_area2angles(u.area_w, u.area_h)
                u.rotate_axis_x.angle = segment_angle_x
                u.rotate_axis_z.angle = segment_angle_z
            to_remove = []
            for w_t, h_t in self.land_tiles_visible.keys():
                _w = w_t - w_i
                _h = h_t - h_i
                area_w, area_h, segment_rotate_x, segment_rotate_z, static_units_at_segment = self.land_tiles_visible[(w_t, h_t)]
                if (_w, _h) in self.land_area_mask:
                # if new_area_left <= w_t and w_t <= new_area_right and new_area_top <= h_t and h_t <= new_area_bottom:
                    area_w -= wd
                    area_h -= hd
                    segment_angle_x, segment_angle_z = self.coords_area2angles(area_w, area_h)
                    segment_rotate_x.angle = segment_angle_x
                    segment_rotate_z.angle = segment_angle_z
                    self.land_tiles_visible[(w_t, h_t)][0] = area_w
                    self.land_tiles_visible[(w_t, h_t)][1] = area_h
                    for static_unit_name in static_units_at_segment:
                        static_unit = self.units[static_unit_name]
                        static_unit.rotate_axis_x.angle = segment_angle_x
                        static_unit.rotate_axis_z.angle = segment_angle_z
                else:
                    to_remove.append((w_t, h_t))
            for w_t, h_t in to_remove:
                self.remove_land_segment(w_t, h_t)
                removed += 1
            for _w, _h in self.land_area_mask.keys():
                w_t = w_i + _w
                h_t = h_i + _h
                if (w_t, h_t) not in self.land_tiles_visible:
                    self.add_land_segment(w_t, h_t, _w, _h)
                    added += 1
            # for _w in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
            #     for _h in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
            #         w_t = new_area_left + _w
            #         h_t = new_area_top + _h
            #         if (w_t, h_t) not in self.land_tiles_visible:
            #             self.add_land_segment(w_t, h_t, _w, _h)
            #             added += 1
            # self.land_area_left = new_area_left
            # self.land_area_top = new_area_top
            # if _Debug:
            #     print(f'        updated land area at {w_i} {h_i}, moved by {wd},{hd} shift:{round(self.segment_shift_w, 2)},{round(self.segment_shift_h, 2)} segments added:{added} removed:{removed}')
        # else:
        #     # for u in self.units.values():
        #     #     if u.static:
        #     #         continue
        #     #     segment_angle_z = mth.w2lat_degrees(float(u.area_w) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        #     #     segment_angle_x = mth.h2lon_degrees(float(u.area_h) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        #
        #     for k in self.land_tiles_visible.keys():
        #         w_t, h_t = k
        #         area_w, area_h, segment_rotate_x, segment_rotate_z, static_units_at_segment = self.land_tiles_visible[(w_t, h_t)]
        #         segment_angle_z = mth.w2lat_degrees(float(area_w) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        #         segment_angle_x = mth.h2lon_degrees(float(area_h) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        #         segment_rotate_x.angle = segment_angle_x
        #         segment_rotate_z.angle = segment_angle_z
        #         self.land_tiles_visible[(w_t, h_t)][0] = area_w
        #         self.land_tiles_visible[(w_t, h_t)][1] = area_h
        #         for static_unit_name in static_units_at_segment:
        #             static_unit = self.units[static_unit_name]
        #             static_unit.rotate_axis_x.angle = segment_angle_x
        #             static_unit.rotate_axis_z.angle = segment_angle_z
            # if _Debug:
            #     print(f'        updated land area at {w_i} {h_i}, shift:{round(self.segment_shift_w, 2)},{round(self.segment_shift_h, 2)}')

    def add_land_segment(self, map_w, map_h, area_w, area_h):
        # _get_elevation = self.land.get_elevation
        _get_texture = self.land.get_texture
        w_t = int(map_w)
        h_t = int(map_h)
        w = float(area_w)
        h = float(area_h)
        e00, e01, e10, e11 = self.get_segment_elevation(w_t, h_t)
        e_min = min(e00, e01, e10, e11)
        e_max = max(e00, e01, e10, e11)
        e_correction = (e_max - e_min) * 0.15
        y00 = e00 * self.SEGMENT_COS
        y01 = e01 * self.SEGMENT_COS
        y10 = e10 * self.SEGMENT_COS
        y11 = e11 * self.SEGMENT_COS
        c00 = e00 * self.SEGMENT_SIN
        c01 = e01 * self.SEGMENT_SIN
        c10 = e10 * self.SEGMENT_SIN
        c11 = e11 * self.SEGMENT_SIN
        v00 = (c00 * self.PI_4_SIN, y00, -c00 * self.PI_4_COS)
        v10 = (-c10 * self.PI_4_COS, y10, -c10 * self.PI_4_SIN)
        v01 = (c01 * self.PI_4_COS, y01, c01 * self.PI_4_SIN)
        v11 = (-c11 * self.PI_4_SIN, y11, c11 * self.PI_4_COS)
        tex_source, rotate = _get_texture(w_t, h_t)
        if rotate == 270:
            tex_coord00 = (0.0, 1.0)
            tex_coord01 = (1.0, 1.0)
            tex_coord10 = (0.0, 0.0)
            tex_coord11 = (1.0, 0.0)
        elif rotate == 0:
            tex_coord00 = (0.0, 0.0)
            tex_coord01 = (0.0, 1.0)
            tex_coord10 = (1.0, 0.0)
            tex_coord11 = (1.0, 1.0)
        elif rotate == 90:
            tex_coord00 = (1.0, 0.0)
            tex_coord01 = (0.0, 0.0)
            tex_coord10 = (1.0, 1.0)
            tex_coord11 = (0.0, 1.0)
        elif rotate == 180:
            tex_coord00 = (1.0, 1.0)
            tex_coord01 = (1.0, 0.0)
            tex_coord10 = (0.0, 1.0)
            tex_coord11 = (0.0, 0.0)
        vert = [
            v00[0], v00[1], v00[2], 1, 0, 0, tex_coord00[0], tex_coord00[1],
            v01[0], v01[1], v01[2], 1, 0, 0, tex_coord01[0], tex_coord01[1],
            v10[0], v10[1], v10[2], 1, 0, 0, tex_coord10[0], tex_coord10[1],
            v11[0], v11[1], v11[2], 1, 0, 0, tex_coord11[0], tex_coord11[1],
        ]
        segment_group_name = f'land_{map_w}_{map_h}'
        segment_rotate_x = Rotate(0, 1, 0, 0, group=segment_group_name)
        segment_rotate_z = Rotate(0, 0, 0, 1, group=segment_group_name)
        segment_angle_x, segment_angle_z = self.coords_area2angles(w, h)
        segment_rotate_x.angle = segment_angle_x
        segment_rotate_z.angle = segment_angle_z
        self.container_land_tiles.add(PushMatrix(group=segment_group_name))
        self.container_land_tiles.add(segment_rotate_x)
        self.container_land_tiles.add(segment_rotate_z)
        # if _Debug:
        #     if map_w == self.area_center_w and map_h == self.area_center_h:
        #         tex_source = None
        self.container_land_tiles.add(BindTexture(source=tex_source, index=1, group=segment_group_name))
        self.container_land_tiles.add(Mesh(
            vertices=vert,
            indices=[0, 1, 2, 1, 2, 3],
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=segment_group_name,
        ))
        static_units_at_segment = []
        if (w_t, h_t) in self.land.plants_map_data:
            for i in range(len(self.land.plants_map_data[(w_t, h_t)])):
                plant = self.land.plants_map_data[(w_t, h_t)][i]
                plant_variant = None
                static_object_name = None
                plant_key = plant['k']
                if plant_key in self.land.plants_variants:
                    plant_variant = self.land.plants_variants[plant_key]
                    if plant_variant['so']:
                        static_object_name = plant_variant['so']
                if not static_object_name:
                    so = self.create_static_object_from_model_data(
                        template=plant_variant['m'],
                        coefs=plant_variant['c'],
                        textures={'*': 'textures/model/' + plant_variant['t'] + '.png', },
                    )
                    static_object_name = so.name
                    if plant_key not in self.land.plants_variants:
                        variant = dict(plant)
                        variant.pop('x', None)
                        variant.pop('y', None)
                        variant['so'] = static_object_name
                        self.land.plants_variants[plant_key] = variant
                    else:
                        if not self.land.plants_variants[plant_key]['so']:
                            self.land.plants_variants[plant_key]['so'] = static_object_name
                self.land.plants_map_data[(w_t, h_t)][i]['so'] = static_object_name
                shift_vector = self.coords_map2xyz(w_t, h_t, plant['sw'], plant['sh'], elevation_correction=e_correction)
                unit = self.construct_unit_from_object_data(
                    container=self.container_static_objects,
                    object_name=static_object_name,
                    angle_coords=(
                        segment_angle_x,
                        segment_angle_z,
                    ),
                    shift_vector=shift_vector,
                    direction=45,  # random.randint(0, 360),
                    static=True,
                )
                static_units_at_segment.append(unit.name)
        self.container_land_tiles.add(PopMatrix(group=segment_group_name))
        self.land_tiles_visible[(w_t, h_t)] = [area_w, area_h, segment_rotate_x, segment_rotate_z, static_units_at_segment]
        # if _Debug:
        #     print(f'     added land segment at w:{map_w} h:{map_h} area_w:{area_w} area_h:{area_h} e_min:{e_min} with {len(static_units_at_segment)} static units')

    def remove_land_segment(self, w_t, h_t):
        tile_group_name = f'land_{w_t}_{h_t}'
        _, _, _, _, static_units_at_segment = self.land_tiles_visible[(w_t, h_t)]
        for static_unit_name in static_units_at_segment:
            self.remove_unit_from_stage(container=self.container_static_objects, unit_name=static_unit_name)
        self.container_land_tiles.remove_group(tile_group_name)
        self.land_tiles_visible.pop((w_t, h_t))

    def shift_land(self, shift_w, shift_h):
        if shift_h != 0:
            if shift_h > 0:
                if self.area_center_h + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF + 1 < self.map_height:
                    self.segment_shift_h = self.segment_shift_h + self.LAND_MOVE_SPEED
                    self.update_land()
            else:
                if self.area_center_h - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF > 0:
                    self.segment_shift_h = self.segment_shift_h - self.LAND_MOVE_SPEED
                    self.update_land()
        if shift_w != 0:
            if shift_w > 0:
                if self.area_center_w + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF + 1 < self.map_width:
                    self.segment_shift_w = self.segment_shift_w + self.LAND_MOVE_SPEED
                    self.update_land()
            else:
                if self.area_center_w - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF > 0:
                    self.segment_shift_w = self.segment_shift_w - self.LAND_MOVE_SPEED
                    self.update_land()

    def place_animated_unit_on_land(self, template, map_w, map_h, shift_w=0.5, shift_h=0.5, direction=0, texture=None, coefs=[0, 0, 0]):
        if template not in self.models:
            m = dat.ModelData()
            m.unpack_figure_data('figures.res', 'models', template=template)
            if texture:
                if not os.path.isfile('textures/model/' + texture + '.png'):
                    m.unpack_texture('textures.res', 'textures/model', texture)
            self.add_model_template(template, m)
        else:
            m = self.models[template]
        selected_parts = res.flat_tree(m.links[template]['ordered'])
        selected_animations = list(m.animations.keys())   
        ao = self.create_animated_object_from_model_data(
            template=template,
            coefs=coefs,
            selected_parts=selected_parts,
            selected_animations=selected_animations,
            textures={'*': 'textures/model/' + texture + '.png', },
        )
        map_w = int(map_w)
        map_h = int(map_h)
        area_w = map_w - int(self.area_center_w) 
        area_h = map_h - int(self.area_center_h)
        e_correction = 0  # (e_max - e_min) * 0.2
        if ao.root_mesh_center:
            e_correction = ao.root_mesh_center[0][2]
        segment_angle_x, segment_angle_z = self.coords_area2angles(area_w, area_h)
        shift_vector = self.coords_map2xyz(map_w, map_h, shift_w, shift_h, elevation_correction=e_correction)
        unit = self.construct_unit_from_object_data(
            container=self.container_animated_objects,
            object_name=ao.name,
            angle_coords=(
                segment_angle_x,
                segment_angle_z,
            ),
            shift_vector=shift_vector,
            direction=direction,
            static=False,
        )
        unit.w = map_w
        unit.h = map_h
        unit.shift_w = shift_w
        unit.shift_h = shift_h
        unit.area_w = area_w
        unit.area_h = area_h
        unit.direction = direction
        if unit.animations_list:
            unit.animation_playing = unit.animations_list[0]
        return unit

    def on_run_units(self, delta):
        new_land_pos = None
        for unit in self.units.values():
            if unit.static:
                continue
            unit.run(self)
            if self.renderer.camera_unit_lock and unit.name == self.renderer.camera_unit_lock:
                new_land_pos = (unit.w, unit.h, unit.shift_w, unit.shift_h)
        if new_land_pos:
            self.update_land(new_position=new_land_pos)

    def on_update_animations(self, delta):
        # return
        # TODO: maintain separate list of active animations for all units
        # then it is not required to loop all units
        for unit in self.units.values():
            if not unit.animations_list:
                continue
            ao = self.animated_objects[unit.object_name]
            animation = ao.animations[unit.animation_playing]
            root_part_name = ao.parts[0]
            root_part_animation = animation.parts.get(root_part_name)
            if unit.animation_frame >= root_part_animation.frames:
                # if _Debug:
                #     print(f'restarting unit ({unit.name}) animation {unit.animation_playing} after frame {unit.animation_frame}')
                unit.animation_frame = 0
            frame = unit.animation_frame
            for part_name in ao.parts:
                if part_name not in animation.parts:
                    continue
                part_animation = animation.parts.get(part_name)
                if not part_animation:
                    continue
                r = part_animation.rotation_frames[frame]
                t = part_animation.translation_frames[frame]
                mesh_transform = unit.meshes_transforms[part_name]
                translate_mat = Matrix()
                translate_mat.translate(t[0], t[1], t[2])
                mesh_transform.part_translate.matrix = translate_mat
                rotate_mat = Matrix()
                rotate_mat.set(array=mth.quaternion_to_matrix(r[0], r[1], r[2], r[3]))
                mesh_transform.part_rotate.matrix = rotate_mat.inverse()
            unit.animation_frame += 1
