import os
import sys
import json
import time
import math
import random
import numpy as np

from kivy.core.image import Image
from kivy.cache import Cache
from kivy.resources import resource_find
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
        self.onstage = False
        self.parts = []
        self.meshes_transforms = {}
        self.meshes_names = {}
        self.root_mesh_center = None
        self.rotate_axis_x = None
        self.rotate_axis_z = None
        self.rotate_vertical = None
        self.translate_shift = None
        self.context_state = None
        self.animations_list = []
        self.animation_playing = None
        self.animation_frame = 0
        self.direction = 0.0
        self.acceleration = 0.0
        self.speed = 0.0
        self.max_speed = 0.0

    def run(self, scene):
        if self.speed > self.max_speed:
            self.speed -= self.acceleration
        else:
            self.speed += self.acceleration
        # if self.speed > self.max_speed:
        #     self.speed = self.max_speed
        self.direction += 1.0
        if self.direction > 360.0:
            # self.direction = random.randint(0, 360)
            # self.max_speed = float(random.randint(1, 100)) / 10000.0
            # self.acceleration = float(random.randint(1, 2)) / 10000.0
            self.max_speed = random.randint(5, 50) / 10000.0
            self.acceleration = random.randint(1, 5) / 10000.0
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
            _w = int(self.w) - int(scene.area_center_w)
            _h = int(self.h) - int(scene.area_center_h)
            if self.onstage:
                if (_w, _h) in scene.land_area_mask:
                    pass
                else:
                    scene.hide_unit(container=scene.container_animated_objects, unit_name=self.name)
            else:
                if (_w, _h) in scene.land_area_mask:
                    scene.show_unit(container=scene.container_animated_objects, unit_name=self.name)


class Scene(object):

    SEGMENT_SIZE = 5.0
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
    ELEVATION_FACTOR = PLANET_RADIUS / 4.0
    ELEVATION_CORRECTION = 2.0
    VISIBLE_AREA_SIZE_SEGMENTS = 40
    VISIBLE_AREA_SIZE_SEGMENTS_HALF = int(VISIBLE_AREA_SIZE_SEGMENTS / 2.0)
    LAND_MOVE_SPEED = 0.2

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
        self.container_land_tiles = None
        self.container_static_objects = None
        self.container_animated_objects = None
        self.global_translate_before = None
        self.global_translate_after = None
        self.global_rotate_x = None
        self.global_rotate_z = None
        self.map_width = int(self.land.width / 2)
        self.map_height = int(self.land.height / 2)
        self.area_center_w = None
        self.area_center_h = None
        self.segment_shift_w = None
        self.segment_shift_h = None
        self.land_area_mask = {}
        self.land_tiles_visible = {}
        self.land_vertices = {}

    def coords_area2angles(self, w, h):
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

    def create_container(self):
        self.container = InstructionGroup()

    def init_scene(self, map_center_w, map_center_h):
        for _w in range(-self.VISIBLE_AREA_SIZE_SEGMENTS_HALF, self.VISIBLE_AREA_SIZE_SEGMENTS_HALF):
            for _h in range(-self.VISIBLE_AREA_SIZE_SEGMENTS_HALF, self.VISIBLE_AREA_SIZE_SEGMENTS_HALF):
                dist = int(math.sqrt(_w * _w + _h * _h))
                if dist < self.VISIBLE_AREA_SIZE_SEGMENTS_HALF:
                    self.land_area_mask[(_w, _h)] = dist
        self.global_rotate_x = Rotate(0, 1, 0, 0, group='land')
        self.global_rotate_z = Rotate(0, 0, 0, 1, group='land')
        self.area_center_w = int(map_center_w)
        self.area_center_h = int(map_center_h)
        self.segment_shift_w = 0.5
        self.segment_shift_h = 0.5
        w = int(self.area_center_w)
        h = int(self.area_center_h)
        camera_shift_angle_x, camera_shift_angle_z = self.coords_area2angles(0.5-self.segment_shift_w, 0.5-self.segment_shift_h)
        elevation_at_center = self.land.get_elevation(w * 2, h * 2)
        planet_shift_y = self.PLANET_RADIUS + elevation_at_center * self.ELEVATION_FACTOR + self.ELEVATION_CORRECTION
        self.global_translate_before = Translate(0, -planet_shift_y, 0, group='land')
        self.global_translate_after = Translate(0, planet_shift_y, 0, group='land')
        self.global_rotate_x.angle = camera_shift_angle_x
        self.global_rotate_z.angle = camera_shift_angle_z
        self.container.add(PushMatrix(group='land'))
        self.container.add(self.global_translate_before)
        self.container.add(self.global_rotate_x)
        self.container.add(self.global_rotate_z)
        self.container_animated_objects = InstructionGroup()
        self.container_static_objects = InstructionGroup()
        self.container_land_tiles = InstructionGroup()
        self.container.add(self.container_animated_objects)
        self.container.add(self.container_land_tiles)
        self.container.add(self.container_static_objects)
        self.container.add(self.global_translate_after)
        self.container.add(PopMatrix(group='land'))
        added = 0
        for k, dist_to_center in self.land_area_mask.items():
            _w, _h = k
            w_t = w + _w
            h_t = h + _h
            if (w_t, h_t) not in self.land_tiles_visible:
                self.add_land_segment(w_t, h_t, _w, _h, dist_to_center)
                added += 1
        if _Debug:
            print(f'prepare land area at {w} {h} with {added} segments planet angle x:0 z:0')

    def mesh_key(self, template, part_name, coefs):
        c = mth.quantize_coefs(coefs)
        return f'{template}_{part_name}_{c[0]}_{c[1]}_{c[2]}'

    # def get_segment_elevation(self, map_w, map_h):
    #     e00 = self.PLANET_RADIUS + self.land.get_elevation(map_w, map_h) * self.ELEVATION_FACTOR
    #     e01 = self.PLANET_RADIUS + self.land.get_elevation(map_w, map_h + 1) * self.ELEVATION_FACTOR
    #     e10 = self.PLANET_RADIUS + self.land.get_elevation(map_w + 1, map_h) * self.ELEVATION_FACTOR
    #     e11 = self.PLANET_RADIUS + self.land.get_elevation(map_w + 1, map_h + 1) * self.ELEVATION_FACTOR
    #     return e00, e01, e10, e11

    def get_segment_elevation(self, map_w, map_h):
        e00 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 - 1, map_h * 2 - 1) * self.ELEVATION_FACTOR
        e01 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 - 1, map_h * 2 + 0) * self.ELEVATION_FACTOR
        e02 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 - 1, map_h * 2 + 1) * self.ELEVATION_FACTOR
        e10 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 0, map_h * 2 - 1) * self.ELEVATION_FACTOR
        e11 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 0, map_h * 2 + 0) * self.ELEVATION_FACTOR
        e12 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 0, map_h * 2 + 1) * self.ELEVATION_FACTOR
        e20 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 1, map_h * 2 - 1) * self.ELEVATION_FACTOR
        e21 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 1, map_h * 2 + 0) * self.ELEVATION_FACTOR
        e22 = self.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 1, map_h * 2 + 1) * self.ELEVATION_FACTOR
        return e00, e01, e02, e10, e11, e12, e20, e21, e22

    def calculate_elevation(self, w_i, h_i, shift_w, shift_h):
        # e00, e01, e10, e11 = self.get_segment_elevation(w_i, h_i)
        e00, e01, e02, e10, e11, e12, e20, e21, e22 = self.get_segment_elevation(w_i, h_i)
        e_min = min(e00, e01, e02, e10, e11, e12, e20, e21, e22)
        e_max = max(e00, e01, e02, e10, e11, e12, e20, e21, e22)
        a = self.SEGMENT_ANGLE
        if shift_w < 0.5:
            if shift_h < 0.5:
                p00 = (0, 0, e00)
                p01 = (0, a, e01)
                p10 = (a, 0, e10)
                p11 = (a, a, e11)
            else:
                p00 = (0, 0, e01)
                p01 = (0, a, e02)
                p10 = (a, 0, e11)
                p11 = (a, a, e12)
        else:
            if shift_h < 0.5:
                p00 = (0, 0, e10)
                p01 = (0, a, e11)
                p10 = (a, 0, e20)
                p11 = (a, a, e21)
            else:
                p00 = (0, 0, e11)
                p01 = (0, a, e12)
                p10 = (a, 0, e21)
                p11 = (a, a, e22)
        w_f = shift_w * a
        h_f = shift_h * a
        if mth.point_line_left_or_right(w_f, h_f, p00[0], p00[1], p11[0], p11[1]) == -1:
            e = mth.get_z_in_triangle(w_f, h_f, p00, p11, p01)
        else:
            e = mth.get_z_in_triangle(w_f, h_f, p11, p00, p10)
        return e, e_min, e_max

    def calculate_land_vertices(self):
        t1 = time.time()
        for w in range(1, self.map_width-1):
            for h in range(1, self.map_height-1):
                e00, e01, e02, e10, e11, e12, e20, e21, e22 = self.get_segment_elevation(w, h)
                e_min = min(e00, e01, e02, e10, e11, e12, e20, e21, e22)
                e_max = max(e00, e01, e02, e10, e11, e12, e20, e21, e22)
                y00 = e00 * self.SEGMENT_COS
                y01 = e01 * self.SEGMENT_COS
                y02 = e02 * self.SEGMENT_COS
                y10 = e10 * self.SEGMENT_COS
                y11 = e11 * self.SEGMENT_COS
                y12 = e12 * self.SEGMENT_COS
                y20 = e20 * self.SEGMENT_COS
                y21 = e21 * self.SEGMENT_COS
                y22 = e22 * self.SEGMENT_COS
                c00 = e00 * self.SEGMENT_SIN
                c01 = e01 * self.SEGMENT_SIN
                c02 = e02 * self.SEGMENT_SIN
                c10 = e10 * self.SEGMENT_SIN
                c11 = e11 * self.SEGMENT_SIN
                c12 = e12 * self.SEGMENT_SIN
                c20 = e20 * self.SEGMENT_SIN
                c21 = e21 * self.SEGMENT_SIN
                c22 = e22 * self.SEGMENT_SIN
                v00 = (c00 * self.PI_4_SIN, y00, -c00 * self.PI_4_COS)
                v01 = (c01 * self.PI_4_COS, y01, c01 * 0)
                v02 = (c02 * self.PI_4_COS, y02, c02 * self.PI_4_SIN)
                v10 = (c10 * 0, y10, -c10 * self.PI_4_COS)
                v11 = (c11 * 0, y11, c11 * 0)
                v12 = (c12 * 0, y12, c12 * self.PI_4_SIN)
                v20 = (-c20 * self.PI_4_COS, y20, -c20 * self.PI_4_SIN)
                v21 = (-c21 * self.PI_4_COS, y21, c21 * 0)
                v22 = (-c22 * self.PI_4_SIN, y22, c22 * self.PI_4_COS)
                self.land_vertices[(w, h)] = (v00, v01, v02, v10, v11, v12, v20, v21, v22, e_min, e_max)
        t2 = time.time()
        if _Debug:
            print(f'calculated land vertices for {len(self.land.elevation_map_data)} segments in {t2 - t1} sec')

    def add_model_template(self, template, model):
        self.models[template] = model

    def create_mesh_from_fig_data(self, fig_data, prefix='', texture=None, coefs=[0, 0, 0]):
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
            material={'map_Kd': 'textures/model/' + texture + '.png'} if texture else None,
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

    def create_object_data_from_model_data(self, template, coefs=[0, 0, 0], selected_parts=[], excluded_parts=[], selected_animations=None, textures=None):
        global _NextObjectID
        _NextObjectID += 1
        if textures is None:
            textures = {'*': 'default0'}
        if template not in self.models:
            m = dat.ModelData()
            m.unpack_figure_data('data/figures.res', 'models', template=template)
            for texture in textures.values():
                tex_file_path = 'textures/model/' + texture + '.png'
                if not os.path.isfile(tex_file_path):
                    m.unpack_texture('data/textures.res', 'textures/model', texture)
                tex_file_path_source = resource_find(tex_file_path)
                if tex_file_path_source:
                    _tex = Cache.get('kv.texture', tex_file_path)
                    if not _tex:
                        _tex = Image(tex_file_path_source).texture
                        Cache.append('kv.texture', tex_file_path, _tex)
                        if _Debug:
                            print(f'  cached texture {texture} at {tex_file_path} for model {template}')
            self.add_model_template(template, m)
        m = self.models[template]
        static = False if selected_animations else True
        o = dat.ObjectData(name=template+'#'+str(_NextObjectID), static=static)
        coefs = mth.quantize_coefs(coefs)
        o.template = template
        o.textures = textures
        o.parts_tree_ordered = m.links[template]['ordered']
        o.parts_tree = m.links[template]['tree']
        o.parts_parents = m.links[template]['parents']
        ordered_parts_list = res.flat_tree(o.parts_tree_ordered)
        if selected_animations:
            if selected_animations == '*':
                o.animations_loaded = list(m.animations.keys())
            else:
                o.animations_loaded = selected_animations
        if not selected_parts:
            selected_parts = ordered_parts_list
        for exclude in excluded_parts:
            if exclude in selected_parts:
                selected_parts.remove(exclude)
        o.root_part_name = selected_parts[0]
        # if _Debug:
        #     print(f'about to prepare unit ({ao.name}) with {len(selected_parts)} parts and {len(ao.animations_loaded)} animations from model {{{template}}}')
        # t1 = time.time()
        for part_name in selected_parts:
            o.parts.append(part_name)
            part_info = m.bones[part_name]
            o.bones[part_name] = mth.ei2xyz_list([
                mth.trilinear([part_info[i][0] for i in range(8)], coefs),
                mth.trilinear([part_info[i][1] for i in range(8)], coefs),
                mth.trilinear([part_info[i][2] for i in range(8)], coefs),
            ])
            mesh_key = self.mesh_key(o.template, part_name, coefs)
            if mesh_key in self.meshes_index:
                mesh = self.meshes[self.meshes_index[mesh_key]]
                if _Debug:
                    print(f'    reused mesh {mesh.name} for part {o.name}:{part_name} with texture {mesh.material["map_Kd"]}')
            else:
                mesh = self.create_mesh_from_fig_data(
                    fig_data=m.figures[part_name],
                    prefix=o.template + '_' + part_name,
                    texture=o.textures[part_name] if part_name in o.textures else o.textures['*'],
                    coefs=coefs,
                )
                mesh.object_name = o.name
                mesh.object_part_name = part_name
                self.meshes_index[mesh_key] = mesh.name
            o.meshes[part_name] = mesh.name
            for anim_name in o.animations_loaded:
                if part_name not in m.animations[anim_name]:
                    continue
                animation_info = m.animations[anim_name][part_name]
                if anim_name not in o.animations:
                    o.animations[anim_name] = dat.ObjectAnimationData(template, anim_name)
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
                o.animations[anim_name].parts[part_name] = a
            if part_name == o.root_part_name:
                o.root_mesh_name = mesh.name
                o.root_mesh_center = mesh.center
        if static:
            self.static_objects[o.name] = o
        else:
            o.calculate_animations()
            self.animated_objects[o.name] = o
        # t2 = time.time()
        # if _Debug:
        #     print(f'  {"static" if static else "animated"} object {o.name} with {len(selected_parts)} parts and {len(o.animations_loaded)} animations created in {t2 - t1} sec from template {template}')
        return o

    def construct_unit_from_object_data(self, container, object_name, angle_coords, static=True, onstage=True, shift_vector=None, direction=0, w=0, h=0, shift_w=0, shift_h=0, area_w=0, area_h=0):
        global _NextUnitID
        _source_dict = self.static_objects if static else self.animated_objects
        if object_name not in _source_dict:
            raise Exception(f'Model data object {object_name} was not prepared')
        if not shift_vector:
            shift_vector = [0.0, 0.0, 0.0]
        _NextUnitID += 1
        unit = Unit(name=object_name+'#'+str(_NextUnitID), object_name=object_name)
        unit.static = static
        unit.onstage = onstage
        if static:
            source_object = self.static_objects[object_name]
        else:
            source_object = self.animated_objects[object_name]
            unit.animations_list = source_object.animations_loaded.copy()
        unit.root_mesh_center = source_object.root_mesh_center
        unit.direction = direction
        unit.w = w
        unit.h = h
        unit.shift_w = shift_w
        unit.shift_h = shift_h
        unit.area_w = area_w
        unit.area_h = area_h

        def _visitor(part_name, parent_part_name):
            mesh_name = source_object.meshes[part_name]
            mesh = self.meshes[mesh_name]
            if part_name in unit.meshes_transforms:
                raise Exception(f'Mesh transform for part [{part_name}] of unit ({unit.name}) already exists')
            mesh_transform = dat.MeshTransformData()
            mesh_transform.part_translate = Transform(group=unit.name)
            mesh_transform.part_rotate = Transform(group=unit.name)
            unit.meshes_transforms[part_name] = mesh_transform
            unit.parts.append(part_name)
            unit.meshes_names[part_name] = mesh_name
            if onstage:
                container.add(PushMatrix(group=unit.name))
                container.add(mesh_transform.part_translate)
                container.add(PushMatrix(group=unit.name))
                container.add(mesh_transform.part_rotate)
                container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=unit.name))
                container.add(Mesh(
                    vertices=mesh.vertices,
                    indices=mesh.indices,
                    fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                    mode='triangles',
                    group=unit.name,
                ))
                container.add(PopMatrix(group=unit.name))  # part_rotate
                container.add(PopMatrix(group=unit.name))  # part_translate

        unit.rotate_axis_x = Rotate(angle_coords[0], 1, 0, 0, group=unit.name)
        unit.rotate_axis_z = Rotate(angle_coords[1], 0, 0, 1, group=unit.name)
        unit.rotate_vertical = Rotate(direction + 90, 0, 1, 0, group=unit.name)
        unit.translate_shift = Translate(shift_vector[0], shift_vector[1], shift_vector[2], group=unit.name)
        unit.context_state = ChangeState(material_density=0.0, group=unit.name)
        if onstage:
            container.add(PushMatrix(group=unit.name))  # unit
            container.add(unit.rotate_axis_x)
            container.add(unit.rotate_axis_z)
            container.add(PushMatrix(group=unit.name))  # unit shift
            container.add(unit.translate_shift)
            container.add(PushMatrix(group=unit.name))  # unit rotate
            container.add(unit.rotate_vertical)
            container.add(unit.context_state)

        source_object.walk_parts_ordered(_visitor)
        if onstage:
            container.add(ChangeState(material_density=0.0, group=unit.name))
            container.add(PopMatrix(group=unit.name))  # unit rotate
            container.add(PopMatrix(group=unit.name))  # unit shift
            container.add(PopMatrix(group=unit.name))  # unit
        self.units[unit.name] = unit
        # if static is False:
        #     if _Debug:
        #         print(f'created animated unit at {angle_coords} with shift {shift_vector} from object {object_name} and placed on scene')
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

    def hide_unit(self, container, unit_name):
        if unit_name not in self.units:
            raise Exception(f'Unit {unit_name} is not on the stage at the moment')
        unit = self.units[unit_name]
        if not unit.onstage:
            raise Exception(f'Unit {unit_name} is already hidden')
        container.remove_group(unit.name)
        unit.onstage = False
        if _Debug:
            print(f'  unit {unit.name} was hidden')

    def show_unit(self, container, unit_name):
        if unit_name not in self.units:
            raise Exception(f'Unit {unit_name} is not on the stage at the moment')
        unit = self.units[unit_name]
        if unit.onstage:
            raise Exception(f'Unit {unit_name} is already visible')
        source_object = self.static_objects[unit.object_name] if unit.static else self.animated_objects[unit.object_name]

        def _visitor(part_name, parent_part_name):
            mesh_name = unit.meshes_names[part_name]
            mesh = self.meshes[mesh_name]
            mesh_transform = unit.meshes_transforms[part_name]
            container.add(PushMatrix(group=unit.name))
            container.add(mesh_transform.part_translate)
            container.add(PushMatrix(group=unit.name))
            container.add(mesh_transform.part_rotate)
            container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=unit.name))
            container.add(Mesh(
                vertices=mesh.vertices,
                indices=mesh.indices,
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=unit.name,
            ))
            container.add(PopMatrix(group=unit.name))  # part_rotate
            container.add(PopMatrix(group=unit.name))  # part_translate

        # open unit context and prepare transforms and state
        container.add(PushMatrix(group=unit.name))  # unit
        container.add(unit.rotate_axis_x)
        container.add(unit.rotate_axis_z)
        container.add(PushMatrix(group=unit.name))  # unit shift
        container.add(unit.translate_shift)
        container.add(PushMatrix(group=unit.name))  # unit rotate
        container.add(unit.rotate_vertical)
        container.add(unit.context_state)
        # push unit meshes
        source_object.walk_parts_ordered(_visitor)
        # close unit context
        container.add(ChangeState(material_density=0.0, group=unit.name))
        container.add(PopMatrix(group=unit.name))  # unit rotate
        container.add(PopMatrix(group=unit.name))  # unit shift
        container.add(PopMatrix(group=unit.name))  # unit
        unit.onstage = True
        if _Debug:
            print(f'  unit {unit.name} was shown')

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
        planet_shift_y = e + self.ELEVATION_CORRECTION # self.PLANET_RADIUS + e * self.ELEVATION_FACTOR
        self.global_translate_before.y = -planet_shift_y
        self.global_translate_after.y = planet_shift_y
        camera_shift_angle_x, camera_shift_angle_z = self.coords_area2angles(0.5-self.segment_shift_w, 0.5-self.segment_shift_h)
        self.global_rotate_x.angle = camera_shift_angle_x
        self.global_rotate_z.angle = camera_shift_angle_z
        added = 0
        removed = 0
        if wd != 0 or hd != 0:
            for unit_name in self.units.keys():
                unit = self.units[unit_name]
                if unit.static:
                    continue
                if not unit.onstage:
                    continue
                unit.area_w = unit.w - w_i
                unit.area_h = unit.h - h_i
                segment_angle_x, segment_angle_z = self.coords_area2angles(unit.area_w, unit.area_h)
                unit.rotate_axis_x.angle = segment_angle_x
                unit.rotate_axis_z.angle = segment_angle_z
            to_remove = []
            for w_t, h_t in self.land_tiles_visible.keys():
                _w = w_t - w_i
                _h = h_t - h_i
                area_w, area_h, segment_rotate_x, segment_rotate_z, static_units_at_segment, _ = self.land_tiles_visible[(w_t, h_t)]
                if (_w, _h) in self.land_area_mask:
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
                        static_unit.area_w = area_w
                        static_unit.area_h = area_h
                else:
                    to_remove.append((w_t, h_t))
            for w_t, h_t in to_remove:
                self.remove_land_segment(w_t, h_t)
                removed += 1
            for unit in self.units.values():
                if unit.static:
                    continue
                if not unit.onstage:
                    continue
                _w = unit.w - w_i
                _h = unit.h - h_i
                if (_w, _h) not in self.land_area_mask:
                    self.hide_unit(container=self.container_animated_objects, unit_name=unit.name)
            for k, dist_to_center in self.land_area_mask.items():
                _w, _h = k
                w_t = w_i + _w
                h_t = h_i + _h
                if (w_t, h_t) not in self.land_tiles_visible:
                    self.add_land_segment(w_t, h_t, _w, _h, dist_to_center)
                    added += 1

    def add_land_segment(self, map_w, map_h, area_w, area_h, dist_to_center):
        _get_texture = self.land.get_texture
        w_t = int(map_w)
        h_t = int(map_h)
        w = float(area_w)
        h = float(area_h)
        v00, v01, v02, v10, v11, v12, v20, v21, v22, e_min, e_max = self.land_vertices[(w_t, h_t)]
        e_correction = (e_max - e_min) * 0.18
        tex00_file_path, tex00_coord00, tex00_coord01, tex00_coord10, tex00_coord11 = _get_texture(w_t*2, h_t*2)
        tex01_file_path, tex01_coord00, tex01_coord01, tex01_coord10, tex01_coord11 = _get_texture(w_t*2, h_t*2+1)
        tex10_file_path, tex10_coord00, tex10_coord01, tex10_coord10, tex10_coord11 = _get_texture(w_t*2+1, h_t*2)
        tex11_file_path, tex11_coord00, tex11_coord01, tex11_coord10, tex11_coord11 = _get_texture(w_t*2+1, h_t*2+1)
        vert00 = [
            v00[0], v00[1], v00[2], 1, 0, 0, tex00_coord00[0], tex00_coord00[1],
            v01[0], v01[1], v01[2], 1, 0, 0, tex00_coord01[0], tex00_coord01[1],
            v10[0], v10[1], v10[2], 1, 0, 0, tex00_coord10[0], tex00_coord10[1],
            v11[0], v11[1], v11[2], 1, 0, 0, tex00_coord11[0], tex00_coord11[1],
        ]
        vert01 = [
            v01[0], v01[1], v01[2], 1, 0, 0, tex01_coord00[0], tex01_coord00[1],
            v02[0], v02[1], v02[2], 1, 0, 0, tex01_coord01[0], tex01_coord01[1],
            v11[0], v11[1], v11[2], 1, 0, 0, tex01_coord10[0], tex01_coord10[1],
            v12[0], v12[1], v12[2], 1, 0, 0, tex01_coord11[0], tex01_coord11[1],
        ]
        vert10 = [
            v10[0], v10[1], v10[2], 1, 0, 0, tex10_coord00[0], tex10_coord00[1],
            v11[0], v11[1], v11[2], 1, 0, 0, tex10_coord01[0], tex10_coord01[1],
            v20[0], v20[1], v20[2], 1, 0, 0, tex10_coord10[0], tex10_coord10[1],
            v21[0], v21[1], v21[2], 1, 0, 0, tex10_coord11[0], tex10_coord11[1],
        ]
        vert11 = [
            v11[0], v11[1], v11[2], 1, 0, 0, tex11_coord00[0], tex11_coord00[1],
            v12[0], v12[1], v12[2], 1, 0, 0, tex11_coord01[0], tex11_coord01[1],
            v21[0], v21[1], v21[2], 1, 0, 0, tex11_coord10[0], tex11_coord10[1],
            v22[0], v22[1], v22[2], 1, 0, 0, tex11_coord11[0], tex11_coord11[1],
        ]
        segment_group_name = f'l_{map_w}_{map_h}'
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
        segment_state = ChangeState(material_density=1.0, group=segment_group_name)
        self.container_land_tiles.add(segment_state)
        self.container_land_tiles.add(BindTexture(source=tex00_file_path, index=1, group=segment_group_name))
        self.container_land_tiles.add(Mesh(
            vertices=vert00,
            indices=[0, 1, 2, 1, 2, 3],
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=segment_group_name,
        ))
        self.container_land_tiles.add(BindTexture(source=tex01_file_path, index=1, group=segment_group_name))
        self.container_land_tiles.add(Mesh(
            vertices=vert01,
            indices=[0, 1, 2, 1, 2, 3],
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=segment_group_name,
        ))
        self.container_land_tiles.add(BindTexture(source=tex10_file_path, index=1, group=segment_group_name))
        self.container_land_tiles.add(Mesh(
            vertices=vert10,
            indices=[0, 1, 2, 1, 2, 3],
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=segment_group_name,
        ))
        self.container_land_tiles.add(BindTexture(source=tex11_file_path, index=1, group=segment_group_name))
        self.container_land_tiles.add(Mesh(
            vertices=vert11,
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
                    so = self.create_object_data_from_model_data(
                        template=plant_variant['m'],
                        coefs=plant_variant['c'],
                        textures={'*': plant_variant['t']},
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
                    direction=random.randint(0, 360),
                    static=True,
                    onstage=True,
                    w=map_w,
                    h=map_h,
                    shift_w=plant['sw'],
                    shift_h=plant['sh'],
                    area_w=area_w,
                    area_h=area_h,
                )
                static_units_at_segment.append(unit.name)
        for unit in self.units.values():
            if unit.static:
                continue
            if unit.onstage:
                continue
            if int(unit.w) == w_t and int(unit.h) == h_t:
                self.show_unit(container=self.container_animated_objects, unit_name=unit.name)
        self.container_land_tiles.add(ChangeState(material_density=0.0, group=segment_group_name))
        self.container_land_tiles.add(PopMatrix(group=segment_group_name))
        self.land_tiles_visible[(w_t, h_t)] = [area_w, area_h, segment_rotate_x, segment_rotate_z, static_units_at_segment, segment_state]
        # if _Debug:
        #     print(f'     added land segment at w:{map_w} h:{map_h} area_w:{area_w} area_h:{area_h} e_min:{e_min} with {len(static_units_at_segment)} static units')

    def remove_land_segment(self, w_t, h_t):
        tile_group_name = f'l_{w_t}_{h_t}'
        _, _, _, _, static_units_at_segment, _ = self.land_tiles_visible[(w_t, h_t)]
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

    def place_animated_unit_on_land(self, template, map_w, map_h, shift_w=0.5, shift_h=0.5, direction=0, textures=None, coefs=[0, 0, 0]):
        ao = self.create_object_data_from_model_data(
            template=template,
            coefs=coefs,
            selected_animations='*',
            textures=textures,
        )
        map_w = int(map_w)
        map_h = int(map_h)
        area_w = map_w - int(self.area_center_w) 
        area_h = map_h - int(self.area_center_h)
        e_correction = 0
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
            w=map_w,
            h=map_h,
            shift_w=shift_w,
            shift_h=shift_h,
            area_w=area_w,
            area_h=area_h,
        )
        if unit.animations_list:
            unit.animation_playing = unit.animations_list[0]
        return unit

    def on_run_units(self, delta):
        if self.renderer.camera_unit_lock:
            u = self.units.get(self.renderer.camera_unit_lock)
            if u:
                self.update_land(new_position=(u.w, u.h, u.shift_w, u.shift_h))
        for unit in self.units.values():
            if unit.static:
                continue
            unit.run(self)

    def on_update_animations(self, delta):
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
                current_animation = unit.animations_list.index(unit.animation_playing)
                # current_animation += 1
                if current_animation >= len(unit.animations_list):
                    current_animation = 0
                unit.animation_playing = unit.animations_list[current_animation]
                animation = ao.animations[unit.animation_playing]
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
