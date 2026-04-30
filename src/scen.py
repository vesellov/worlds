import os
import sys
import json
import time
import math
import random
import numpy as np

from kivy.core.image import Image
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.resources import resource_find
from kivy.graphics import (
    RenderContext, Callback, BindTexture,
    ChangeState, PushState, PopState,
    PushMatrix, PopMatrix,
    Color, Translate, Rotate, Mesh,
)
from kivy.graphics.instructions import InstructionGroup  # @UnresolvedImport
from kivy.graphics.context_instructions import Transform  # @UnresolvedImport


import const
import res
import mth
import dat
import unt


_Debug = True

QUADRO_SEGMENTS = False

_NextUnitID = 0
_NextObjectID = 0
_NextMeshID = 0


class Scene(object):

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
        self.container_water_tiles = None
        self.global_translate_before = None
        self.global_translate_after = None
        self.global_rotate_x = None
        self.global_rotate_z = None
        self.global_water_translate_before = None
        self.global_water_translate_after = None
        self.global_water_rotate_x = None
        self.global_water_rotate_z = None
        if QUADRO_SEGMENTS:
            self.map_width = int(self.land.width / 2)
            self.map_height = int(self.land.height / 2)
        else:
            self.map_width = int(self.land.width)
            self.map_height = int(self.land.height)
        self.area_center_w = None
        self.area_center_h = None
        self.segment_shift_w = None
        self.segment_shift_h = None
        self.visible_area_mask = {}
        self.land_tiles_visible = {}
        self.land_vertices = {}
        self.water_tiles_visible = {}
        self.segments_waiting = set()
        self.segments_queue = []
        self.segments_cleanup_queue = []

    def coords_area2angles(self, w, h):
        angle_z = mth.w2lat_degrees(float(w), const.PLANET_EQUATOR_SEGMENTS)
        angle_x = mth.h2lon_degrees(float(h), const.PLANET_EQUATOR_SEGMENTS)
        return angle_x, angle_z

    def coords_map2xyz(self, map_w, map_h, shift_w, shift_h, elevation_correction=None):
        e, _, _ = self.calculate_elevation(map_w, map_h, shift_w, shift_h)
        c = e * const.SEGMENT_SIN
        e_correction = 0
        if elevation_correction is not None:
            e_correction = elevation_correction
        return (
            c * const.PI_4_SIN * ((0.5 - shift_w) * 2.0),
            e - e_correction,
            c * const.PI_4_COS * ((shift_h - 0.5) * 2.0),
        )

    def create_container(self):
        self.container = InstructionGroup()

    def init_scene(self, map_center_w, map_center_h):
        for _w in range(-const.VISIBLE_AREA_SIZE_SEGMENTS_HALF, const.VISIBLE_AREA_SIZE_SEGMENTS_HALF):
            for _h in range(-const.VISIBLE_AREA_SIZE_SEGMENTS_HALF, const.VISIBLE_AREA_SIZE_SEGMENTS_HALF):
                dist = int(math.sqrt(_w * _w + _h * _h))
                if dist < const.VISIBLE_AREA_SIZE_SEGMENTS_HALF:
                    self.visible_area_mask[(_w, _h)] = dist
        if QUADRO_SEGMENTS:
            self.area_center_w = int(map_center_w / 2)
            self.area_center_h = int(map_center_h / 2)
        else:
            self.area_center_w = int(map_center_w)
            self.area_center_h = int(map_center_h)
        self.segment_shift_w = 0.5
        self.segment_shift_h = 0.5
        w = int(self.area_center_w)
        h = int(self.area_center_h)
        camera_shift_angle_x, camera_shift_angle_z = self.coords_area2angles(0.5-self.segment_shift_w, 0.5-self.segment_shift_h)
        if QUADRO_SEGMENTS:
            elevation_at_center = self.land.get_elevation(w * 2, h * 2)
        else:
            elevation_at_center = self.land.get_elevation(w, h)
        planet_shift_y = const.PLANET_RADIUS + elevation_at_center * const.ELEVATION_FACTOR + const.ELEVATION_CORRECTION
        self.global_translate_before = Translate(0, -planet_shift_y, 0, group='land')
        self.global_translate_after = Translate(0, planet_shift_y, 0, group='land')
        self.global_rotate_x = Rotate(camera_shift_angle_x, 1, 0, 0, group='land')
        self.global_rotate_z = Rotate(camera_shift_angle_z, 0, 0, 1, group='land')
        self.container_animated_objects = InstructionGroup()
        self.container_static_objects = InstructionGroup()
        self.container_land_tiles = InstructionGroup()
        self.container.add(PushMatrix(group='land'))
        self.container.add(self.global_translate_before)
        self.container.add(self.global_rotate_x)
        self.container.add(self.global_rotate_z)
        self.container.add(self.container_animated_objects)
        self.container.add(self.container_land_tiles)
        self.container.add(self.container_static_objects)
        self.container.add(self.global_translate_after)
        self.container.add(PopMatrix(group='land'))
        added = 0
        for k, dist_to_center in self.visible_area_mask.items():
            _w, _h = k
            w_t = w + _w
            h_t = h + _h
            if (w_t, h_t) not in self.land_tiles_visible:
                if (w_t, h_t) not in self.segments_waiting:
                    self.segments_queue.append((w_t, h_t, dist_to_center))
                    self.segments_waiting.add((w_t, h_t))
                    added += 1
        self.global_water_translate_before = Translate(0, -planet_shift_y, 0, group='water')
        self.global_water_translate_after = Translate(0, planet_shift_y, 0, group='water')
        self.global_water_rotate_x = Rotate(camera_shift_angle_x, 1, 0, 0, group='water')
        self.global_water_rotate_z = Rotate(camera_shift_angle_z, 0, 0, 1, group='water')
        self.container_water_tiles = InstructionGroup()
        self.container.add(PushMatrix(group='water'))
        self.container.add(self.global_water_translate_before)
        self.container.add(self.global_water_rotate_x)
        self.container.add(self.global_water_rotate_z)
        self.container.add(self.container_water_tiles)
        for k, dist_to_center in self.visible_area_mask.items():
            _w, _h = k
            w_t = w + _w
            h_t = h + _h
            if (w_t, h_t) not in self.water_tiles_visible:
                segment_angle_x, segment_angle_z = self.coords_area2angles(_w, _h)
                water_segment_group_name = f'w_{w_t}_{h_t}'
                ew = const.WATER_LEVEL_ELEVATION #  + (8.0 / 255.0 ) * self.ELEVATION_FACTOR
                y00 = ew * const.SEGMENT_COS
                y01 = ew * const.SEGMENT_COS
                y10 = ew * const.SEGMENT_COS
                y11 = ew * const.SEGMENT_COS
                c00 = ew * const.SEGMENT_SIN
                c01 = ew * const.SEGMENT_SIN
                c10 = ew * const.SEGMENT_SIN
                c11 = ew * const.SEGMENT_SIN
                v00 = (c00 * const.PI_4_SIN, y00, -c00 * const.PI_4_COS)
                v01 = (c01 * const.PI_4_COS, y01, c01 * const.PI_4_SIN)
                v10 = (-c10 * const.PI_4_COS, y10, -c10 * const.PI_4_SIN)
                v11 = (-c11 * const.PI_4_SIN, y11, c11 * const.PI_4_COS)
                water_vertices = [
                    v00[0], v00[1], v00[2], 1, 0, 0, 0.0, 0.0,
                    v01[0], v01[1], v01[2], 1, 0, 0, 0.0, 1.0,
                    v10[0], v10[1], v10[2], 1, 0, 0, 1.0, 0.0,
                    v11[0], v11[1], v11[2], 1, 0, 0, 1.0, 1.0,
                ]
                water_segment_rotate_x = Rotate(segment_angle_x, 1, 0, 0, group=water_segment_group_name)
                water_segment_rotate_z = Rotate(segment_angle_z, 0, 0, 1, group=water_segment_group_name)
                self.container_water_tiles.add(PushMatrix(group=water_segment_group_name))
                self.container_water_tiles.add(water_segment_rotate_x)
                self.container_water_tiles.add(water_segment_rotate_z)
                self.container_water_tiles.add(ChangeState(material_density=0.0, water_transparency=(128.0 / 255.0), group=water_segment_group_name))
                self.container_water_tiles.add(BindTexture(source='assets/water8a.png', index=1, group=water_segment_group_name))
                self.container_water_tiles.add(Mesh(
                    vertices=water_vertices,
                    indices=[0, 1, 2, 1, 3, 2],
                    fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                    mode='triangles',
                    group=water_segment_group_name,
                ))
                self.container_water_tiles.add(ChangeState(material_density=0.0, water_transparency=(10.0 / 255.0), group=water_segment_group_name))
                self.container_water_tiles.add(PopMatrix(group=water_segment_group_name))
                self.water_tiles_visible[(w_t, h_t)] = [_w, _h, water_segment_rotate_x, water_segment_rotate_z]
        self.container.add(self.global_water_translate_after)
        self.container.add(PopMatrix(group='water'))
        if _Debug:
            print(f'visible area created at {w} {h} with {added} segments, elevation at center is {elevation_at_center} with planet shift {planet_shift_y}, queue is {len(self.segments_waiting)} / {len(self.segments_cleanup_queue)}')
        self.update_segments()

    def mesh_key(self, template, part_name, coefs):
        c = mth.quantize_coefs(coefs)
        return f'{template}_{part_name}_{c[0]}_{c[1]}_{c[2]}'

    def get_segment_elevation(self, map_w, map_h):
        if QUADRO_SEGMENTS:
            e00 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2,     map_h * 2    ) * const.ELEVATION_FACTOR
            e01 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2,     map_h * 2 + 1) * const.ELEVATION_FACTOR
            e02 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2,     map_h * 2 + 2) * const.ELEVATION_FACTOR
            e10 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 1, map_h * 2    ) * const.ELEVATION_FACTOR
            e11 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 1, map_h * 2 + 1) * const.ELEVATION_FACTOR
            e12 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 1, map_h * 2 + 2) * const.ELEVATION_FACTOR
            e20 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 2, map_h * 2    ) * const.ELEVATION_FACTOR
            e21 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 2, map_h * 2 + 1) * const.ELEVATION_FACTOR
            e22 = const.PLANET_RADIUS + self.land.get_elevation(map_w * 2 + 2, map_h * 2 + 2) * const.ELEVATION_FACTOR
            return e00, e01, e02, e10, e11, e12, e20, e21, e22
        e00 = const.PLANET_RADIUS + self.land.get_elevation(map_w, map_h) * const.ELEVATION_FACTOR
        e01 = const.PLANET_RADIUS + self.land.get_elevation(map_w, map_h + 1) * const.ELEVATION_FACTOR
        e10 = const.PLANET_RADIUS + self.land.get_elevation(map_w + 1, map_h) * const.ELEVATION_FACTOR
        e11 = const.PLANET_RADIUS + self.land.get_elevation(map_w + 1, map_h + 1) * const.ELEVATION_FACTOR
        return e00, e01, e10, e11

    def calculate_unpacked_elevation(self, h):
        """
        h is from 0 to 100
        result is from -water_level*underwater_factor to 100^height_exponent
        """
        if h > const.INPUT_WATER_LEVEL:
            return pow(h - 18, const.ELEVATION_UNPACK_EXPONENT)
        if h <= 0:
            return -1 * (const.INPUT_WATER_LEVEL - 1) * const.ELEVATION_UNPACK_UNDERWATER_FACTOR
        return (float(h - const.INPUT_WATER_LEVEL) / h) * float(const.ELEVATION_UNPACK_UNDERWATER_FACTOR)

    def calculate_elevation(self, w_i, h_i, shift_w, shift_h):
        if QUADRO_SEGMENTS:
            e00, e01, e02, e10, e11, e12, e20, e21, e22 = self.get_segment_elevation(w_i, h_i)
            e_min = min(e00, e01, e02, e10, e11, e12, e20, e21, e22)
            e_max = max(e00, e01, e02, e10, e11, e12, e20, e21, e22)
            a2 = const.SEGMENT_ANGLE
            a1 = a2 / 2.0
            w_f = shift_w * a2
            h_f = shift_h * a2
            tl_br = None
            if shift_w < 0.5:
                if shift_h < 0.5:
                    p00 = (0, 0, e00)
                    p01 = (0, a1, e01)
                    p10 = (a1, 0, e10)
                    p11 = (a1, a1, e11)
                    tl_br = True
                else:
                    p00 = (0, a1, e01)
                    p01 = (0, a2, e02)
                    p10 = (a1, a1, e11)
                    p11 = (a1, a2, e12)
                    tl_br = False
            else:
                if shift_h < 0.5:
                    p00 = (a1, 0, e10)
                    p01 = (a1, a1, e11)
                    p10 = (a2, 0, e20)
                    p11 = (a2, a1, e21)
                    tl_br = False
                else:
                    p00 = (a1, a1, e11)
                    p01 = (a1, a2, e12)
                    p10 = (a2, a1, e21)
                    p11 = (a2, a2, e22)
                    tl_br = True
            if mth.point_line_left_or_right(w_f, h_f, p01[0], p01[1], p10[0], p10[1]) == 1:
                e = mth.get_z_in_triangle(w_f, h_f, p00, p01, p10)
            else:
                e = mth.get_z_in_triangle(w_f, h_f, p01, p10, p11)
            return e, e_min, e_max
        e00, e01, e10, e11 = self.get_segment_elevation(w_i, h_i)
        e_min = min(e00, e01, e10, e11)
        e_max = max(e00, e01, e10, e11)
        a = const.SEGMENT_ANGLE
        p00 = (0, 0, e00)
        p01 = (0, a, e01)
        p10 = (a, 0, e10)
        p11 = (a, a, e11)
        w_f = shift_w * a
        h_f = shift_h * a
        if mth.point_line_left_or_right(w_f, h_f, p01[0], p01[1], p10[0], p10[1]) == 1:
            e = mth.get_z_in_triangle(w_f, h_f, p00, p01, p10)
        else:
            e = mth.get_z_in_triangle(w_f, h_f, p01, p10, p11)
        return e, e_min, e_max

    def calculate_land_vertices(self):        
        t1 = time.time()
        if QUADRO_SEGMENTS:
            for w in range(1, self.map_width-1):
                for h in range(1, self.map_height-1):
                    e00, e01, e02, e10, e11, e12, e20, e21, e22 = self.get_segment_elevation(w, h)
                    e_min = min(e00, e01, e02, e10, e11, e12, e20, e21, e22)
                    e_max = max(e00, e01, e02, e10, e11, e12, e20, e21, e22)
                    y00 = e00 * const.SEGMENT_COS
                    y01 = e01 * const.SEGMENT_COS
                    y02 = e02 * const.SEGMENT_COS
                    y10 = e10 * const.SEGMENT_COS
                    y11 = e11 * const.SEGMENT_COS
                    y12 = e12 * const.SEGMENT_COS
                    y20 = e20 * const.SEGMENT_COS
                    y21 = e21 * const.SEGMENT_COS
                    y22 = e22 * const.SEGMENT_COS
                    c00 = e00 * const.SEGMENT_SIN
                    c01 = e01 * const.SEGMENT_SIN
                    c02 = e02 * const.SEGMENT_SIN
                    c10 = e10 * const.SEGMENT_SIN
                    c11 = e11 * const.SEGMENT_SIN
                    c12 = e12 * const.SEGMENT_SIN
                    c20 = e20 * const.SEGMENT_SIN
                    c21 = e21 * const.SEGMENT_SIN
                    c22 = e22 * const.SEGMENT_SIN
                    v00 = ( c00 * const.PI_4_SIN,  y00,  -c00 * const.PI_4_COS)
                    v01 = ( c01 * const.PI_4_COS,  y01,   c01 * 0)
                    v02 = ( c02 * const.PI_4_COS,  y02,   c02 * const.PI_4_SIN)
                    v10 = ( c10 * 0,              y10,  -c10 * const.PI_4_COS)
                    v11 = ( c11 * 0,              y11,   c11 * 0)
                    v12 = ( c12 * 0,              y12,   c12 * const.PI_4_SIN)
                    v20 = (-c20 * const.PI_4_COS,  y20,  -c20 * const.PI_4_SIN)
                    v21 = (-c21 * const.PI_4_COS,  y21,   c21 * 0)
                    v22 = (-c22 * const.PI_4_SIN,  y22,   c22 * const.PI_4_COS)
                    self.land_vertices[(w, h)] = (v00, v01, v02, v10, v11, v12, v20, v21, v22, e_min, e_max)
            t2 = time.time()
            if _Debug:
                print(f'calculated {len(self.land_vertices)} land vertices segments in {t2 - t1} sec')
            return
        for w, h in self.land.elevation_map_data.keys():
            e00, e01, e10, e11 = self.get_segment_elevation(w, h)
            e_min = min(e00, e01, e10, e11)
            e_max = max(e00, e01, e10, e11)
            y00 = e00 * const.SEGMENT_COS
            y01 = e01 * const.SEGMENT_COS
            y10 = e10 * const.SEGMENT_COS
            y11 = e11 * const.SEGMENT_COS
            c00 = e00 * const.SEGMENT_SIN
            c01 = e01 * const.SEGMENT_SIN
            c10 = e10 * const.SEGMENT_SIN
            c11 = e11 * const.SEGMENT_SIN
            v00 = (c00 * const.PI_4_SIN, y00, -c00 * const.PI_4_COS)
            v01 = (c01 * const.PI_4_COS, y01, c01 * const.PI_4_SIN)
            v10 = (-c10 * const.PI_4_COS, y10, -c10 * const.PI_4_SIN)
            v11 = (-c11 * const.PI_4_SIN, y11, c11 * const.PI_4_COS)
            self.land_vertices[(w, h)] = (v00, v01, v10, v11, e_min, e_max)
        t2 = time.time()
        if _Debug:
            print(f'calculated land vertices for {len(self.land.elevation_map_data)} segments in {t2 - t1} sec')

    def calculate_scaled_elevation_map(self):
        return
        # e_min_unpacked = self.calculate_unpacked_elevation(0)
        # e_max_unpacked = self.calculate_unpacked_elevation(100)
        # unpacked_delta = e_max_unpacked - e_min_unpacked
        # for w in range(self.land.width):
        #     for h in range(self.land.height):
        #         e = self.land.elevation_map_data[(w, h)]
        #         e_unpacked = self.calculate_unpacked_elevation(e)
        #         e_scaled = float(e_unpacked - e_min_unpacked) / unpacked_delta
        #         self.land.elevation_map_data[(w, h)] = e_scaled

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
        if _Debug:
            print(f'  prepared mesh {name} with {idx} faces texture:{texture} coefs:{coefs}')
            # center:{mesh.center} min:{mesh.min} max:{mesh.max} radius:{mesh.radius} 
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
        unit = unt.Unit(name=object_name+'#'+str(_NextUnitID), object_name=object_name)
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
            unit.container_unit.add(PushMatrix(group=unit.name))
            unit.container_unit.add(mesh_transform.part_translate)
            unit.container_unit.add(PushMatrix(group=unit.name))
            unit.container_unit.add(mesh_transform.part_rotate)
            unit.container_unit.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=unit.name))
            unit.container_unit.add(Mesh(
                vertices=mesh.vertices,
                indices=mesh.indices,
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=unit.name,
            ))
            unit.container_unit.add(PopMatrix(group=unit.name))  # part_rotate
            unit.container_unit.add(PopMatrix(group=unit.name))  # part_translate

        # prepare unit container and transforms
        unit.container_unit = InstructionGroup(group=unit.name)
        unit.rotate_axis_x = Rotate(angle_coords[0], 1, 0, 0, group=unit.name)
        unit.rotate_axis_z = Rotate(angle_coords[1], 0, 0, 1, group=unit.name)
        unit.rotate_vertical = Rotate(direction + 90, 0, 1, 0, group=unit.name)
        unit.translate_shift = Translate(shift_vector[0], shift_vector[1], shift_vector[2], group=unit.name)
        unit.context_state = ChangeState(material_density=0.0, group=unit.name)
        # open unit context and push transforms and state
        unit.container_unit.add(PushMatrix(group=unit.name))  # unit
        unit.container_unit.add(unit.rotate_axis_x)
        unit.container_unit.add(unit.rotate_axis_z)
        unit.container_unit.add(PushMatrix(group=unit.name))  # unit shift
        unit.container_unit.add(unit.translate_shift)
        unit.container_unit.add(PushMatrix(group=unit.name))  # unit rotate
        unit.container_unit.add(unit.rotate_vertical)
        unit.container_unit.add(unit.context_state)
        # push unit meshes recursively
        source_object.walk_parts_ordered(_visitor)
        # close unit context
        unit.container_unit.add(ChangeState(material_density=0.0, group=unit.name))
        unit.container_unit.add(PopMatrix(group=unit.name))  # unit rotate
        unit.container_unit.add(PopMatrix(group=unit.name))  # unit shift
        unit.container_unit.add(PopMatrix(group=unit.name))  # unit
        # save unit
        self.units[unit.name] = unit
        if not static:
            self.animating_units.add(unit.name)
        # place unit on stage if needed
        if onstage:
            container.add(unit.container_unit)
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
        container.remove(unit.container_unit)
        unit.container_unit.clear()
        # container.remove_group(unit.name)
        for part_name in unit.meshes_transforms.keys():
            unit.meshes_transforms[part_name].part_rotate = None
            unit.meshes_transforms[part_name].part_translate = None
        unit.meshes_transforms.clear()
        unit.rotate_x_axis = None
        unit.rotate_z_axis = None
        if not unit.static:
            self.animating_units.discard(unit.name)
        self.units.pop(unit_name)
        # if _Debug:
        #     print(f'  removed unit {unit.name} from scene')

    def hide_unit(self, container, unit_name):
        if unit_name not in self.units:
            raise Exception(f'Unit {unit_name} is not on the stage at the moment')
        unit = self.units[unit_name]
        if not unit.onstage:
            raise Exception(f'Unit {unit_name} is already hidden')
        # container.remove_group(unit.name)
        container.remove(unit.container_unit)
        unit.onstage = False
        if _Debug:
            print(f'  unit {unit.name} was hidden')

    def show_unit(self, container, unit_name):
        if unit_name not in self.units:
            raise Exception(f'Unit {unit_name} is not on the stage at the moment')
        unit = self.units[unit_name]
        if unit.onstage:
            raise Exception(f'Unit {unit_name} is already visible')
        # source_object = self.static_objects[unit.object_name] if unit.static else self.animated_objects[unit.object_name]
        container.add(unit.container_unit)

        # def _visitor(part_name, parent_part_name):
        #     mesh_name = unit.meshes_names[part_name]
        #     mesh = self.meshes[mesh_name]
        #     mesh_transform = unit.meshes_transforms[part_name]
        #     container.add(PushMatrix(group=unit.name))
        #     container.add(mesh_transform.part_translate)
        #     container.add(PushMatrix(group=unit.name))
        #     container.add(mesh_transform.part_rotate)
        #     container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=unit.name))
        #     container.add(Mesh(
        #         vertices=mesh.vertices,
        #         indices=mesh.indices,
        #         fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
        #         mode='triangles',
        #         group=unit.name,
        #     ))
        #     container.add(PopMatrix(group=unit.name))  # part_rotate
        #     container.add(PopMatrix(group=unit.name))  # part_translate

        # # open unit context and prepare transforms and state
        # container.add(PushMatrix(group=unit.name))  # unit
        # container.add(unit.rotate_axis_x)
        # container.add(unit.rotate_axis_z)
        # container.add(PushMatrix(group=unit.name))  # unit shift
        # container.add(unit.translate_shift)
        # container.add(PushMatrix(group=unit.name))  # unit rotate
        # container.add(unit.rotate_vertical)
        # container.add(unit.context_state)
        # # push unit meshes
        # source_object.walk_parts_ordered(_visitor)
        # # close unit context
        # container.add(ChangeState(material_density=0.0, group=unit.name))
        # container.add(PopMatrix(group=unit.name))  # unit rotate
        # container.add(PopMatrix(group=unit.name))  # unit shift
        # container.add(PopMatrix(group=unit.name))  # unit
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
        if e < const.WATER_LEVEL_ELEVATION + (2.0 / 100.0) * const.ELEVATION_FACTOR:
            e = const.WATER_LEVEL_ELEVATION + (2.0 / 100.0) * const.ELEVATION_FACTOR
        # e = self.PLANET_RADIUS + 0.5 * self.ELEVATION_FACTOR
        # if _Debug:
        #     print(f'  map from {w0},{h0} shift:{w0shift},{h0shift} to {w_i},{h_i} with e:{e} new shift is {self.segment_shift_w},{self.segment_shift_h}')
        planet_shift_y = e + const.ELEVATION_CORRECTION # self.PLANET_RADIUS + e * self.ELEVATION_FACTOR
        self.global_translate_before.y = -planet_shift_y
        self.global_translate_after.y = planet_shift_y
        self.global_water_translate_before.y = -planet_shift_y
        self.global_water_translate_after.y = planet_shift_y
        camera_shift_angle_x, camera_shift_angle_z = self.coords_area2angles(0.5-self.segment_shift_w, 0.5-self.segment_shift_h)
        self.global_rotate_x.angle = camera_shift_angle_x
        self.global_rotate_z.angle = camera_shift_angle_z
        self.global_water_rotate_x.angle = camera_shift_angle_x
        self.global_water_rotate_z.angle = camera_shift_angle_z
        added = 0
        removed = 0
        if abs(wd) > 3 or abs(hd) > 3:
            if _Debug:
                print(f'  big shift for land update at {w_i} {h_i} with shift {self.segment_shift_w} {self.segment_shift_h} and delta {wd} {hd}')
            self.segments_cleanup_queue.clear()
        if new_position or wd != 0 or hd != 0:
            for unit_name in self.units.keys():
                unit = self.units[unit_name]
                unit.area_w = unit.w - w_i
                unit.area_h = unit.h - h_i
                segment_angle_x, segment_angle_z = self.coords_area2angles(unit.area_w, unit.area_h)
                unit.rotate_axis_x.angle = segment_angle_x
                unit.rotate_axis_z.angle = segment_angle_z
            for w_t, h_t in self.land_tiles_visible.keys():
                _w = w_t - w_i
                _h = h_t - h_i
                area_w, area_h, segment_rotate_x, segment_rotate_z, static_units_at_segment, _ = self.land_tiles_visible[(w_t, h_t)]
                area_w -= wd
                area_h -= hd
                segment_angle_x, segment_angle_z = self.coords_area2angles(area_w, area_h)
                segment_rotate_x.angle = segment_angle_x
                segment_rotate_z.angle = segment_angle_z
                self.land_tiles_visible[(w_t, h_t)][0] = area_w
                self.land_tiles_visible[(w_t, h_t)][1] = area_h
                # for unit_name in static_units_at_segment:
                #     unit = self.units[unit_name]
                #     if not unit.onstage:
                #         continue
                #     unit.area_w = area_w
                #     unit.area_h = area_h
                #     unit.rotate_axis_x.angle = segment_angle_x
                #     unit.rotate_axis_z.angle = segment_angle_z
                if (_w, _h) not in self.visible_area_mask:
                    if (w_t, h_t) not in self.segments_cleanup_queue:
                        self.segments_cleanup_queue.append((w_t, h_t))
                        removed += 1
            for w_t, h_t in self.water_tiles_visible.keys():
                _w = w_t - w_i
                _h = h_t - h_i
                water_area_w, water_area_h, water_segment_rotate_x, water_segment_rotate_z = self.water_tiles_visible[(w_t, h_t)]
                water_area_w -= wd
                water_area_h -= hd
                segment_angle_x, segment_angle_z = self.coords_area2angles(water_area_w, water_area_h)
                # water_segment_rotate_x.angle = segment_angle_x
                # water_segment_rotate_z.angle = segment_angle_z
                # self.water_tiles_visible[(w_t, h_t)][0] = water_area_w
                # self.water_tiles_visible[(w_t, h_t)][1] = water_area_h
            for unit_name in self.animating_units:
                unit = self.units[unit_name]
                if not unit.onstage:
                    continue
                _w = unit.w - w_i
                _h = unit.h - h_i
                if (_w, _h) not in self.visible_area_mask:
                    self.hide_unit(container=self.container_animated_objects, unit_name=unit.name)
            for k, dist_to_center in self.visible_area_mask.items():
                _w, _h = k
                w_t = w_i + _w
                h_t = h_i + _h
                if (w_t, h_t) not in self.land_tiles_visible:
                    if (w_t, h_t) not in self.segments_waiting:
                        self.segments_queue.append((w_t, h_t, dist_to_center))
                        self.segments_waiting.add((w_t, h_t))
                        added += 1
        if _Debug:
            print(f'visible area at {w_i} {h_i} with {added} added and {removed} removed segments, elevation is {e} / {planet_shift_y}, queue is {len(self.segments_waiting)} / {len(self.segments_cleanup_queue)}')
        self.update_segments()

    def update_segments(self, dt=None):
        if not self.segments_queue and not self.segments_cleanup_queue:
            return
        added = 0
        removed = 0
        self.segments_queue.sort(key=lambda x: -x[2])
        chunk = 30
        w_i = self.area_center_w
        h_i = self.area_center_h
        while chunk and self.segments_queue:
            chunk -= 1
            w_t, h_t, _ = self.segments_queue.pop()
            self.segments_waiting.discard((w_t, h_t))
            _w = w_t - w_i
            _h = h_t - h_i
            dist_to_center = self.visible_area_mask.get((_w, _h), None)
            if dist_to_center is not None:
                if (w_t, h_t) not in self.land_tiles_visible:
                    self.add_land_segment(w_t, h_t, _w, _h, dist_to_center)
                    added += 1
        if not added:
            for w_t, h_t in self.land_tiles_visible.keys():
                _w = w_t - w_i
                _h = h_t - h_i
                if (_w, _h) not in self.visible_area_mask:
                    if (w_t, w_t) not in self.segments_cleanup_queue:
                        self.segments_cleanup_queue.append((w_t, h_t))
        chunk = 100 if not added else int(added / 2)
        while chunk and self.segments_cleanup_queue:
            chunk -= 1
            w_t, h_t = self.segments_cleanup_queue.pop()
            if (w_t, h_t) in self.land_tiles_visible:
                _w = w_t - w_i
                _h = h_t - h_i
                if (_w, _h) not in self.visible_area_mask:
                    self.remove_land_segment(w_t, h_t)
                    removed += 1
        if added:
            Clock.schedule_once(self.update_segments, 0.2 * (1.0 / 120.0))
            if _Debug:
                print(f'  land updated at {w_i},{h_i} added:{added} removed:{removed} visible:{len(self.land_tiles_visible)}')
            return
        if removed:
            Clock.schedule_once(self.update_segments, 10.0)
            if _Debug:
                print(f'  land cleaned at {w_i},{h_i} added:{added} removed:{removed} visible:{len(self.land_tiles_visible)}')
            return

    def add_land_segment(self, map_w, map_h, area_w, area_h, dist_to_center):
        _get_texture = self.land.get_texture
        w_t = int(map_w)
        h_t = int(map_h)
        w = float(area_w)
        h = float(area_h)
        if QUADRO_SEGMENTS:
            v00, v01, v02, v10, v11, v12, v20, v21, v22, e_min, e_max = self.land_vertices[(w_t, h_t)]
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
        else:
            v00, v01, v10, v11, e_min, e_max = self.land_vertices[(w_t, h_t)]
            tex_file_path, tex_coord00, tex_coord01, tex_coord10, tex_coord11 = _get_texture(w_t, h_t)
            vert = [
                v00[0], v00[1], v00[2], 1, 0, 0, tex_coord00[0], tex_coord00[1],
                v01[0], v01[1], v01[2], 1, 0, 0, tex_coord01[0], tex_coord01[1],
                v10[0], v10[1], v10[2], 1, 0, 0, tex_coord10[0], tex_coord10[1],
                v11[0], v11[1], v11[2], 1, 0, 0, tex_coord11[0], tex_coord11[1],
            ]
        e_correction = 0
        # e_correction = (e_max - e_min) * 0.05
        segment_group_name = f'l_{map_w}_{map_h}'
        segment_angle_x, segment_angle_z = self.coords_area2angles(w, h)
        segment_rotate_x = Rotate(segment_angle_x, 1, 0, 0, group=segment_group_name)
        segment_rotate_z = Rotate(segment_angle_z, 0, 0, 1, group=segment_group_name)
        self.container_land_tiles.add(PushMatrix(group=segment_group_name))
        self.container_land_tiles.add(segment_rotate_x)
        self.container_land_tiles.add(segment_rotate_z)
        # if _Debug:
        #     if map_w == self.area_center_w and map_h == self.area_center_h:
        #         tex_source = None
        segment_state = ChangeState(material_density=0.0, group=segment_group_name)
        self.container_land_tiles.add(segment_state)
        if QUADRO_SEGMENTS:
            self.container_land_tiles.add(BindTexture(source=tex00_file_path, index=1, group=segment_group_name))
            self.container_land_tiles.add(Mesh(
                vertices=vert00,
                indices=[0, 1, 2, 1, 3, 2],
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=segment_group_name,
            ))
            self.container_land_tiles.add(BindTexture(source=tex01_file_path, index=1, group=segment_group_name))
            self.container_land_tiles.add(Mesh(
                vertices=vert01,
                indices=[0, 1, 3, 3, 2, 0],
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=segment_group_name,
            ))
            self.container_land_tiles.add(BindTexture(source=tex10_file_path, index=1, group=segment_group_name))
            self.container_land_tiles.add(Mesh(
                vertices=vert10,
                indices=[0, 1, 3, 3, 2, 0],
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
        else:
            self.container_land_tiles.add(BindTexture(source=tex_file_path, index=1, group=segment_group_name))
            self.container_land_tiles.add(Mesh(
                vertices=vert,
                indices=[0, 1, 2, 1, 3, 2],
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=segment_group_name,
            ))
        static_units_at_segment = []
        if QUADRO_SEGMENTS:
            plant_segments = [
                (w_t*2, h_t*2),
                (w_t*2, h_t*2+1),
                (w_t*2+1, h_t*2),
                (w_t*2+1, h_t*2+1),
            ]
        else:
            plant_segments = [
                (w_t, h_t),
            ]
        for wn, hn in plant_segments:
            plants_list = self.land.plants_map_data.get((wn, hn), [])
            for i in range(len(plants_list)):
                plant = plants_list[i]
                plant_variant = None
                static_object_name = None
                plant_key = plant['k']
                e_plant_correction = 0
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
                    e_plant_correction = so.root_mesh_center[0][2]
                else:
                    so = self.static_objects[static_object_name]
                    e_plant_correction = so.root_mesh_center[0][2]
                self.land.plants_map_data[(wn, hn)][i]['so'] = static_object_name
                shift_vector = self.coords_map2xyz(w_t, h_t, plant['sw'], plant['sh']) # , elevation_correction=e_correction+e_plant_correction)
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
        segment_group_name = f'l_{w_t}_{h_t}'
        _, _, _, _, static_units_at_segment, _ = self.land_tiles_visible[(w_t, h_t)]
        for static_unit_name in static_units_at_segment:
            self.remove_unit_from_stage(container=self.container_static_objects, unit_name=static_unit_name)
        self.container_land_tiles.remove_group(segment_group_name)
        # water_segment_group_name = f'w_{w_t}_{h_t}'
        # self.container_water_tiles.remove_group(water_segment_group_name)
        self.land_tiles_visible.pop((w_t, h_t))

    def land_shift(self, shift_w, shift_h):
        require_update = False
        if shift_h != 0:
            if shift_h > 0:
                if self.area_center_h + const.VISIBLE_AREA_SIZE_SEGMENTS_HALF + 1 < self.map_height:
                    self.segment_shift_h = self.segment_shift_h + shift_h
                    require_update = True
            else:
                if self.area_center_h - const.VISIBLE_AREA_SIZE_SEGMENTS_HALF > 0:
                    self.segment_shift_h = self.segment_shift_h + shift_h
                    require_update = True
        if shift_w != 0:
            if shift_w > 0:
                if self.area_center_w + const.VISIBLE_AREA_SIZE_SEGMENTS_HALF + 1 < self.map_width:
                    self.segment_shift_w = self.segment_shift_w + shift_w
                    require_update = True
            else:
                if self.area_center_w - const.VISIBLE_AREA_SIZE_SEGMENTS_HALF > 0:
                    self.segment_shift_w = self.segment_shift_w + shift_w
                    require_update = True
        if require_update:
            self.update_land()

    def land_move(self, direction_angle, distance):
        shift_w = math.cos(math.radians(direction_angle - 90)) * distance
        shift_h = math.sin(math.radians(direction_angle - 90)) * distance
        self.land_shift(shift_w, shift_h)

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
            e_correction = ao.root_mesh_center[0][2] * const.MODELS_SCALE_FACTOR
        segment_angle_x, segment_angle_z = self.coords_area2angles(area_w, area_h)
        shift_vector = self.coords_map2xyz(map_w, map_h, shift_w, shift_h) # , elevation_correction=e_correction)
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
        if _Debug:
            print(f'  new animated unit {unit.name} from template {template} at {map_w},{map_h} shift:{shift_w},{shift_h} direction:{direction} coefs:{coefs}')
        return unit

    def on_camera_rotate(self, camera_angle_y, camera_angle_z):
        return

    def on_run_units(self, delta):
        if self.renderer.camera_unit_lock:
            u = self.units.get(self.renderer.camera_unit_lock)
            if u:
                self.update_land(new_position=(u.w, u.h, u.shift_w, u.shift_h))
        for unit in self.units.values():
            if not unit.static:
                unit.run(self)

    def on_update_animations(self, delta):
        # TODO: maintain separate list of active animations for all units
        # then it is not required to loop all units
        for unit in self.units.values():
            if not unit.static and unit.onstage:
                unit.animate(self, delta)

