import os
import sys
import math


_Debug = True


from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix  # @UnresolvedImport
from kivy.graphics.opengl import glGetError, glEnable, glDisable, GL_DEPTH_TEST  # @UnresolvedImport
from kivy.graphics.instructions import InstructionGroup  # @UnresolvedImport
from kivy.graphics.context_instructions import Transform  # @UnresolvedImport
from kivy.graphics import (
    RenderContext, Callback, BindTexture,
    ChangeState, PushState, PopState,
    PushMatrix, PopMatrix, Scale,
    Color, Translate, Rotate, Mesh,
)

import mth
import dat


vertex_shader_src = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec3  v_normal;
attribute vec2  v_tex_coord;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec2 tex_coord0;
varying vec4 normal_vec;
varying vec4 vertex_pos;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos, 1.0);
    vertex_pos = pos;
    normal_vec = vec4(v_normal,0.0);
    gl_Position = projection_mat * pos;
    tex_coord0 = v_tex_coord;
}
"""

fragment_shader_src = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 tex_coord0;

uniform sampler2D texture_id;
uniform mat4 normal_mat;

void main (void) {
    gl_FragColor = texture2D(texture_id, tex_coord0);
}
"""


# fragment_shader_src = """
# #ifdef GL_ES
#     precision highp float;
# #endif
#
# varying vec4 normal_vec;
# varying vec4 vertex_pos;
# varying vec2 tex_coord0;
#
# uniform sampler2D texture_id;
# uniform mat4 normal_mat;
# uniform vec4 line_color;
#
# void main (void) {
#     gl_FragColor = texture2D(texture_id, tex_coord0) * line_color;
# }
# """


# vertex_shader_src = """
# #ifdef GL_ES
#     precision highp float;
# #endif
# attribute vec3  v_pos;
# attribute vec3  v_normal;
# attribute vec2  v_tex_coord;
# uniform mat4 modelview_mat;
# uniform mat4 projection_mat;
# uniform float Tr;
# varying vec2 tex_coord0;
# varying vec4 normal_vec;
# varying vec4 vertex_pos;
# void main (void) {
#     vec4 pos = modelview_mat * vec4(v_pos, 1.0);
#     vertex_pos = pos;
#     normal_vec = vec4(v_normal,0.0);
#     gl_Position = projection_mat * pos;
#     tex_coord0 = v_tex_coord;
# }
# """
# fragment_shader_src = """
# #ifdef GL_ES
#     precision highp float;
# #endif
# varying vec4 normal_vec;
# varying vec4 vertex_pos;
# varying vec2 tex_coord0;
# uniform sampler2D texture_id;
# uniform mat4 normal_mat;
# uniform vec4 line_color;
# uniform vec3 Kd;
# uniform vec3 Ka;
# uniform vec3 Ks;
# uniform float Tr;
# uniform float Ns;
# uniform float intensity;
# void main (void) {
#     vec4 v_normal = normalize( normal_mat * normal_vec );
#     vec4 v_light = normalize( vec4(0,0,0,1) - vertex_pos );
#     vec3 Ia = intensity * Kd;
#     vec3 Id = intensity * Ka * max(dot(v_light, v_normal), 0.0);
#     vec3 Is = intensity * Ks * pow(max(dot(v_light, v_normal), 0.0), Ns); 
#     gl_FragColor = vec4(Ia + Id + Is, Tr);
# }
# """


def ignore_undertouch(func):
    def wrap(self, touch):
        glst = touch.grab_list
        if len(glst) == 0 or (self is glst[ 0 ]()):
            return func(self, touch)
    return wrap


class Renderer(Widget):

    SCALE_FACTOR = 0.2
    SCALE_INITIAL = 1.0
    MAX_SCALE = 25.0
    MIN_SCALE = 0.2
    ROTATE_SPEED = 1.0
    ROTATE_VERTICAL_MIN = 1
    ROTATE_VERTICAL_MAX = 90
    ROTATE_VERTICAL_INITIAL = 25

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
    ELEVATION_FACTOR = PLANET_RADIUS / 10.0
    VISIBLE_AREA_SIZE_SEGMENTS = 36
    VISIBLE_AREA_SIZE_SEGMENTS_HALF = int(VISIBLE_AREA_SIZE_SEGMENTS / 2.0)
    LAND_MOVE_SPEED = 0.1

    def __init__(self, app_root, scene, **kwargs):
        self.app_root = app_root
        self.scene = scene
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.fs = fragment_shader_src
        self.canvas.shader.vs = vertex_shader_src
        self.container = None
        self.container_land = None
        self.container_land_tiles = None
        self.container_static_objects = None
        self.meshes_onstage = set()
        self.global_translate = None
        self.global_rotate_x = None
        self.global_rotate_y = None
        self.global_scale = None
        self.global_land_translate_before = None
        self.global_land_translate_after = None
        self.global_land_rotate_x = None
        self.global_land_rotate_y = None
        self.global_land_rotate_z = None
        self.map_center_w = None
        self.map_center_h = None
        self.map_width = None
        self.map_height = None
        self.area_center_w = None
        self.area_center_h = None
        self.segment_shift_w = None
        self.segment_shift_h = None
        self.land_area_left = 0
        self.land_area_top = 0
        self.land_tiles_visible = {}
        self.touches = []
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.on_setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.on_reset_gl_context)
        self.canvas['texture_id'] = 1
        self.keyboard_handler = Window.request_keyboard(self.on_keyboard_closed, self)
        self.keyboard_handler.bind(on_key_down=self.on_keyboard_down)
        Clock.schedule_interval(self.on_update_glsl, 1 / 60)
        Clock.schedule_interval(self.on_update_animations, 1 / 25)

    def setup_scene(self):
        if False:
            ChangeState(
                line_color=(1., 1., 1., 1.),
                Kd=(0.0, 1.0, 0.0),
                Ka=(1.0, 1.0, 0.0),
                Ks=(0.3, 0.3, 0.3),
                Tr=1.0,
                Ns=1.0,
                intensity=1.0,
            )
        PushMatrix()
        self.global_translate = Translate(0, -1, 0)
        self.global_rotate_x = Rotate(-self.ROTATE_VERTICAL_INITIAL, 1, 0, 0)
        self.global_rotate_y = Rotate(0, 0, 1, 0)
        self.global_scale = Scale(self.SCALE_INITIAL)
        sz = 1
        PushState()
        if False:
            Mesh(
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
            )
            ChangeState(line_color=(1., 0., 0., 1.))
            Mesh(
                vertices=[1 * sz, 0, 0, 0, 0, 0],
                indices=[0, 1],
                fmt=[(b'v_pos', 3, 'float'), ],
                mode='lines',
            )
            ChangeState(line_color=(0., 1., 0., 1.))
            Mesh(
                vertices=[0, 1 * sz, 0, 0, 0, 0],
                indices=[0, 1],
                fmt=[(b'v_pos', 3, 'float'), ],
                mode='lines',
            )
            ChangeState(line_color=(0., 0., 1., 1.))
            Mesh(
                vertices=[0, 0, 1 * sz, 0, 0, 0],
                indices=[0, 1],
                fmt=[(b'v_pos', 3, 'float'), ],
                mode='lines',
            )
            ChangeState(line_color=(1.,1.,1.,1.))
        PopState()
        Color(1, 1, 1)
        self.container = InstructionGroup()
        self.container_land = InstructionGroup()
        if False:
            self.container_land.add(ChangeState(
                line_color=(1., 1., 1., 1.),
                Kd=(0.0, 1.0, 0.0),
                Ka=(1.0, 1.0, 0.0),
                Ks=(0.3, 0.3, 0.3),
                Tr=1.0,
                Ns=1.0,
                intensity=1.0,
            ))
            self.container.add(ChangeState(
                line_color=(1., 1., 1., 1.),
                Kd=(0.0, 1.0, 0.0),
                Ka=(1.0, 1.0, 0.0),
                Ks=(0.3, 0.3, 0.3),
                Tr=1.0,
                Ns=1.0,
                intensity=1.0,
            ))
        PopMatrix()

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx / self.width) * 360.0 * self.ROTATE_SPEED
        y_angle = -1 * (touch.dy / self.height) * 360.0 * self.ROTATE_SPEED
        return x_angle, y_angle

    def prepare_land(self, map_center_w, map_center_h, map_width, map_height):
        self.global_land_rotate_x = Rotate(0, 1, 0, 0, group='land')
        self.global_land_rotate_y = Rotate(0, 0, 1, 0, group='land')
        self.global_land_rotate_z = Rotate(0, 0, 0, 1, group='land')
        self.map_center_w = map_center_w
        self.map_center_h = map_center_h
        self.map_width = map_width
        self.map_height = map_height
        self.area_center_w = int(map_center_w)
        self.area_center_h = int(map_center_h)
        self.segment_shift_w = 0.5
        self.segment_shift_h = 0.5
        w = int(self.area_center_w)
        h = int(self.area_center_h)
        self.land_area_left = w - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
        self.land_area_top  = h - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
        camera_shift_angle_z = mth.w2lat_degrees(-self.segment_shift_w+0.5, self.PLANET_EQUATOR_SEGMENTS)
        camera_shift_angle_x = mth.h2lon_degrees(-self.segment_shift_h+0.5, self.PLANET_EQUATOR_SEGMENTS)
        elevation_at_center = self.scene.land.get_elevation(w, h)
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
        self.container_land_tiles = InstructionGroup()
        self.container_static_objects = InstructionGroup()
        self.container_land.add(self.container_land_tiles)
        self.container_land.add(self.container_static_objects)
        self.container_land.add(self.global_land_translate_after)
        self.container_land.add(PopMatrix(group='land'))
        added = 0
        for _w in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
            for _h in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
                w_t = self.land_area_left + _w
                h_t = self.land_area_top + _h
                if (w_t, h_t) not in self.land_tiles_visible:
                    self.add_land_segment(w_t, h_t, _w, _h)
                    added += 1
        if _Debug:
            print(f'prepare land area at {w} {h} with {added} segments planet angle x:0 z:0')

    def calculate_elevation(self, w_i, h_i, shift_w, shift_h):
        _get_elevation = self.scene.land.get_elevation
        e00 = self.PLANET_RADIUS + _get_elevation(w_i, h_i) * self.ELEVATION_FACTOR
        e01 = self.PLANET_RADIUS + _get_elevation(w_i, h_i+1) * self.ELEVATION_FACTOR
        e10 = self.PLANET_RADIUS + _get_elevation(w_i+1, h_i) * self.ELEVATION_FACTOR
        e11 = self.PLANET_RADIUS + _get_elevation(w_i+1, h_i+1) * self.ELEVATION_FACTOR
        a = self.SEGMENT_ANGLE
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
        return e

    def update_land(self):
        w0shift = float(self.segment_shift_w)
        h0shift = float(self.segment_shift_h)
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
        e = self.calculate_elevation(w_i, h_i, self.segment_shift_w, self.segment_shift_h)
        # if _Debug:
        #     print(f'  map from {w0},{h0} shift:{w0shift},{h0shift} to {w_i},{h_i} with e:{e} new shift is {self.segment_shift_w},{self.segment_shift_h}')
        planet_shift_y = e # self.PLANET_RADIUS + e * self.ELEVATION_FACTOR
        self.global_land_translate_before.y = -planet_shift_y
        self.global_land_translate_after.y = planet_shift_y
        camera_shift_angle_z = mth.w2lat_degrees(-self.segment_shift_w+0.5, self.PLANET_EQUATOR_SEGMENTS)
        camera_shift_angle_x = mth.h2lon_degrees(-self.segment_shift_h+0.5, self.PLANET_EQUATOR_SEGMENTS)
        self.global_land_rotate_x.angle = camera_shift_angle_x
        self.global_land_rotate_z.angle = camera_shift_angle_z
        added = 0
        removed = 0
        if wd != 0 or hd != 0:
            new_area_left = w_i - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
            new_area_top = h_i - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF
            new_area_right = w_i + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF - 1
            new_area_bottom = h_i + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF - 1
            to_remove = []
            for k in self.land_tiles_visible.keys():
                w_t, h_t = k
                area_w, area_h, segment_rotate_x, segment_rotate_z, static_objects_at_segment = self.land_tiles_visible[(w_t, h_t)]
                if new_area_left <= w_t and w_t <= new_area_right and new_area_top <= h_t and h_t <= new_area_bottom:
                    area_w -= wd
                    area_h -= hd
                    segment_angle_z = mth.w2lat_degrees(float(area_w) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
                    segment_angle_x = mth.h2lon_degrees(float(area_h) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
                    segment_rotate_x.angle = segment_angle_x
                    segment_rotate_z.angle = segment_angle_z
                    self.land_tiles_visible[(w_t, h_t)][0] = area_w
                    self.land_tiles_visible[(w_t, h_t)][1] = area_h
                    for so_name, so_rotate_x, so_rotate_z in static_objects_at_segment:
                        so_rotate_x.angle = segment_angle_x
                        so_rotate_z.angle = segment_angle_z
                else:
                    to_remove.append((w_t, h_t, area_w, area_h))
            for w_t, h_t, area_w, area_h in to_remove:
                self.remove_land_segment(w_t, h_t)
                removed += 1
            for _w in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
                for _h in range(0, self.VISIBLE_AREA_SIZE_SEGMENTS):
                    w_t = new_area_left + _w
                    h_t = new_area_top + _h
                    if (w_t, h_t) not in self.land_tiles_visible:
                        self.add_land_segment(w_t, h_t, _w, _h)
                        added += 1
            self.land_area_left = new_area_left
            self.land_area_top = new_area_top
            if _Debug:
                print(f'        updated land area, moved by {wd},{hd} shift:{self.segment_shift_w},{self.segment_shift_h} segments added:{added} removed:{removed}')
        else:
            for k in self.land_tiles_visible.keys():
                w_t, h_t = k
                area_w, area_h, segment_rotate_x, segment_rotate_z, static_objects_at_segment = self.land_tiles_visible[(w_t, h_t)]
                segment_angle_z = mth.w2lat_degrees(float(area_w) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
                segment_angle_x = mth.h2lon_degrees(float(area_h) - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
                segment_rotate_x.angle = segment_angle_x
                segment_rotate_z.angle = segment_angle_z
                self.land_tiles_visible[(w_t, h_t)][0] = area_w
                self.land_tiles_visible[(w_t, h_t)][1] = area_h
                for so_name, so_rotate_x, so_rotate_z in static_objects_at_segment:
                    so_rotate_x.angle = segment_angle_x
                    so_rotate_z.angle = segment_angle_z

    def add_land_segment(self, map_w, map_h, area_w, area_h):
        _get_elevation = self.scene.land.get_elevation
        _get_texture = self.scene.land.get_texture
        w_t = int(map_w)
        h_t = int(map_h)
        w = float(area_w)
        h = float(area_h)
        e00 = self.PLANET_RADIUS + _get_elevation(w_t, h_t) * self.ELEVATION_FACTOR
        e01 = self.PLANET_RADIUS + _get_elevation(w_t, h_t + 1) * self.ELEVATION_FACTOR
        e10 = self.PLANET_RADIUS + _get_elevation(w_t + 1, h_t) * self.ELEVATION_FACTOR
        e11 = self.PLANET_RADIUS + _get_elevation(w_t + 1, h_t + 1) * self.ELEVATION_FACTOR
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
        segment_angle_z = mth.w2lat_degrees(w - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        segment_angle_x = mth.h2lon_degrees(h - float(self.VISIBLE_AREA_SIZE_SEGMENTS_HALF), self.PLANET_EQUATOR_SEGMENTS)
        segment_rotate_x.angle = segment_angle_x
        segment_rotate_z.angle = segment_angle_z
        self.container_land_tiles.add(PushMatrix(group=segment_group_name))
        self.container_land_tiles.add(segment_rotate_x)
        self.container_land_tiles.add(segment_rotate_z)
        if int(self.area_center_w) == int(map_w) and int(self.area_center_h) == int(map_h):
            if _Debug:
                print(f'     map segment at w:{map_w} h:{map_h} area_w:{area_w} area_h:{area_h} {e00} {e01} {e10} {e11}')
        #     self.container_land_tiles.add(BindTexture(source='./marker.png', index=1, group=segment_group_name))
        # else:
        self.container_land_tiles.add(BindTexture(source=tex_source, index=1, group=segment_group_name))
        self.container_land_tiles.add(Mesh(
            vertices=vert,
            indices=[0, 1, 2, 1, 2, 3],
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=segment_group_name,
        ))
        static_objects_at_segment = []
        if (map_w, map_h) in self.scene.land.plants_map_data:
            for i in range(len(self.scene.land.plants_map_data[(map_w, map_h)])):
                plant = self.scene.land.plants_map_data[(map_w, map_h)][i]
                if plant['m'] not in self.scene.models:
                    m = dat.ModelData()
                    m.unpack_figure_data('figures.res', 'models', template=plant['m'])
                    if not os.path.isfile('textures/model/' + plant['m'] + '.png'):
                        m.unpack_texture('textures.res', 'textures/model', plant['m'])
                    self.scene.add_model(plant['m'], m)
                if not plant.get('so'):
                    so = self.scene.create_static_object_from_model_data(
                        template=plant['m'],
                        coefs=plant['c'],
                        textures={'*': 'textures/model/' + plant['t'] + '.png', },
                    )
                    self.scene.land.plants_map_data[(map_w, map_h)][i]['so'] = so.name
                if True:
                    so_rotate_x, so_rotate_z = self.add_static_object(
                        name=self.scene.land.plants_map_data[(map_w, map_h)][i]['so'],
                        # container=self.container_land_tiles,
                        rotate_x=segment_angle_x,
                        rotate_z=segment_angle_z,
                        x=0,  # self.SEGMENT_SIZE * plant['sh'],
                        y=y00,
                        z=0,  # self.SEGMENT_SIZE * plant['sw'],
                    )
                    static_objects_at_segment.append((plant['so'], so_rotate_x, so_rotate_z))
        self.container_land_tiles.add(PopMatrix(group=segment_group_name))
        self.land_tiles_visible[(map_w, map_h)] = [area_w, area_h, segment_rotate_x, segment_rotate_z, static_objects_at_segment]

    def remove_land_segment(self, w_t, h_t):
        tile_group_name = f'land_{w_t}_{h_t}'
        _, _, _, _, static_objects_at_segment = self.land_tiles_visible[(w_t, h_t)]
        for so_name, _, _ in static_objects_at_segment:
            self.remove_static_object(so_name)
        self.container_land_tiles.remove_group(tile_group_name)
        self.land_tiles_visible.pop((w_t, h_t))

    def add_mesh(self, name):
        # NOT TO BE USED
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.meshes:
            raise Exception(f'Mesh {name} does not exist')
        mesh = self.scene.meshes.get(name)
        self.container.add(PushMatrix(group=mesh.name))
        self.container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=mesh.name))
        self.container.add(Mesh(
            vertices=mesh.vertices,
            indices=mesh.indices,
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=mesh.name,
        ))
        self.container.add(PopMatrix(group=mesh.name))
        self.meshes_onstage.add(name)
        mesh.onstage = True
        if _Debug:
            print(f'added mesh <{name}> on scene with {len(mesh.vertices)} vertices and {len(mesh.indices)} indices')

    def remove_mesh(self, name):
        # NOT TO BE USED
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.meshes:
            raise Exception(f'Mesh {name} does not exist')
        mesh = self.scene.meshes[name]
        self.container.remove_group(name)
        self.meshes_onstage.remove(name)
        mesh.onstage = False
        if _Debug:
            print(f'removed mesh <{name}> from scene')

    def add_unit(self, name):
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.units:
            raise Exception(f'Unit {name} does not exist')
        unit = self.scene.units[name]

        def _visitor(part_name, parent_part_name):
            mesh = unit.meshes.get(part_name)
            mesh.part_translate = Transform(group=mesh.name)
            mesh.part_rotate = Transform(group=mesh.name)
            self.container.add(PushMatrix(group=mesh.name))
            self.container.add(mesh.part_translate)
            self.container.add(PushMatrix(group=mesh.name))
            self.container.add(mesh.part_rotate)
            # TODO: check if we can pass texture data directly to Mesh instruction as a parameter
            self.container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=mesh.name))
            self.container.add(Mesh(
                vertices=mesh.vertices,
                indices=mesh.indices,
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=mesh.name,
                # texture=<already loaded Texture>,
            ))
            self.container.add(PopMatrix(group=mesh.name))  # part_rotate
            self.container.add(PopMatrix(group=mesh.name))  # part_translate
            self.meshes_onstage.add(mesh.name)
            mesh.onstage = True

        self.container.add(PushMatrix(group=unit.name))  # unit
        unit.walk_parts_ordered(_visitor)
        self.container.add(PopMatrix(group=unit.name))  # unit
        unit.onstage = True
        if _Debug:
            print(f'added unit ({unit.name}) on scene')

    def remove_unit(self, name):
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.units:
            raise Exception(f'Unit {name} does not exist')
        unit = self.scene.units[name]
        unit.onstage = False
        for mesh in unit.meshes.values():
            mesh.onstage = False
            self.container.remove_group(mesh.name)
            mesh.part_rotate = None
            mesh.part_translate = None
            self.meshes_onstage.remove(mesh.name)
        self.container.remove_group(unit.name)
        if _Debug:
            print(f'removed unit ({unit.name}) from scene')

    def add_static_object(self, name, rotate_x, rotate_z, x, y, z):
        if name not in self.scene.static_objects:
            raise Exception(f'Static object {name} does not exist')
        so = self.scene.static_objects[name]

        def _visitor(part_name, parent_part_name):
            mesh = so.meshes.get(part_name)
            mesh.part_translate = Transform(group=mesh.name)
            self.container_static_objects.add(PushMatrix(group=mesh.name))
            self.container_static_objects.add(mesh.part_translate)
            translate_mat = Matrix()
            translate_mat.translate(x, y, z)
            mesh.part_translate.matrix = translate_mat
            # TODO: check if we can pass texture data directly to Mesh instruction as a parameter
            self.container_static_objects.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=mesh.name))
            self.container_static_objects.add(Mesh(
                vertices=mesh.vertices,
                indices=mesh.indices,
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=mesh.name,
                # texture=<already loaded Texture>,
            ))
            self.container_static_objects.add(PopMatrix(group=mesh.name))  # part_translate
            self.meshes_onstage.add(mesh.name)
            mesh.onstage = True

        segment_rotate_x = Rotate(0, 1, 0, 0, group=so.name)
        segment_rotate_z = Rotate(0, 0, 0, 1, group=so.name)
        segment_rotate_x.angle = rotate_x
        segment_rotate_z.angle = rotate_z
        self.container_static_objects.add(PushMatrix(group=so.name))  # static object
        self.container_static_objects.add(segment_rotate_x)
        self.container_static_objects.add(segment_rotate_z)
        so.walk_parts_ordered(_visitor)
        self.container_static_objects.add(PopMatrix(group=so.name))  # static object
        so.onstage = True
        if _Debug:
            print(f'added static object ({so.name}) on scene')
        return segment_rotate_x, segment_rotate_z

    def remove_static_object(self, name):
        if name not in self.scene.static_objects:
            raise Exception(f'Static object {name} does not exist')
        so = self.scene.static_objects[name]
        so.onstage = False
        for mesh in so.meshes.values():
            mesh.onstage = False
            self.container_static_objects.remove_group(mesh.name)
            mesh.part_rotate = None
            mesh.part_translate = None
            self.meshes_onstage.remove(mesh.name)
        self.container_static_objects.remove_group(so.name)
        if _Debug:
            print(f'removed static object ({so.name}) from scene')

    def on_keyboard_closed(self):
        self.keyboard_handler.unbind(on_key_down=self.on_keyboard_down)
        self.keyboard_handler = None

    def on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
            App.get_running_app().stop()
        elif keycode[1] == 'z':
            for u in self.scene.units.values():
                if not u.onstage:
                    continue
                if not u.animations_loaded:
                    continue
                current_animation_ind = u.animations_loaded.index(u.animation_playing)
                current_animation_ind += 1
                if current_animation_ind >= len(u.animations_loaded):
                    current_animation_ind = 0
                u.animation_playing = u.animations_loaded[current_animation_ind]
                u.animation_frame = 0
                if _Debug:
                    print(f'playing animation {u.animation_playing} for ({u.name})')
                break
        elif keycode[1] == 'x':
            for u in self.scene.units.values():
                if not u.onstage:
                    continue
                if not u.animations_loaded:
                    continue
                current_animation_ind = u.animations_loaded.index(u.animation_playing)
                current_animation_ind -= 1
                if current_animation_ind < 0:
                    current_animation_ind = len(u.animations_loaded) - 1
                u.animation_playing = u.animations_loaded[current_animation_ind]
                u.animation_frame = 0
                if _Debug:
                    print(f'playing animation {u.animation_playing} for ({u.name})')
                break
        elif keycode[1] == 'c':
            units_onstage = []
            for unit in self.scene.units.values():
                if unit.onstage:
                    units_onstage.append(unit.name)
            for name in units_onstage:
                self.remove_unit(name)
            self.app_root.test_id += 1
            unit = self.app_root.prepare_test_unit(scene=self.scene, test=self.app_root.test_id)
            if unit and not unit.onstage:
                if unit.animations_loaded:
                    unit.animation_playing = unit.animations_loaded[0]
                self.add_unit(unit.name)
        elif keycode[1] == 'v':
            units_onstage = []
            for unit in self.scene.units.values():
                if unit.onstage:
                    units_onstage.append(unit.name)
            for name in units_onstage:
                self.remove_unit(name)
            self.app_root.test_id -= 1
            unit = self.app_root.prepare_test_unit(scene=self.scene, test=self.app_root.test_id)
            if unit and not unit.onstage:
                if unit.animations_loaded:
                    unit.animation_playing = unit.animations_loaded[0]
                self.add_unit(unit.name)
        elif keycode[1] == 'a':
            if self.area_center_h + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF + 1 < self.map_height:
                self.segment_shift_h = self.segment_shift_h + self.LAND_MOVE_SPEED
                self.update_land()
        elif keycode[1] == 'd':
            if self.area_center_h - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF > 0:
                self.segment_shift_h = self.segment_shift_h - self.LAND_MOVE_SPEED
                self.update_land()
        elif keycode[1] == 's':
            if self.area_center_w + self.VISIBLE_AREA_SIZE_SEGMENTS_HALF + 1 < self.map_width:
                self.segment_shift_w = self.segment_shift_w + self.LAND_MOVE_SPEED
                self.update_land()
        elif keycode[1] == 'w':
            if self.area_center_w - self.VISIBLE_AREA_SIZE_SEGMENTS_HALF > 0:
                self.segment_shift_w = self.segment_shift_w - self.LAND_MOVE_SPEED
                self.update_land()
        return True

    @ignore_undertouch
    def on_touch_down(self, touch):
        touch.grab(self)
        self.touches.append(touch)
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown'):
            if touch.button == "scrolldown":
                scale = 1.0 + self.SCALE_FACTOR
            if touch.button == "scrollup":
                scale = 1.0 - self.SCALE_FACTOR
            xyz = self.global_scale.xyz
            scale = xyz[0] * scale
            if scale < self.MAX_SCALE and scale > self.MIN_SCALE:
                self.global_scale.xyz = (scale, scale, scale)
                if _Debug:
                    print(f'new scale is {scale}')

    @ignore_undertouch
    def on_touch_up(self, touch):
        touch.ungrab(self)
        if touch in self.touches:
            self.touches.remove(touch)

    @ignore_undertouch
    def on_touch_move(self, touch):
        if touch in self.touches and touch.grab_current == self:
            if len(self.touches) == 1:
                ax, ay = self.define_rotate_angle(touch)
                new_global_rotate_x_angle = self.global_rotate_x.angle - ay
                if new_global_rotate_x_angle > -self.ROTATE_VERTICAL_MIN:
                    new_global_rotate_x_angle = -self.ROTATE_VERTICAL_MIN
                if new_global_rotate_x_angle < -self.ROTATE_VERTICAL_MAX:
                    new_global_rotate_x_angle = -self.ROTATE_VERTICAL_MAX
                self.global_rotate_y.angle -= ax
                self.global_rotate_x.angle = new_global_rotate_x_angle

    def on_setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def on_reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def on_gl_error(self, text='', kill=True):
        err = glGetError()
        if not err:
            return 
        while err:
            if _Debug:
                print('## GL ## = ' + text + 'OPENGL Error Code = ' + str(err))
            err = glGetError()
        if kill == True:
            sys.exit(0)

    def on_update_glsl(self, delta):
        asp = self.width / float(self.height)
        self.on_gl_error('step 1')
        self.canvas['texture_id'] = 1
        self.canvas['projection_mat'] = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['modelview_mat'] = Matrix().look_at(0, 0, -5, 0, 0, 0, 0, 1, 0)
        self.canvas['diffuse_light'] = (1.0, 1.0, 1.0)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.canvas['Kd'] = (0.0, 1.0, 0.0)
        self.canvas['Ka'] = (1.0, 1.0, 0.0)
        self.canvas['Ks'] = (0.3, 0.3, 0.3)
        self.canvas['Tr'] = 1.0
        self.canvas['Ns'] = 1.0
        self.canvas['intensity'] = 1.0
        self.canvas['line_color'] = (1., 1., 1., 1.)
        self.on_gl_error('step 2')

    def on_update_animations(self, delta):
        # TODO: maintain separate list of active animations for all units
        # then it is not required to loop all units
        for unit in self.scene.units.values():
            if not unit.onstage:
                continue
            if not unit.animation_playing:
                continue
            a = unit.animations[unit.animation_playing]
            root_part_name = unit.parts[0]
            root_part_animation = a.parts.get(root_part_name)
            if unit.animation_frame >= root_part_animation.frames:
                # if _Debug:
                #     print(f'restarting unit ({unit.name}) animation {unit.animation_playing} after frame {unit.animation_frame}')
                unit.animation_frame = 0
            f = unit.animation_frame
            for part_name in unit.parts:
                if part_name not in a.parts:
                    continue
                part_animation = a.parts.get(part_name)
                if not part_animation:
                    continue
                q = part_animation.rotation_frames[f]
                t = part_animation.translation_frames[f]
                mesh = unit.meshes[part_name]
                translate_mat = Matrix()
                translate_mat.translate(t[0], t[1], t[2])
                mesh.part_translate.matrix = translate_mat
                rotate_mat = Matrix()
                rotate_mat.set(array=mth.quaternion_to_matrix(q[0], q[1], q[2], q[3]))
                mesh.part_rotate.matrix = rotate_mat.inverse()
            unit.animation_frame += 1
