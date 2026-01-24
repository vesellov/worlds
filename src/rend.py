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
uniform vec4 line_color;

void main (void) {
    gl_FragColor = texture2D(texture_id, tex_coord0) * line_color;
}
"""


def ignore_undertouch(func):
    def wrap(self, touch):
        glst = touch.grab_list
        if len(glst) == 0 or (self is glst[ 0 ]()):
            return func(self, touch)
    return wrap


class Renderer(Widget):

    SCALE_FACTOR = 0.2
    SCALE_INITIAL = 1.0
    MAX_SCALE = 4.0
    MIN_SCALE = 0.25
    ROTATE_SPEED = 1.
    ROTATE_VERTICAL_MIN = 15
    ROTATE_VERTICAL_MAX = 90
    ROTATE_VERTICAL_INITIAL = 25
    PLANET_RADIUS = 200.0
    LAND_CELL_SCALE_FACTOR = 10.0
    ELEVATION_FACTOR = PLANET_RADIUS / 8.0
    LAND_AREA_SIZE_VISIBLE = 36
    LAND_AREA_HALF_SIZE_VISIBLE = int(LAND_AREA_SIZE_VISIBLE / 2)
    PLANET_LAND_SIZE = float(LAND_AREA_SIZE_VISIBLE * LAND_CELL_SCALE_FACTOR)
    LAND_MOVE_SPEED = 0.5

    def __init__(self, app_root, scene, **kwargs):
        self.app_root = app_root
        self.scene = scene
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.fs = fragment_shader_src
        self.canvas.shader.vs = vertex_shader_src
        self.container = None
        self.container_land = None
        self.container_land_tiles = None
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

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx / self.width) * 360.0 * self.ROTATE_SPEED
        y_angle = -1 * (touch.dy / self.height) * 360.0 * self.ROTATE_SPEED
        return x_angle, y_angle

    def add_land_tile(self, w, h, w_t, h_t):
        step = 1.0 / 8.0

        def _get_tile_texture_info(w_t, h_t, e_mid):
            ind = [0, 1, 2, 1, 2, 3]
            # water
            bit_x = 7
            bit_y = 4
            tex = 'zone19000.png'
            if e_mid > 0.001:
                # sand
                bit_x = 0
                bit_y = 0
                tex = 'zone19000.png'
            if e_mid > 0.5:
                # green grass
                bit_x = 7
                bit_y = 1
                tex = 'zone71002.png'            
            if e_mid > 0.75:
                # red soil
                bit_x = 5
                bit_y = 5
                tex = 'zone19000.png'            
            if e_mid > 0.75:
                # rocks
                bit_x = 0
                bit_y = 0
                tex = 'zone71002.png'            
            if e_mid > 0.9:
                # snow
                bit_x = 0
                bit_y = 0
                tex = 'bz10k000.png'
            if (w_t, h_t) == (70, 63):
                tex = 'textures/land/basegipat000.png'
                bit_x = 0
                bit_y = 6
            if (w_t, h_t) == (70, 62):
                tex = 'textures/land/basegipat000.png'
                bit_x = 7
                bit_y = 5

            if (w_t, h_t) == (72, 63):
                tex = 'textures/land/basegipat000.png'
                bit_x = 6
                bit_y = 0
            if (w_t, h_t) == (72, 62):
                tex = 'textures/land/basegipat007.png'
                bit_x = 0
                bit_y = 0
            if (w_t, h_t) == (71, 63):
                tex = 'textures/land/basegipat000.png'
                bit_x = 6
                bit_y = 0
            if (w_t, h_t) == (73, 62):
                tex = 'textures/land/basegipat000.png'
                bit_x = 6
                bit_y = 0
            if (w_t, h_t) == (73, 63):
                tex = 'textures/land/basegipat000.png'
                bit_x = 6
                bit_y = 0
            if (w_t, h_t) == (71, 62):
                tex = 'textures/land/basegipat005.png'
                bit_x = 1
                bit_y = 4
            corr = 0.001
            tc00 = (bit_x * step + corr, bit_y * step + corr)
            tc01 = (bit_x * step + corr, (bit_y + 1) * step - corr)
            tc10 = ((bit_x + 1) * step - corr, bit_y * step + corr)
            tc11 = ((bit_x + 1) * step - corr, (bit_y+1) * step - corr)
            return tex, ind, tc00, tc01, tc10, tc11

        half_f = float(self.LAND_AREA_HALF_SIZE_VISIBLE)
        _get_elevation = self.scene.land.get_elevation
        w_f = float(w)
        h_f = float(h)
        e00 = _get_elevation(w_t, h_t)
        e01 = _get_elevation(w_t, h_t + 1)
        e10 = _get_elevation(w_t + 1, h_t)
        e11 = _get_elevation(w_t + 1, h_t + 1)
        e_mid = (e00 + e01 + e10 + e11) / 4.0
        w00 = w_f - half_f
        h00 = h_f - half_f
        w01 = w_f - half_f
        h01 = h_f + 1.0 - half_f
        w10 = w_f + 1.0 - half_f
        h10 = h_f - half_f
        w11 = w_f + 1.0 - half_f
        h11 = h_f + 1.0 - half_f
        v00 = mth.wh2xyz(w00, h00, self.PLANET_LAND_SIZE, self.PLANET_LAND_SIZE, radius=self.PLANET_RADIUS + e00 * self.ELEVATION_FACTOR)
        v01 = mth.wh2xyz(w01, h01, self.PLANET_LAND_SIZE, self.PLANET_LAND_SIZE, radius=self.PLANET_RADIUS + e01 * self.ELEVATION_FACTOR)
        v10 = mth.wh2xyz(w10, h10, self.PLANET_LAND_SIZE, self.PLANET_LAND_SIZE, radius=self.PLANET_RADIUS + e10 * self.ELEVATION_FACTOR)
        v11 = mth.wh2xyz(w11, h11, self.PLANET_LAND_SIZE, self.PLANET_LAND_SIZE, radius=self.PLANET_RADIUS + e11 * self.ELEVATION_FACTOR)
        tex_source, indices, tex_coord00, tex_coord01, tex_coord10, tex_coord11 = _get_tile_texture_info(w_t, h_t, e_mid)
        vert = [
            # v00[0], v00[1], v00[2], 1, 0, 0, 0.2 * (w / width_f), 0.2 * (h / height_f),
            # v01[0], v01[1], v01[2], 1, 0, 0, 0.2 * (w / width_f) , 0.2 * (h + 1.0) / height_f,
            # v10[0], v10[1], v10[2], 1, 0, 0, 0.2 * (w + 1.0) / width_f, 0.2 * (h / height_f),
            # v11[0], v11[1], v11[2], 1, 0, 0, 0.2 * (w + 1.0) / width_f, 0.2 * (h + 1.0) / height_f,
            # v00[0], v00[1], v00[2], 1, 0, 0, bit_x * step, bit_y * step,
            # v01[0], v01[1], v01[2], 1, 0, 0, bit_x * step, (bit_y+1) * step,
            # v10[0], v10[1], v10[2], 1, 0, 0, (bit_x+1) * step, bit_y * step,
            # v11[0], v11[1], v11[2], 1, 0, 0, (bit_x+1) * step, (bit_y+1) * step,
            v00[0], v00[1], v00[2], 1, 0, 0, tex_coord00[0], tex_coord00[1],
            v01[0], v01[1], v01[2], 1, 0, 0, tex_coord01[0], tex_coord01[1],
            v10[0], v10[1], v10[2], 1, 0, 0, tex_coord10[0], tex_coord10[1],
            v11[0], v11[1], v11[2], 1, 0, 0, tex_coord11[0], tex_coord11[1],
        ]
        tile_group_name = f'land_{w_t}_{h_t}'
        self.container_land_tiles.add(BindTexture(source=tex_source, index=1, group=tile_group_name))
        self.container_land_tiles.add(Mesh(
            vertices=vert,
            indices=indices,
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=tile_group_name,
        ))
        self.land_tiles_visible[(w_t, h_t)] = (w, h)

    def remove_land_tile(self, w_t, h_t):
        tile_group_name = f'land_{w_t}_{h_t}'
        self.container_land_tiles.remove_group(tile_group_name)
        self.land_tiles_visible.pop((w_t, h_t))

    def prepare_land(self, initial_area_center_w, initial_area_center_h):
        self.global_land_rotate_x = Rotate(0, 1, 0, 0, group='land')
        self.global_land_rotate_y = Rotate(0, 0, 1, 0, group='land')
        self.global_land_rotate_z = Rotate(0, 0, 0, 1, group='land')
        # window_width = self.LAND_AREA_SIZE_VISIBLE
        # window_height = self.LAND_AREA_SIZE_VISIBLE
        self.initial_area_center_w = initial_area_center_w
        self.initial_area_center_h = initial_area_center_h
        latitude_radians = math.radians(-self.global_land_rotate_x.angle)
        longitude_radians = math.radians(90.0-self.global_land_rotate_z.angle)
        planet_width = self.PLANET_LAND_SIZE
        planet_height = self.PLANET_LAND_SIZE
        w_f = mth.lat2w(latitude_radians, width=planet_width)
        h_f = mth.lon2h(longitude_radians, height=planet_height)
        w_i = math.ceil(w_f) - 1
        h_i = math.ceil(h_f) - 1
        w = w_i + self.initial_area_center_w
        h = h_i + self.initial_area_center_h
        self.land_area_left = w - self.LAND_AREA_HALF_SIZE_VISIBLE
        self.land_area_top  = h - self.LAND_AREA_HALF_SIZE_VISIBLE
        planet_radius = self.PLANET_RADIUS
        elevation_factor = self.ELEVATION_FACTOR
        _get_elevation = self.scene.land.get_elevation
        elevation_at_center = _get_elevation(w, h)
        planed_shift_y = - planet_radius - elevation_at_center * elevation_factor
        planet_xyz = [0.0, planed_shift_y, 0.0]
        self.global_land_translate_before = Translate(planet_xyz[0], planet_xyz[1], planet_xyz[2], group='land')
        self.container_land.add(PushMatrix(group='land'))
        self.container_land.add(self.global_land_translate_before)
        self.container_land.add(self.global_land_rotate_x)
        self.container_land.add(self.global_land_rotate_y)
        self.container_land.add(self.global_land_rotate_z)
        self.container_land_tiles = InstructionGroup()
        self.container_land.add(self.container_land_tiles)
        self.global_land_translate_after = Translate(-planet_xyz[0], -planet_xyz[1], -planet_xyz[2], group='land')
        self.container_land.add(self.global_land_translate_after)
        self.container_land.add(PopMatrix(group='land'))
        if _Debug:
            print(f'prepare land area at {w} {h}')

    def update_land(self):
        window_width = self.LAND_AREA_SIZE_VISIBLE
        window_height = self.LAND_AREA_SIZE_VISIBLE
        planet_radius = self.PLANET_RADIUS
        elevation_factor = self.ELEVATION_FACTOR
        latitude_radians = math.radians(-self.global_land_rotate_x.angle)
        longitude_radians = math.radians(90.0-self.global_land_rotate_z.angle)
        planet_width = float(self.LAND_AREA_SIZE_VISIBLE * self.LAND_CELL_SCALE_FACTOR)
        planet_height = float(self.LAND_AREA_SIZE_VISIBLE * self.LAND_CELL_SCALE_FACTOR)
        w_f = mth.lat2w(latitude_radians, width=planet_width)
        h_f = mth.lon2h(longitude_radians, height=planet_height)
        w_i = math.ceil(w_f) - 1
        h_i = math.ceil(h_f) - 1
        w = w_i + self.initial_area_center_w
        h = h_i + self.initial_area_center_h
        _get_elevation = self.scene.land.get_elevation
        e00 = _get_elevation(w, h)
        e01 = _get_elevation(w, h + 1)
        e10 = _get_elevation(w + 1, h)
        e11 = _get_elevation(w + 1, h + 1)
        p00 = (float(w_i), float(h_i), e00)
        p01 = (float(w_i), float(h_i) + 1.0, e01)
        p10 = (float(w_i) + 1.0, float(h_i), e10)
        p11 = (float(w_i) + 1.0, float(h_i) + 1.0, e11)
        if mth.point_line_left_or_right(w_f, h_f, p01[0], p01[1], p10[0], p10[1]) == -1:
            e = mth.get_z_in_triangle(w_f, h_f, p00, p01, p10)
        else:
            e = mth.get_z_in_triangle(w_f, h_f, p01, p10, p11)
        planed_shift_y = - planet_radius - e * elevation_factor
        self.global_land_translate_before.y = planed_shift_y
        new_area_center_w = w
        new_area_center_h = h
        new_area_left = w - self.LAND_AREA_HALF_SIZE_VISIBLE
        new_area_top = h - self.LAND_AREA_HALF_SIZE_VISIBLE
        new_area_right = w + self.LAND_AREA_HALF_SIZE_VISIBLE
        new_area_bottom = h + self.LAND_AREA_HALF_SIZE_VISIBLE
        added = 0
        removed = 0
        if self.land_area_left != new_area_left or self.land_area_top != new_area_top or not self.land_tiles_visible:
            w_delta = new_area_center_w - self.initial_area_center_w
            h_delta = new_area_center_h - self.initial_area_center_h
            for _w in range(0, window_width - 1):
                for _h in range(0, window_height - 1):
                    w_t = new_area_left + _w
                    h_t = new_area_top + _h
                    if (w_t, h_t) not in self.land_tiles_visible:
                        self.add_land_tile(_w + w_delta, _h + h_delta, w_t, h_t)
                        added += 1
            to_remove = []
            for k, v in self.land_tiles_visible.items():
                w_t, h_t = k
                _w, _h = v
                if new_area_left - 2 <= w_t and w_t <= new_area_right + 2 and new_area_top - 2 <= h_t and h_t <= new_area_bottom + 2:
                    continue
                to_remove.append((w_t, h_t))
            for w_t, h_t in to_remove:
                self.remove_land_tile(w_t, h_t)
                removed += 1
        self.land_area_left = new_area_left
        self.land_area_top = new_area_top
        if added or removed:
            if _Debug:
                print(f'    updated land area at {w} {h}, added {added} and removed {removed} tiles, lat:{self.global_land_rotate_x.angle} lon:{self.global_land_rotate_z.angle}')

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
            self.container.add(PushMatrix(group=mesh.name))
            mesh.part_translate = Transform(group=mesh.name)
            self.container.add(mesh.part_translate)
            self.container.add(PushMatrix(group=mesh.name))
            mesh.part_rotate = Transform(group=mesh.name)
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
            print(f'removed mesh unit ({unit.name}) from scene')

    def setup_scene(self):
        PushMatrix()
        self.global_translate = Translate(0, -1, 0)
        self.global_rotate_x = Rotate(-self.ROTATE_VERTICAL_INITIAL, 1, 0, 0)
        self.global_rotate_y = Rotate(0, 0, 1, 0)
        self.global_scale = Scale(self.SCALE_INITIAL)
        sz = 1
        PushState()
        ChangeState(line_color=(0.5, 0.5, 0.5, 1.))
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
        self.container_land = InstructionGroup()
        self.container = InstructionGroup()
        PopMatrix()

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
            if self.app_root.test_id > 3:
                self.app_root.test_id = 1
            unit = self.app_root.prepare_test_unit(scene=self.scene, test=self.app_root.test_id)
            if unit and not unit.onstage:
                self.add_unit(name)
        elif keycode[1] == 'v':
            units_onstage = []
            for unit in self.scene.units.values():
                if unit.onstage:
                    units_onstage.append(unit.name)
            for name in units_onstage:
                self.remove_unit(name)
            self.app_root.test_id -= 1
            if self.app_root.test_id == 0:
                self.app_root.test_id = 3
            unit = self.app_root.prepare_test_unit(scene=self.scene, test=self.app_root.test_id)
            if unit and not unit.onstage:
                self.add_unit(name)
        elif keycode[1] == 'd':
            new_global_land_rotate_z_angle = self.global_land_rotate_z.angle - self.LAND_MOVE_SPEED
            if new_global_land_rotate_z_angle < -180.0:
                new_global_land_rotate_z_angle += 360.0
            self.global_land_rotate_z.angle = new_global_land_rotate_z_angle
            self.update_land()
        elif keycode[1] == 'a':
            new_global_land_rotate_z_angle = self.global_land_rotate_z.angle + self.LAND_MOVE_SPEED
            if new_global_land_rotate_z_angle > 180.0:
                new_global_land_rotate_z_angle -= 360.0
            self.global_land_rotate_z.angle = new_global_land_rotate_z_angle
            self.update_land()
        elif keycode[1] == 'w':
            new_global_land_rotate_x_angle = self.global_land_rotate_x.angle - self.LAND_MOVE_SPEED
            if new_global_land_rotate_x_angle < -180.0:
                new_global_land_rotate_x_angle += 360.0
            self.global_land_rotate_x.angle = new_global_land_rotate_x_angle
            self.update_land()
        elif keycode[1] == 's':
            new_global_land_rotate_x_angle = self.global_land_rotate_x.angle + self.LAND_MOVE_SPEED
            if new_global_land_rotate_x_angle > 180.0:
                new_global_land_rotate_x_angle -= 360.0
            self.global_land_rotate_x.angle = new_global_land_rotate_x_angle
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
