import sys


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
    Color, Translate, Rotate, Mesh, UpdateNormalMatrix,
)

import dat
import mth
import res


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

    SCALE_FACTOR = 0.05
    MAX_SCALE = 10.0
    MIN_SCALE = 0.1
    ROTATE_SPEED = 1.

    def __init__(self, app_root, scene, **kwargs):
        self.app_root = app_root
        self.scene = scene
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.fs = fragment_shader_src
        self.canvas.shader.vs = vertex_shader_src
        self.container = None
        self.container_land = None
        self.meshes_onstage = set()
        self.touches = []
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
        self.canvas['texture_id'] = 1
        Clock.schedule_interval(self.update_glsl, 1 / 60)
        Clock.schedule_interval(self.update_animations, 1 / 25)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
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
        return True

    @ignore_undertouch
    def on_touch_down(self, touch):
        touch.grab(self)
        self.touches.append(touch)
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown'):
            if touch.button == "scrolldown":
                scale = self.SCALE_FACTOR
            if touch.button == "scrollup":
                scale = -self.SCALE_FACTOR
            xyz = self.global_scale.xyz
            scale = xyz[0] + scale
            if scale < self.MAX_SCALE and scale > self.MIN_SCALE:
                self.global_scale.xyz = (scale, scale, scale)

    @ignore_undertouch
    def on_touch_up(self, touch):
        touch.ungrab(self)
        if touch in self.touches:
            self.touches.remove(touch)

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx / self.width) * 360.0 * self.ROTATE_SPEED
        y_angle = -1 * (touch.dy / self.height) * 360.0 * self.ROTATE_SPEED
        return x_angle, y_angle

    @ignore_undertouch
    def on_touch_move(self, touch):
        if touch in self.touches and touch.grab_current == self:
            if len(self.touches) == 1:
                ax, ay = self.define_rotate_angle(touch)
                self.global_rotate_y.angle -= ax
                self.global_rotate_x.angle -= ay

    def add_land(self):
        window_w = 20
        window_h = 64
        window_width = 32
        window_height = 32
        window_center_w = window_w + int(window_width / 2)
        window_center_h = window_h + int(window_height / 2)
        planet_radius = 150.0
        elevation_factor = planet_radius / 5.0
        cells_scale_factor = 20
        width = window_width
        height = window_height
        # width = self.land.width
        # height = self.land.height
        width_half = int(width / 2.0)
        height_half = int(height / 2.0)
        _get_elevation = self.scene.land.get_elevation
        elevation_at_center = _get_elevation(window_center_w, window_center_h)
        # planed_shift_y = - planet_radius - elevation_at_00 * elevation_factor - elevation_factor / 12 - 1.0
        planed_shift_y = - planet_radius - elevation_at_center * elevation_factor
        planet_xyz = [0.0, planed_shift_y, 0.0]
        w2f = float(width / 2)
        h2f = float(height / 2)
        width_f = float(width)
        height_f = float(height)
        planet_width = float(width * cells_scale_factor)
        planet_height = float(height * cells_scale_factor)
        self.container_land.add(PushMatrix(group='land'))
        self.container_land.add(Translate(planet_xyz[0], planet_xyz[1], planet_xyz[2]))
        # self.container_land.add(PushMatrix(group='land'))
        for _w in range(0, window_width - 1):
            for _h in range(0, window_height - 1):
                w = _w
                h = _h
        # for _w in range(0, width - 1):
        #     for _h in range(0, height - 1):
                # w = _w
                # h = _h
                w_f = float(_w)
                h_f = float(_h)
                e00 = _get_elevation(window_w + w, window_h + h)
                e01 = _get_elevation(window_w + w, window_h + h + 1)
                e10 = _get_elevation(window_w + w + 1, window_h + h)
                e11 = _get_elevation(window_w + w + 1, window_h + h + 1)
                # e00 = self.land.get_elevation(w, h)
                # e01 = self.land.get_elevation(w, h+1)
                # e10 = self.land.get_elevation(w+1, h)
                # e11 = self.land.get_elevation(w+1, h+1)
                if 0 in (e00, e01, e10, e11):
                    # TODO: pain water
                    continue
                w00 = w_f - w2f
                h00 = h_f - h2f
                w01 = w_f - w2f
                h01 = h_f + 1.0 - h2f
                w10 = w_f + 1.0 - w2f
                h10 = h_f - h2f
                w11 = w_f + 1.0 - w2f
                h11 = h_f + 1.0 - h2f
                v00 = mth.wh2xyz(w00, h00, planet_width, planet_height, radius=planet_radius+e00*elevation_factor)
                v01 = mth.wh2xyz(w01, h01, planet_width, planet_height, radius=planet_radius+e01*elevation_factor)
                v10 = mth.wh2xyz(w10, h10, planet_width, planet_height, radius=planet_radius+e10*elevation_factor)
                v11 = mth.wh2xyz(w11, h11, planet_width, planet_height, radius=planet_radius+e11*elevation_factor)
                vert = [
                    v00[0], v00[1], v00[2], 1, 0, 0, w / width_f, h / height_f,
                    v01[0], v01[1], v01[2], 1, 0, 0, w / width_f, (h + 1.0) / height_f,
                    v10[0], v10[1], v10[2], 1, 0, 0, (w + 1.0) / width_f, h / height_f,
                    v11[0], v11[1], v11[2], 1, 0, 0, (w + 1.0) / width_f, (h + 1.0) / height_f,
                ]
                ind = [0, 1, 2, 1, 2, 3]
                self.container_land.add(BindTexture(source='land.png', index=1, group='land'))
                self.container_land.add(Mesh(
                    vertices=vert,
                    indices=ind,
                    fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                    mode='triangles',
                    group='land',
                ))
        # self.container_land.add(PopMatrix(group='land'))
        self.container_land.add(Translate(-planet_xyz[0], -planet_xyz[1], -planet_xyz[2]))
        self.container_land.add(PopMatrix(group='land'))

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

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def gl_error(self, text='', kill=True):
        err = glGetError()
        if not err:
            return 
        while err:
            if _Debug:
                print('## GL ## = ' + text + 'OPENGL Error Code = ' + str(err))
            err = glGetError()
        if kill == True:
            sys.exit(0)

    def update_glsl(self, delta):
        asp = self.width / float(self.height)
        self.gl_error('step 1')
        self.canvas['texture_id'] = 1
        self.canvas['projection_mat'] = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['modelview_mat'] = Matrix().look_at(0, 0, -5, 0, 0, 0, 0, 1, 0)
        self.canvas['diffuse_light'] = (1.0, 1.0, 1.0)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.gl_error('step 2')

    def update_animations(self, delta):
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
                if _Debug:
                    print(f'restarting unit ({unit.name}) animation {unit.animation_playing} after frame {unit.animation_frame}')
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

    def setup_scene(self):
        PushMatrix()
        self.global_translate = Translate(0, 0, 0)
        self.global_rotate_x = Rotate(0, 1, 0, 0)
        self.global_rotate_y = Rotate(0, 0, 1, 0)
        self.global_scale = Scale(0.5)
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
