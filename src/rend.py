import os
import sys
import math
import random

_Debug = True


from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix  # @UnresolvedImport
from kivy.graphics.opengl import (
    glGetError, glEnable, glDisable, GL_BLEND, GL_DEPTH_TEST,
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glDepthFunc, GL_LEQUAL,
)
from kivy.graphics.instructions import InstructionGroup  # @UnresolvedImport
from kivy.graphics import (
    RenderContext, Callback, BindTexture,
    ChangeState, PushState, PopState,
    PushMatrix, PopMatrix,
    # Scale,
    Color, Translate, Rotate, Mesh,
    # UpdateNormalMatrix,
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
    normal_vec = vec4(v_normal, 0.0);
    gl_Position = projection_mat * pos;
    tex_coord0 = v_tex_coord;
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
#
# void main (void) {
#     gl_FragColor = texture2D(texture_id, tex_coord0);
# }
# """


fragment_shader_src = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 tex_coord0;

uniform sampler2D texture_id;
uniform mat4 normal_mat;
uniform float brightness;
uniform float contrast;

void main (void) {
    vec4 color = texture2D(texture_id, tex_coord0).rgba;
    vec3 new_color = (color.rgb - 0.5) * contrast + 0.5 + brightness;
    gl_FragColor = vec4(new_color, color.a);
}
"""


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

    CAMERA_DISTANCE_TO_CENTER_INITIAL = 5.0
    CAMERA_VIEW_CLIP_NEAR = 1.0
    CAMERA_VIEW_CLIP_FAR = 200.0

    SCALE_INITIAL = 2.0
    SCALE_MIN = 0.5
    SCALE_MAX = 12.0
    SCALE_SPEED_FACTOR = 0.2

    ROTATE_SPEED = 1.0
    ROTATE_VERTICAL_MIN = 1
    ROTATE_VERTICAL_MAX = 79
    ROTATE_VERTICAL_INITIAL = 45

    def __init__(self, app_root, scene, **kwargs):
        self.app_root = app_root
        self.scene = scene
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.fs = fragment_shader_src
        self.canvas.shader.vs = vertex_shader_src
        self.camera_distance_scale_factor = self.SCALE_INITIAL
        self.camera_distance_to_center = self.CAMERA_DISTANCE_TO_CENTER_INITIAL
        self.camera_angle_y = float(self.ROTATE_VERTICAL_INITIAL)
        self.camera_angle_z = 180.0
        self.camera_unit_lock = None
        self.global_eye_x = 0
        self.global_eye_y = 0
        self.global_eye_z = 0
        self.global_center_x = 0
        self.global_center_y = 0
        self.global_center_z = 0
        self.touches = []
        self.brightness = 0.0
        self.contrast = 1.0
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.on_setup_gl_context)
            PushMatrix()
            # UpdateNormalMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.on_reset_gl_context)
        self.canvas['texture_id'] = 1
        self.keyboard_handler = Window.request_keyboard(self.on_keyboard_closed, self)
        self.keyboard_handler.bind(on_key_down=self.on_keyboard_down)
        Clock.schedule_interval(self.on_update_glsl, 1 / 60)
        Clock.schedule_interval(self.scene.on_update_animations, 1 / 25)
        Clock.schedule_interval(self.scene.on_run_units, 1 / 60)

    def setup_scene(self):
        # if False:
        #     ChangeState(
        #         line_color=(1., 1., 1., 1.),
        #         Kd=(0.0, 1.0, 0.0),
        #         Ka=(1.0, 1.0, 0.0),
        #         Ks=(0.3, 0.3, 0.3),
        #         Tr=1.0,
        #         Ns=1.0,
        #         intensity=1.0,
        #     )
        PushMatrix()
        sz = 1
        # if False:
        #     PushState()
        #     Mesh(
        #         vertices=[
        #             -1 * sz, -1 * sz, -1 * sz,
        #             -1 * sz, -1 * sz, 1 * sz,
        #             -1 * sz, 1 * sz, 1 * sz,
        #             -1 * sz, 1 * sz, -1 * sz,
        #             1 * sz, -1 * sz, -1 * sz,
        #             1 * sz, -1 * sz, 1 * sz,
        #             1 * sz, 1 * sz, 1 * sz,
        #             1 * sz, 1 * sz, -1 * sz,
        #         ],
        #         indices=[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7],
        #         fmt=[(b'v_pos', 3, 'float'), ],
        #         mode='lines',
        #     )
        #     ChangeState(line_color=(1., 0., 0., 1.))
        #     Mesh(
        #         vertices=[1 * sz, 0, 0, 0, 0, 0],
        #         indices=[0, 1],
        #         fmt=[(b'v_pos', 3, 'float'), ],
        #         mode='lines',
        #     )
        #     ChangeState(line_color=(0., 1., 0., 1.))
        #     Mesh(
        #         vertices=[0, 1 * sz, 0, 0, 0, 0],
        #         indices=[0, 1],
        #         fmt=[(b'v_pos', 3, 'float'), ],
        #         mode='lines',
        #     )
        #     ChangeState(line_color=(0., 0., 1., 1.))
        #     Mesh(
        #         vertices=[0, 0, 1 * sz, 0, 0, 0],
        #         indices=[0, 1],
        #         fmt=[(b'v_pos', 3, 'float'), ],
        #         mode='lines',
        #     )
        #     ChangeState(line_color=(1., 1., 1., 1.))
        #     PopState()
        # if False:
        #     Color(1, 1, 1)
        # SCENE BEGIN
        self.scene.create_containers()
        # SCENE END
        # if False:
        #     self.scene.container_land.add(ChangeState(
        #         line_color=(1., 1., 1., 1.),
        #         Kd=(0.0, 1.0, 0.0),
        #         Ka=(1.0, 1.0, 0.0),
        #         Ks=(0.3, 0.3, 0.3),
        #         Tr=1.0,
        #         Ns=1.0,
        #         intensity=1.0,
        #     ))
        #     self.scene.container.add(ChangeState(
        #         line_color=(1., 1., 1., 1.),
        #         Kd=(0.0, 1.0, 0.0),
        #         Ka=(1.0, 1.0, 0.0),
        #         Ks=(0.3, 0.3, 0.3),
        #         Tr=1.0,
        #         Ns=1.0,
        #         intensity=1.0,
        #     ))
        PopMatrix()

    def setup_canvas(self):
        asp = self.width / float(self.height)
        # self.on_gl_error('step 1')
        # self.canvas['texture_id'] = 1
        self.global_eye_x = float(self.camera_distance_scale_factor) * self.camera_distance_to_center * math.sin(math.radians(self.camera_angle_y)) * math.sin(math.radians(self.camera_angle_z))
        self.global_eye_y = float(self.camera_distance_scale_factor) * self.camera_distance_to_center * math.cos(math.radians(self.camera_angle_y))
        self.global_eye_z = float(self.camera_distance_scale_factor) * self.camera_distance_to_center * math.sin(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_z))
        self.canvas['projection_mat'] = Matrix().view_clip(-asp, asp, -1, 1, self.CAMERA_VIEW_CLIP_NEAR, self.CAMERA_VIEW_CLIP_FAR, 1)
        self.canvas['modelview_mat'] = Matrix().look_at(
            self.global_eye_x, self.global_eye_y, self.global_eye_z,
            self.global_center_x, self.global_center_y, self.global_center_z,
            0, 1, 0,  # up vector
        )
        # self.canvas['diffuse_light'] = (1.0, 1.0, 1.0)
        # self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        # self.canvas['Kd'] = (0.0, 1.0, 0.0)
        # self.canvas['Ka'] = (1.0, 1.0, 0.0)
        # self.canvas['Ks'] = (0.3, 0.3, 0.3)
        # self.canvas['Tr'] = 1.0
        # self.canvas['Ns'] = 1.0
        # self.canvas['intensity'] = 1.0
        # self.canvas['line_color'] = (1.0, 1.0, 1.0, 1.0)
        self.canvas['brightness'] = self.brightness
        self.canvas['contrast'] = self.contrast
        # self.on_gl_error('step 2')

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx / self.width) * 360.0 * self.ROTATE_SPEED
        y_angle = -1 * (touch.dy / self.height) * 360.0 * self.ROTATE_SPEED
        return x_angle, y_angle


    def on_keyboard_closed(self):
        self.keyboard_handler.unbind(on_key_down=self.on_keyboard_down)
        self.keyboard_handler = None

    def on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
            App.get_running_app().stop()
        # elif keycode[1] == 't':
        #     self.camera_distance_to_center += 1.0
        # elif keycode[1] == 'y':
        #     self.camera_distance_to_center -= 1.0
        # elif keycode[1] == 'l':
        #     self.camera_angle_z += 1.0
        # elif keycode[1] == 'j':
        #     self.camera_angle_z -= 1.0
        # elif keycode[1] == 'i':
        #     self.camera_angle_y += 1.0
        # elif keycode[1] == 'k':
        #     self.camera_angle_y -= 1.0
        elif keycode[1] == 'u':
            self.contrast += 0.1
        elif keycode[1] == 'i':
            self.contrast -= 0.1
        elif keycode[1] == 'o':
            self.brightness += 0.1
        elif keycode[1] == 'p':
            self.brightness -= 0.1
        # elif keycode[1] == 'f':
        #     self.global_center_x += 1.0
        # elif keycode[1] == 'g':
        #     self.global_center_x -= 1.0
        # elif keycode[1] == 'h':
        #     self.global_center_y += 1.0
        # elif keycode[1] == 'j':
        #     self.global_center_y -= 1.0
        # elif keycode[1] == 'k':
        #     self.global_center_z += 1.0
        # elif keycode[1] == 'l':
        #     self.global_center_z -= 1.0
        elif keycode[1] == 'z':
            for unit in self.scene.units.values():
                if not unit.animations_list:
                    continue
                current_animation_ind = unit.animations_list.index(unit.animation_playing)
                current_animation_ind += 1
                if current_animation_ind >= len(unit.animations_list):
                    current_animation_ind = 0
                unit.animation_playing = unit.animations_list[current_animation_ind]
                unit.animation_frame = 0
                if _Debug:
                    print(f'playing next animation {unit.animation_playing} for unit {unit.name}')
        elif keycode[1] == 'x':
            for unit in self.scene.units.values():
                if not unit.animations_list:
                    continue
                current_animation_ind = unit.animations_list.index(unit.animation_playing)
                current_animation_ind -= 1
                if current_animation_ind < 0:
                    current_animation_ind = len(unit.animations_list) - 1
                unit.animation_playing = unit.animations_list[current_animation_ind]
                unit.animation_frame = 0
                if _Debug:
                    print(f'playing previous animation {unit.animation_playing} for unit {unit.name}')
        elif keycode[1] == 'c':
            animated_units_onstage = []
            for unit in self.scene.units.values():
                if unit.static:
                    continue
                animated_units_onstage.append(unit.name)
            # for name in animated_units_onstage:
            #     self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=name)
            self.app_root.test_id += 1
            if self.app_root.test_id >= len(self.app_root.known_templates):
                self.app_root.test_id = 0
            template_name = sorted(self.app_root.known_templates.keys())[self.app_root.test_id]
            while template_name in ['unhufe', 'unhuma', 'unorfe', 'unorma', 'unmoco2', 'efcu0']:
                self.app_root.test_id += 1
                if self.app_root.test_id >= len(self.app_root.known_templates):
                    self.app_root.test_id = 0
                template_name = sorted(self.app_root.known_templates.keys())[self.app_root.test_id]
            test_model_data = self.app_root.known_templates[template_name][0]
            unit = self.scene.place_animated_unit_on_land(
                template=template_name,
                map_w=self.scene.area_center_w,
                map_h=self.scene.area_center_h,
                shift_w=0.5,
                shift_h=0.5,
                direction=random.randint(0, 360),
                textures={'*': test_model_data['t'].lower()},
                coefs=test_model_data['c'],
            )
            unit.max_speed = random.randint(5, 40) / 1000.0
            unit.acceleration = random.randint(1, 10) / 1000.0
        elif keycode[1] == 'l':
            animated_units_onstage = []
            for unit in self.scene.units.values():
                if unit.static:
                    continue
                animated_units_onstage.append(unit.name)
            animated_units_onstage = sorted(animated_units_onstage)
            if self.camera_unit_lock:
                current_index = animated_units_onstage.index(self.camera_unit_lock)
                current_index += 1
                if current_index >= len(animated_units_onstage):
                    current_index = 0
                self.camera_unit_lock = animated_units_onstage[current_index]
            else:
                if animated_units_onstage:
                    self.camera_unit_lock = animated_units_onstage[0]
        elif keycode[1] == 'a':
            self.scene.shift_land(0, self.scene.LAND_MOVE_SPEED)
        elif keycode[1] == 'd':
            self.scene.shift_land(0, -self.scene.LAND_MOVE_SPEED)
        elif keycode[1] == 's':
            self.scene.shift_land(self.scene.LAND_MOVE_SPEED, 0)
        elif keycode[1] == 'w':
            self.scene.shift_land(-self.scene.LAND_MOVE_SPEED, 0)
        return True

    @ignore_undertouch
    def on_touch_down(self, touch):
        touch.grab(self)
        self.touches.append(touch)
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown'):
            factor = self.camera_distance_scale_factor
            if touch.button == "scrolldown":
                factor = factor * (1.0 - self.SCALE_SPEED_FACTOR)
            if touch.button == "scrollup":
                factor = factor * (1.0 + self.SCALE_SPEED_FACTOR)
            if factor >= self.SCALE_MIN and factor <= self.SCALE_MAX:
                self.camera_distance_scale_factor = factor
                # if _Debug:
                #     print(f'new scale factor is {self.camera_distance_scale_factor}')

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
                new_global_rotate_x_angle = self.camera_angle_y - ay
                if new_global_rotate_x_angle < self.ROTATE_VERTICAL_MIN:
                    new_global_rotate_x_angle = self.ROTATE_VERTICAL_MIN
                if new_global_rotate_x_angle > self.ROTATE_VERTICAL_MAX:
                    new_global_rotate_x_angle = self.ROTATE_VERTICAL_MAX
                self.camera_angle_y = new_global_rotate_x_angle
                self.camera_angle_z -= ax
                # if _Debug:
                #     print(f'new camera angle y:{self.camera_angle_y} z:{self.camera_angle_z}')

    def on_setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)
        # glDepthFunc(GL_LEQUAL)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_BLEND)

    def on_reset_gl_context(self, *args):
        # glDisable(GL_BLEND)
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
        self.setup_canvas()
