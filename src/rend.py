import os
import sys
import math
import random

_Debug = True


from kivy.base import EventLoop
from kivy.app import App
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.resources import resource_find
from kivy.properties import ObjectProperty  # @UnresolvedImport
from kivy.graphics.transformation import Matrix  # @UnresolvedImport
from kivy.vector import Vector
from kivy.graphics.opengl import (
    glGetError, glEnable, glDisable, GL_BLEND, GL_DEPTH_TEST,  # @UnresolvedImport
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,  # @UnresolvedImport
    glDepthFunc, GL_LEQUAL,  # @UnresolvedImport
    glGetIntegerv, GL_DEPTH_BITS,  # @UnresolvedImport
)
from kivy.graphics.instructions import InstructionGroup  # @UnresolvedImport
from kivy.graphics import (
    RenderContext, Callback, BindTexture,
    ChangeState, PushState, PopState,
    PushMatrix, PopMatrix,
    # Scale,
    Color, Translate, Rotate, Mesh, Line,
    # UpdateNormalMatrix,
)

import const
import system
import mth
import res


def ignore_undertouch(func):
    def wrap(self, touch):
        glst = touch.grab_list
        if len(glst) == 0 or (self is glst[0]()):
            return func(self, touch)
    return wrap


VERTEX_SHADER = """
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

FRAGMENT_SHADER = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec2 tex_coord0;
varying vec4 normal_vec;
varying vec4 vertex_pos;

uniform sampler2D texture_id;
uniform float brightness;
uniform float contrast;
// uniform float camera_distance;
uniform vec3 center_point;
// uniform float segment_fog_factor;
// uniform float dist_to_center;
uniform float material_density;
uniform float water_transparency;
uniform float fog_density;
uniform float fog_radius;
// uniform vec3 fog_color;
uniform mat4 normal_mat;

void main (void){
    // vec4 back_tex_color = texture2D(background_texture_id, tex_coord0).rgba;
    // vec3 camera_pos = vec3(0.0, 0.0, 0.0);
    // float camera_distance = length(camera_pos - vertex_pos.xyz);
    // float fog_factor = 1.0 / exp(camera_distance * fog_density);
    // float fog_factor = exp(-pow(camera_distance * fog_density, 2.0));
    // float center_point_distance = length(center_point - vertex_pos.xyz) - camera_distance;
    // float fog_factor = 1.0 / exp(center_point_distance * fog_density);
    // float fog_factor = exp(-pow(center_point_distance * fog_density, 2.0));
    // float fog_factor = 1.0;
    // float fog_factor = exp(-pow(center_point_distance * fog_density, 2.0));
    // float fog_factor = 1.0 / exp(center_point_distance * fog_density * (1.0 - segment_fog_factor) * (1.0 - segment_fog_factor));    
    // fog_factor = fog_factor * fog_factor * fog_factor;
    // fog_factor = clamp(fog_factor, 0.0, 1.0);
    // float fog_factor = clamp(segment_fog_factor, 0.0, 1.0);
    // float fog_factor = exp(-pow(camera_distance * fog_density, 2.0));
    // float fog_factor = exp(-pow(dist_to_center * fog_density, 2.0));
    // float fog_factor = 0.5 - dist_to_center / 24.0;
    // fog_factor = clamp(fog_factor, 0.0, 1.0);
    // float distance = ( 1.0 - segment_fog_factor ) * ( 1.0 - segment_fog_factor ) * ( 1.0 - segment_fog_factor ) * 24.0;
    // float fog_factor = exp(-pow(distance * fog_density, 2.0));
    // fog_factor = clamp(fog_factor, 0.0, 1.0);
    // vec3 color_with_fog = mix(fog_color, tex_color.rgb, fog_factor);
    // vec3 color_with_fog = tex_color.rgb;
    // vec3 result_color = mix(fog_color, tex_color.rgb, fog_factor);
    // result_color = (result_color - 0.5) * contrast + 0.5 + brightness;
    // float result_alpha = tex_color.a;
    // vec3 fog_color = vec3(0.7, 0.7, 0.7);
    // vec3 result_color = mix(fog_color, tex_color.rgb, fog_factor);
    // result_color = (result_color - 0.5) * contrast + 0.5 + brightness;
    // float result_alpha = tex_color.a * fog_factor;

    vec4 tex_color = texture2D(texture_id, tex_coord0).rgba;
    float center_point_distance = length(center_point - vertex_pos.xyz);
    center_point_distance = clamp(center_point_distance - fog_radius, 0.0, center_point_distance);
    float fog_distance = center_point_distance * fog_density;
    float fog_factor = exp(-fog_distance * fog_distance);
    fog_factor = clamp(fog_factor + material_density, 0.0, 1.0);

    vec3 result_color = (tex_color.rgb - 0.5) * contrast + 0.5 + brightness;
    float result_alpha = tex_color.a * fog_factor;

    if ( result_alpha < 0.5) {
        discard;
    }

    gl_FragColor = vec4(result_color, result_alpha);
}
"""


class Renderer(Widget):

    def __init__(self, app_root, scene, **kwargs):
        EventLoop.ensure_window()  # Make sure OpenGL context exists
        self.app_root = app_root
        self.scene = scene
        self.canvas = RenderContext(compute_normal_mat=True, vs=VERTEX_SHADER, fs=FRAGMENT_SHADER)
        # self.canvas.shader.source = resource_find('shader.glsl')
        self.camera_distance_scale_factor = const.SCALE_INITIAL
        self.camera_distance_to_center = const.CAMERA_DISTANCE_TO_CENTER_INITIAL
        self.camera_angle_y = float(const.ROTATE_VERTICAL_INITIAL)
        self.camera_angle_z = 180.0
        self.camera_acceleration = 0.0
        self.camera_speed = 0.0
        self.camera_unit_lock = None
        self.camera_capital_lock = 0
        self.camera_move_mode = 1
        self.global_eye_x = 0
        self.global_eye_y = 0
        self.global_eye_z = 0
        self.global_center_x = 0
        self.global_center_y = 0
        self.global_center_z = 0
        self.fog_center_x = 0.0
        self.fog_center_y = 0.0
        self.fog_center_z = 0.0
        self.sky_background_rotate_x = None
        self.sky_background_rotate_y = None
        self.sky_background_rotate_z = None
        self.sky_background_translate = None
        self.sky_background_mesh = None
        self.touches = []
        self.active_touches = {}
        self.initial_double_touch_dist = 0.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.this_template_name = None
        self.this_template_variant_index = None
        self.this_template_variant_coefs_index = None
        self.this_template_model_kind = 0
        self.this_template_coefs = [0.0, 0.0, 0.0]
        self.this_template_scale = [1.0, 1.0, 1.0]
        self.this_template_figure_index = 80
        self.this_template_figure_part_index = 0
        self.this_animation_frame = None
        self.joystick_button = None
        self.joystick_latest_state = None
        self.depth_bits_reported = False
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.on_setup_gl_context)
            PushMatrix()
            self.create_sky_background()
            self.scene.create_container()
            PopMatrix()
            self.cb = Callback(self.on_reset_gl_context)
        self.canvas['texture_id'] = 1
        self.keyboard_handler = None
        if not system.is_android():
            self.keyboard_handler = Window.request_keyboard(self.on_keyboard_closed, self)
            self.keyboard_handler.bind(on_key_down=self.on_keyboard_down)
            self.keyboard_handler.bind(on_key_up=self.on_keyboard_up)
        Clock.schedule_interval(self.on_update_glsl, 1 / 60)
        Clock.schedule_interval(self.on_update_animations, 1 / 20 )  # 1 / 24)
        Clock.schedule_interval(self.on_run_units, 1 / 20)

    def create_sky_background(self):
        PushMatrix()
        sz_w = const.CAMERA_VIEW_CLIP_FAR * 2.0
        sz_h = const.CAMERA_VIEW_CLIP_FAR * 0.8
        shift_down = const.CAMERA_VIEW_CLIP_FAR * 0.3
        self.sky_background_rotate_z = Rotate(0, 0, 0, 1, group='land')
        self.sky_background_rotate_y = Rotate(0, 0, 1, 0, group='land')
        self.sky_background_rotate_x = Rotate(0, 1, 0, 0, group='land')
        self.sky_background_translate = Translate(0, 0, 0, group='land')
        ChangeState(material_density=1.0)
        sky_background_image = Image(source=res.data_path('assets/sky1.png'))
        sky_background_texture = sky_background_image.texture
        sky_background_texture.wrap = 'repeat'
        BindTexture(texture=sky_background_texture, index=1)
        self.sky_background_mesh = Mesh(
            vertices=[
                -1 * sz_w / 2,  sz_h - shift_down, 0, 1, 0, 0, 0.0, 1.0,
                1 * sz_w / 2,   sz_h - shift_down, 0, 1, 0, 0, 1.0, 1.0,
                1 * sz_w / 2,   - shift_down,      0, 1, 0, 0, 1.0, 0.0,
                - 1 * sz_w / 2, - shift_down,      0, 1, 0, 0, 0.0, 0.0,               
            ],
            indices=[0, 1, 2, 0, 2, 3],
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
        )
        ChangeState(material_density=0.0)
        PopMatrix()

    def update_sky_background(self):
        sz_w = const.CAMERA_VIEW_CLIP_FAR * 2.0
        sz_h = const.CAMERA_VIEW_CLIP_FAR * 1.0
        shift_down = const.CAMERA_VIEW_CLIP_FAR * 0.5
        parallax_vertical = (self.camera_angle_y + 105.0) / 95.0
        parallax_horizontal = self.camera_angle_z / 90.0
        self.sky_background_mesh.vertices = [
            -1 * sz_w / 2,  sz_h - shift_down, 0, 1, 0, 0, 0.0 + parallax_horizontal, 0.0 - parallax_vertical,
            1 * sz_w / 2,   sz_h - shift_down, 0, 1, 0, 0, 1.0 + parallax_horizontal, 0.0 - parallax_vertical,
            1 * sz_w / 2,   - shift_down,      0, 1, 0, 0, 1.0 + parallax_horizontal, 1.0 - parallax_vertical,
            - 1 * sz_w / 2, - shift_down,      0, 1, 0, 0, 0.0 + parallax_horizontal, 1.0 - parallax_vertical,
        ]
        self.sky_background_rotate_x.angle = 90 - self.camera_angle_y
        self.sky_background_rotate_y.angle = 180 + self.camera_angle_z
        self.sky_background_translate.z = const.CAMERA_VIEW_CLIP_FAR - 0.5 - self.camera_distance_scale_factor * self.camera_distance_to_center

    def update_canvas(self):
        win_w, win_h = Window.size
        asp = win_w / float(win_h)
        # if _Debug:
        #     print(f'Renderer.update_canvas: win_w={win_w} win_h={win_h} asp={asp} win={Window.size} renderer={self.size}')
        if asp > 2.0:
            asp = 2.0
        if asp < 0.5:
            asp = 0.5
        # self.on_gl_error('step 1')
        # self.canvas['texture_id'] = 1
        self.global_eye_x = float(self.camera_distance_scale_factor) * self.camera_distance_to_center * math.sin(math.radians(self.camera_angle_y)) * math.sin(math.radians(self.camera_angle_z))
        self.global_eye_y = float(self.camera_distance_scale_factor) * self.camera_distance_to_center * math.cos(math.radians(self.camera_angle_y))
        self.global_eye_z = float(self.camera_distance_scale_factor) * self.camera_distance_to_center * math.sin(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_z))
        self.update_sky_background()
        self.canvas['projection_mat'] = Matrix().view_clip(-asp, asp, -1, 1, const.CAMERA_VIEW_CLIP_NEAR, const.CAMERA_VIEW_CLIP_FAR, 1)
        self.canvas['modelview_mat'] = Matrix().look_at(
            self.global_eye_x, self.global_eye_y, self.global_eye_z,
            self.global_center_x, self.global_center_y, self.global_center_z,
            0, 1, 0,  # up vector
        )
        self.canvas['center_point'] = (0.0, 0.0, - float(self.camera_distance_scale_factor) * float(self.camera_distance_to_center))
        # if _Debug:
        #     print(f'updating canvas center_point={self.canvas["center_point"]} asp={asp}')
        self.canvas['brightness'] = self.brightness
        self.canvas['contrast'] = self.contrast
        self.canvas['fog_density'] = 0.08
        self.canvas['fog_radius'] = (const.VISIBLE_AREA_SIZE_SEGMENTS_HALF - 3) * const.SEGMENT_SIZE
        self.canvas['material_density'] = 0.0
        self.canvas['water_transparency'] = 10.0 / 255.0
        # self.on_gl_error('step 2')

    def define_rotate_angle(self, touch):
        x_angle = (float(touch.dx) / float(self.width)) * 360.0 * const.ROTATE_SPEED
        y_angle = -1.0 * (float(touch.dy) / float(self.height)) * 360.0 * const.ROTATE_SPEED
        return x_angle, y_angle

    def on_keyboard_closed(self):
        self.keyboard_handler.unbind(on_key_down=self.on_keyboard_down)
        self.keyboard_handler = None

    def _show_unit(self, template_data, scale=[1.0, 1.0, 1.0], coefs_index=None):
        if coefs_index is None:
            coefs_index = self.this_template_variant_coefs_index
        animated_units_onstage = []
        for unit in self.scene.units.values():
            if unit.static:
                continue
            animated_units_onstage.append(unit.name)
        for name in animated_units_onstage:
            self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=name)
        self.scene.meshes_index.clear()
        coefs = [float(c) for c in (template_data['c'].split(' ')[coefs_index]).split(':')]
        unit = self.scene.place_animated_unit_on_land(
            template=template_data['m'],
            map_w=self.scene.area_center_w,
            map_h=self.scene.area_center_h,
            shift_w=0.5,
            shift_h=0.5,
            direction=0, # random.randint(0, 360),
            elevation_correction=-5.0,
            selected_parts=template_data['p'] if template_data['p'] else None,
            selected_animations='*',
            textures={'*': template_data['t'].lower()},
            coefs=coefs,
            scale=scale,
        )
        if not unit:
            return
        unit.max_speed = 0 # random.randint(1, 50) / 1000.0
        unit.acceleration = 0 # random.randint(1, 5) / 1000.0
        if _Debug:
            d = template_data.copy()
            print(f'    showing template {template_data["m"]} with {len(unit.parts)} parts coefs={coefs} scale={scale}:\n    {d}')

    def on_joystick_move(self, rel_x, rel_y):
        self.joystick_latest_state = (rel_x, rel_y)
        # if not rel_x and not rel_y:
        #     self.camera_speed = 0.0
        #     self.camera_acceleration = 0.0
        #     if self.scene.hero:
        #         self.scene.hero.move(forward=False, backward=False)
        # else:
        #     joystick_direction = math.degrees(math.atan2(rel_y, rel_x)) + self.camera_angle_z
        #     if joystick_direction > 180.0:
        #         joystick_direction -= 360.0
        #     if joystick_direction < -180.0:
        #         joystick_direction += 360.0
        #     hero_direction = self.scene.hero.get_unit().direction
        #     if _Debug:
        #         print(f'            on_joystick_move joystick={joystick_direction} camera={self.camera_angle_z} hero={hero_direction}')
        #     direction_diff = joystick_direction - hero_direction
        #     if direction_diff > 180.0:
        #         direction_diff -= 360.0
        #     if direction_diff < -180.0:
        #         direction_diff += 360.0
        #     if direction_diff > 0.0:
        #         self.scene.hero.turn(right=True, turn_max=abs(direction_diff))
        #     else:
        #         self.scene.hero.turn(left=True, turn_max=abs(direction_diff))
        #     self.scene.hero.move(forward=True)

    def on_keyboard_up(self, keyboard, keycode, *largs):
        self.camera_speed = 0.0
        self.camera_acceleration = 0.0
        if keycode[1] in ['w', 's', ]:
            if self.scene.hero:
                self.scene.hero.move(forward=False, backward=False)
    
    def on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
            App.get_running_app().stop()
        # elif keycode[1] == 'u':
        #     # self.contrast += 0.1
        #     # self.this_template_coefs[0] += 0.1
        #     template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
        #     # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
        #     self.this_template_variant_coefs_index = 0
        #     self.this_template_scale[0] += 0.1
        #     self._show_unit(template_data, scale=self.this_template_scale)
        # elif keycode[1] == 'i':
        #     # self.contrast -= 0.1
        #     # self.this_template_coefs[0] -= 0.1
        #     template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
        #     # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
        #     self.this_template_variant_coefs_index = 0
        #     self.this_template_scale[0] -= 0.1
        #     self._show_unit(template_data, scale=self.this_template_scale)
        # elif keycode[1] == 'o':
        #     # self.brightness += 0.1
        #     # self.this_template_coefs[1] += 0.1
        #     template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
        #     # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
        #     self.this_template_variant_coefs_index = 0
        #     self.this_template_scale[1] += 0.1
        #     self._show_unit(template_data, scale=self.this_template_scale)
        # elif keycode[1] == 'p':
        #     # self.brightness -= 0.1
        #     # self.this_template_coefs[1] -= 0.1
        #     template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
        #     # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
        #     self.this_template_variant_coefs_index = 0
        #     self.this_template_scale[1] -= 0.1
        #     self._show_unit(template_data, scale=self.this_template_scale)
        # elif keycode[1] == 'l':
        #     # self.this_template_coefs[2] += 0.1
        #     template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
        #     # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
        #     self.this_template_variant_coefs_index = 0
        #     self.this_template_scale[2] += 0.1
        #     self._show_unit(template_data, scale=self.this_template_scale)
        # elif keycode[1] == 'k':
        #     # self.this_template_coefs[2] -= 0.1
        #     template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
        #     # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
        #     self.this_template_variant_coefs_index = 0
        #     self.this_template_scale[2] -= 0.1
        #     self._show_unit(template_data, scale=self.this_template_scale)
        elif keycode[1] == 'x':
            self.this_animation_frame = None
            for unit in self.scene.units.values():
                if not unit.animations_list:
                    continue
                current_animation_ind = unit.animations_list.index(unit.animation_playing)
                current_animation_ind += 1
                if current_animation_ind >= len(unit.animations_list):
                    current_animation_ind = 0
                unit.animation_playing = unit.animations_list[current_animation_ind]
                unit.animation_frame = 0
                unit.animation_next = None
                ao = self.scene.animated_objects[unit.object_name]
                animation = ao.animations[unit.animation_playing]
                root_part_name = ao.parts[0]
                root_part_animation = animation.parts.get(root_part_name)
                unit.animation_length = root_part_animation.frames
                if _Debug:
                    print(f'playing animation {unit.animation_playing} for unit {unit.name}')
        elif keycode[1] == 'z':
            self.this_animation_frame = None
            for unit in self.scene.units.values():
                if not unit.animations_list:
                    continue
                current_animation_ind = unit.animations_list.index(unit.animation_playing)
                current_animation_ind -= 1
                if current_animation_ind < 0:
                    current_animation_ind = len(unit.animations_list) - 1
                unit.animation_playing = unit.animations_list[current_animation_ind]
                unit.animation_frame = 0
                unit.animation_next = None
                ao = self.scene.animated_objects[unit.object_name]
                animation = ao.animations[unit.animation_playing]
                root_part_name = ao.parts[0]
                root_part_animation = animation.parts.get(root_part_name)
                unit.animation_length = root_part_animation.frames
                if _Debug:
                    print(f'playing animation {unit.animation_playing} for unit {unit.name}')
        elif keycode[1] == 'p':
            if self.this_animation_frame is None:
                self.this_animation_frame = 0
            else:
                self.this_animation_frame += 1
                if self.this_animation_frame >= 1000000:
                    self.this_animation_frame = 0
            for unit in self.scene.units.values():
                if not unit.animations_list:
                    continue
                ao = self.scene.animated_objects[unit.object_name]
                animation = ao.animations[unit.animation_playing]
                root_part_name = ao.parts[0]
                root_part_animation = animation.parts.get(root_part_name)
                unit.animation_length = root_part_animation.frames
                unit.animation_frame = self.this_animation_frame % unit.animation_length
                if _Debug:
                    print(f'playing animation {unit.animation_playing} for unit {unit.name} at frame {unit.animation_frame}')
                if not unit.static and unit.onstage:
                    unit.animate(self.scene, 0.1)
        elif keycode[1] == 'r':
            if self.this_template_name is None:
                self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
            else:
                current_index = sorted(self.app_root.known_templates.keys()).index(self.this_template_name)
                template_name = None
                while True:
                    current_index += 1
                    if current_index >= len(self.app_root.known_templates):
                        current_index = 0
                    template_name = sorted(self.app_root.known_templates.keys())[current_index]
                    template_data = self.app_root.known_templates[template_name][0]
                    if self.this_template_model_kind == 0:
                        break
                    if self.this_template_model_kind == 1:
                        if template_data['k'] in ['house', 'building', 'bridge', 'gate', 'wall', 'ruins']:
                            break
                    if self.this_template_model_kind == 2:
                        if template_data['t'].count('tree'):
                            break
                self.this_template_name = template_name
                # current_index += 1
                # if current_index >= len(self.app_root.known_templates):
                #     current_index = 0
                # self.this_template_name = sorted(self.app_root.known_templates.keys())[current_index]
            self.this_template_variant_index = 0
            self.this_template_coefs = [0.0, 0.0, 0.0]
            # null_variant_index = 0
            # while True:
            #     template_data = self.app_root.known_templates[self.this_template_name][null_variant_index]
            #     if template_data['p'] is None or template_data['i'].endswith('null'):
            #         break
            #     null_variant_index += 1
            #     if null_variant_index >= len(self.app_root.known_templates[self.this_template_name]):
            #         null_variant_index = 0
            #         break
            self.this_template_variant_coefs_index = 0  # null_variant_index
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            self._show_unit(template_data)
        elif keycode[1] == 'f':
            if self.this_template_name is None:
                self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
            else:
                current_index = sorted(self.app_root.known_templates.keys()).index(self.this_template_name)
                current_index -= 1
                if current_index < 0:
                    current_index = len(self.app_root.known_templates) - 1
                self.this_template_name = sorted(self.app_root.known_templates.keys())[current_index]
            self.this_template_variant_index = 0
            self.this_template_variant_coefs_index = 0
            self.this_template_coefs = [0.0, 0.0, 0.0]
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            self._show_unit(template_data)
        elif keycode[1] == 't':
            if self.this_template_name is None:
                self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
                self.this_template_variant_index = 0
            if self.this_template_variant_index is None:
                self.this_template_variant_index = 0
            else:
                self.this_template_variant_index += 1
            if self.this_template_variant_index >= len(self.app_root.known_templates[self.this_template_name]):
                self.this_template_variant_index = 0
            self.this_template_variant_coefs_index = 0
            animated_units_onstage = []
            for unit in self.scene.units.values():
                if unit.static:
                    continue
                animated_units_onstage.append(unit.name)
            for name in animated_units_onstage:
                self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=name)
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            self._show_unit(template_data)
        elif keycode[1] == 'g':
            if self.this_template_name is None:
                self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
                self.this_template_variant_index = 0
            if self.this_template_variant_index is None:
                self.this_template_variant_index = 0
            else:
                self.this_template_variant_index -= 1
                if self.this_template_variant_index < 0:
                    self.this_template_variant_index = len(self.app_root.known_templates[self.this_template_name]) - 1
            if self.this_template_variant_index >= len(self.app_root.known_templates[self.this_template_name]):
                self.this_template_variant_index = 0
            self.this_template_variant_coefs_index = 0
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            self._show_unit(template_data)
        elif keycode[1] == 'y':
            return True
            self.this_template_figure_index += 1
            if self.this_template_figure_index >= len(self.app_root.known_figures_parts):
                self.this_template_figure_index = 0
            print(f'next figure index is {self.this_template_figure_index}')
            self.this_template_figure_part_index = 0
            model_name, tex_name, parts_list = self.app_root.known_figures_parts[self.this_template_figure_index].split('#')
            parts_list = parts_list.split(':')
            parts_list_out = parts_list[self.this_template_figure_part_index]
            template_data = {
                'i': f'{model_name}#{tex_name}#{parts_list_out}',
                'm': model_name,
                't': tex_name,
                'p': [parts_list_out, ],
                'c': '0.0:0.0:0.0',
                's': '1.0:1.0:1.0',
            }
            try:
                self._show_unit(template_data, coefs_index=0)
            except:
                import traceback
                traceback.print_exc()
            # if self.this_template_name is None:
            #     self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
            #     self.this_template_variant_index = 0
            # if self.this_template_variant_index is None:
            #     self.this_template_variant_index = 0
            #     self.this_template_variant_coefs_index = 0
            # if self.this_template_variant_coefs_index is None:
            #     self.this_template_variant_coefs_index = 0
            # else:
            #     self.this_template_variant_coefs_index += 1
            # template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # if self.this_template_variant_coefs_index >= len(template_data['c'].split(' ')):
            #     self.this_template_variant_coefs_index = 0
            # self._show_unit(template_data)
        elif keycode[1] == 'h':
            return True
            self.this_template_figure_index -= 1
            if self.this_template_figure_index < 0:
                self.this_template_figure_index = len(self.app_root.known_figures_parts) - 1
            self.this_template_figure_part_index = 0
            model_name, tex_name, parts_list = self.app_root.known_figures_parts[self.this_template_figure_index].split('#')
            parts_list = parts_list.split(':')
            parts_list_out = parts_list[self.this_template_figure_part_index]
            template_data = {
                'i': f'{model_name}#{tex_name}#{parts_list_out}',
                'm': model_name,
                't': tex_name,
                'p': [parts_list_out, ],
                'c': '0.0:0.0:0.0',
                's': '1.0:1.0:1.0',
            }
            try:
                self._show_unit(template_data, coefs_index=0)
            except:
                import traceback
                traceback.print_exc()
            # if self.this_template_name is None:
            #     self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
            #     self.this_template_variant_index = 0
            # if self.this_template_variant_index is None:
            #     self.this_template_variant_index = 0
            #     self.this_template_variant_coefs_index = 0
            # if self.this_template_variant_coefs_index is None:
            #     self.this_template_variant_coefs_index = 0
            # else:
            #     if self.this_template_variant_coefs_index > 0:
            #         self.this_template_variant_coefs_index -= 1
            # template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # self._show_unit(template_data)
        elif keycode[1] == 'u':
            return True
            model_name, tex_name, parts_list = self.app_root.known_figures_parts[self.this_template_figure_index].split('#')
            parts_list = parts_list.split(':')
            self.this_template_figure_part_index += 1
            if self.this_template_figure_part_index >= len(parts_list):
                self.this_template_figure_part_index = 0
            parts_list_out = [parts_list[self.this_template_figure_part_index], ]
            # if self.this_template_figure_part_index != 0:
            #     parts_list_out = [parts_list[0], ] + parts_list_out
            template_data = {
                'i': f'{model_name}#{tex_name}#{":".join(parts_list_out)}',
                'm': model_name,
                't': tex_name,
                'p': parts_list_out,
                'c': '0.0:0.0:0.0',
                's': '1.0:1.0:1.0',
            }
            try:
                self._show_unit(template_data, coefs_index=0)
            except:
                import traceback
                traceback.print_exc()
        elif keycode[1] == 'j':
            return True
            model_name, tex_name, parts_list = self.app_root.known_figures_parts[self.this_template_figure_index].split('#')
            parts_list = parts_list.split(':')
            self.this_template_figure_part_index -= 1
            if self.this_template_figure_part_index < 0:
                self.this_template_figure_part_index = len(parts_list) - 1
            parts_list_out = [parts_list[self.this_template_figure_part_index], ]
            # if self.this_template_figure_part_index != 0:
            #     parts_list_out = [parts_list[0], ] + parts_list_out
            template_data = {
                'i': f'{model_name}#{tex_name}#{":".join(parts_list_out)}',
                'm': model_name,
                't': tex_name,
                'p': parts_list_out,
                'c': '0.0:0.0:0.0',
                's': '1.0:1.0:1.0',
            }
            try:
                self._show_unit(template_data, coefs_index=0)
            except:
                import traceback
                traceback.print_exc()
        elif keycode[1] == 'n':
            if self.this_template_name:
                template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
                texture_name = template_data['t'].lower()
                image_file_path = os.path.join('textures', 'model', texture_name+'.png')
                if os.path.isfile(image_file_path):
                    from PIL import Image as PILImage
                    from PIL.Image import Transpose as PILTranspose
                    image = PILImage.open(image_file_path)
                    image.load()
                    flipped_image = image.transpose(PILTranspose.FLIP_TOP_BOTTOM)
                    flipped_image.save(image_file_path)
                    if _Debug:
                        print(f'Flipped image {image_file_path} vertically')
                    file_path_source = resource_find(image_file_path)
                    if file_path_source:
                        _tex = Cache.get('kv.texture', image_file_path)
                        if _tex:
                            Cache.remove('kv.texture', image_file_path)
                            if _Debug:
                                print(f'Cleared cached texture for {image_file_path}')
                            self._show_unit(template_data)
        elif keycode[1] == 'm':
            self.this_template_model_kind += 1
            if self.this_template_model_kind > 2:
                self.this_template_model_kind = 0
            if _Debug:
                print(f'model kind is now {self.this_template_model_kind}')
        elif keycode[1] == 'b':
            if self.camera_unit_lock:
                self.camera_unit_lock = None
        elif keycode[1] == 'v':
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
                if _Debug:
                    print(f'camera locked to unit {self.camera_unit_lock}')
            else:
                if animated_units_onstage:
                    self.camera_unit_lock = animated_units_onstage[0]
        elif keycode[1] == 'c':
            if self.camera_unit_lock:
                self.camera_unit_lock = None
            self.camera_capital_lock += 1
            if self.camera_capital_lock > len(self.scene.land.capitals):
                self.camera_capital_lock = 0
                print('camera unlocked from capital')
            else:
                print(f'camera locked to capital {self.scene.land.capitals[self.camera_capital_lock]["n"]}')
        elif keycode[1] == 'e':
            self.camera_move_mode += 1
            if self.camera_move_mode > 3:
                self.camera_move_mode = 1
        elif keycode[1] == 'a':
            if self.camera_move_mode == 1:
                if self.scene.hero:
                    self.scene.hero.turn(left=True)
            elif self.camera_move_mode == 2:
                self.scene.land_shift(0, const.LAND_MOVE_SPEED)
            elif self.camera_move_mode == 3:
                self.scene.land_move(self.camera_angle_z + 90, const.LAND_MOVE_SPEED)
        elif keycode[1] == 'd':
            if self.camera_move_mode == 1:
                if self.scene.hero:
                    self.scene.hero.turn(right=True)
            elif self.camera_move_mode == 2:
                self.scene.land_shift(0, -const.LAND_MOVE_SPEED)
            elif self.camera_move_mode == 3:
                self.scene.land_move(self.camera_angle_z - 90, const.LAND_MOVE_SPEED)
        elif keycode[1] == 's':
            if self.camera_move_mode == 1:
                return True
                if self.scene.hero:
                    self.scene.hero.move(backward=True)
            elif self.camera_move_mode == 2:
                self.scene.land_shift(-const.LAND_MOVE_SPEED, 0)
            elif self.camera_move_mode == 3:
                self.camera_acceleration -= const.LAND_MOVE_SPEED / 20.0
                self.camera_speed += self.camera_acceleration
                if self.camera_speed < -const.LAND_MOVE_SPEED:
                    self.camera_speed = -const.LAND_MOVE_SPEED
                self.scene.land_move(self.camera_angle_z, self.camera_speed)
                self.camera_acceleration = 0.0
        elif keycode[1] == 'w':
            if self.camera_move_mode == 1:
                if self.scene.hero:
                    self.scene.hero.move(forward=True)
            elif self.camera_move_mode == 2:
                self.scene.land_shift(const.LAND_MOVE_SPEED, 0)
            elif self.camera_move_mode == 3:
                self.camera_acceleration += const.LAND_MOVE_SPEED / 20.0
                self.camera_speed += self.camera_acceleration
                if self.camera_speed > const.LAND_MOVE_SPEED:
                    self.camera_speed = const.LAND_MOVE_SPEED
                self.scene.land_move(self.camera_angle_z, self.camera_speed)
                self.camera_acceleration = 0.0
        elif keycode[1] == 'o':
            self.camera_unit_lock = None
            u = self.scene.hero.get_unit()
            coefs = u.coefs
            coefs[2] += 0.1
            self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=u.name)
            self.scene.meshes_index.clear()
            self.scene.models.clear()
            self.scene.hero.create_unit(u.w, u.h, coefs=coefs)
        elif keycode[1] == 'i':
            self.camera_unit_lock = None
            u = self.scene.hero.get_unit()
            coefs = u.coefs
            coefs[2] -= 0.1
            self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=u.name)
            self.scene.meshes_index.clear()
            self.scene.models.clear()
            self.scene.hero.create_unit(u.w, u.h, coefs=coefs)
        return True

    @ignore_undertouch
    def on_touch_down(self, touch):
        # if _Debug:
        #     print(f'        renderer.on_touch_down {touch} is_touch={touch.is_touch} type_id={touch.type_id} profile={touch.profile} button={touch.button} shape={touch.shape} is_double_tap={touch.is_double_tap} is_triple_tap={touch.is_triple_tap}')
        # this is rendering context and there is nothing expected to be behind it - no need to check for collide_point
        touch.grab(self)
        self.touches.append(touch)
        self.active_touches[touch.id] = touch.pos
        if len(self.active_touches) == 2:
            positions = list(self.active_touches.values())
            p1 = Vector(positions[0])
            p2 = Vector(positions[1])
            self.initial_double_touch_dist = p1.distance(p2)
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown'):
            factor = self.camera_distance_scale_factor
            if touch.button == "scrolldown":
                factor = factor * (1.0 - const.SCALE_SPEED_FACTOR)
            if touch.button == "scrollup":
                factor = factor * (1.0 + const.SCALE_SPEED_FACTOR)
            if factor < const.SCALE_MIN:
                factor = const.SCALE_MIN
            if factor > const.SCALE_MAX:
                factor = const.SCALE_MAX
            if factor != self.camera_distance_scale_factor:
                self.camera_distance_scale_factor = factor
                # if _Debug:
                #     print(f'new scale factor is {self.camera_distance_scale_factor}, camera distance to center is {float(self.camera_distance_scale_factor) * self.camera_distance_to_center}')
        return True

    @ignore_undertouch
    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            if touch.id in self.active_touches:
                del self.active_touches[touch.id]
            if touch in self.touches:
                self.touches.remove(touch)
            return True
        return super(Renderer, self).on_touch_up(touch)

    @ignore_undertouch
    def on_touch_move(self, touch):
        if touch in self.touches and touch.grab_current == self:
            self.active_touches[touch.id] = touch.pos
            if len(self.active_touches) == 2:
                positions = list(self.active_touches.values())
                p1 = Vector(positions[0])
                p2 = Vector(positions[1])
                current_dist = p1.distance(p2)
                if self.initial_double_touch_dist > 0:
                    scale_factor = current_dist / self.initial_double_touch_dist
                    factor = self.camera_distance_scale_factor
                    if scale_factor > 1.0:
                        factor = factor * (1.0 - const.SCALE_SPEED_FACTOR / 2.0)
                    else:
                        factor = factor * (1.0 + const.SCALE_SPEED_FACTOR / 2.0)
                    if factor < const.SCALE_MIN:
                        factor = const.SCALE_MIN
                    if factor > const.SCALE_MAX:
                        factor = const.SCALE_MAX
                    if factor != self.camera_distance_scale_factor:
                        self.camera_distance_scale_factor = factor
                        # if _Debug:
                        #     print(f'new scale factor is {self.camera_distance_scale_factor}, camera distance to center is {float(self.camera_distance_scale_factor) * self.camera_distance_to_center}')
            elif len(self.touches) == 1:
                ax, ay = self.define_rotate_angle(touch)
                new_global_rotate_angle = self.camera_angle_y - ay
                if new_global_rotate_angle < const.ROTATE_VERTICAL_MIN:
                    new_global_rotate_angle = const.ROTATE_VERTICAL_MIN
                if new_global_rotate_angle > const.ROTATE_VERTICAL_MAX:
                    new_global_rotate_angle = const.ROTATE_VERTICAL_MAX
                self.camera_angle_y = new_global_rotate_angle
                self.camera_angle_z -= ax
                if self.camera_angle_z > 360.0:
                    self.camera_angle_z -= 360.0
                if self.camera_angle_z < 0.0:
                    self.camera_angle_z += 360.0
                self.scene.on_camera_rotate(self.camera_angle_y, self.camera_angle_z)
                # if _Debug:
                #     print(f'new camera angle y:{self.camera_angle_y} z:{self.camera_angle_z}')
            return True
        return super(Renderer, self).on_touch_move(touch)

    def on_setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)
        if _Debug and not self.depth_bits_reported:
            self.depth_bits_reported = True
            try:
                print(f'depth buffer bits: {glGetIntegerv(GL_DEPTH_BITS)}')
            except Exception as exc:
                print(f'failed to read depth buffer bits: {exc}')
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
        self.update_canvas()

    def on_update_animations(self, delta):
        if self.this_animation_frame is not None:
            return
        # TODO: maintain separate list of active animations for all units
        # then it is not required to loop all units
        for unit in self.scene.units.values():
            if not unit.static and unit.onstage:
                unit.animate(self.scene, delta)

    def on_run_units(self, delta):
        if self.camera_unit_lock:
            u = self.scene.units.get(self.camera_unit_lock)
            if u:
                if self.joystick_latest_state and self.scene.hero.unit_name == u.name:
                    rel_x, rel_y = self.joystick_latest_state
                    if not rel_x and not rel_y:
                        self.camera_speed = 0.0
                        self.camera_acceleration = 0.0
                        if self.scene.hero:
                            self.scene.hero.move(forward=False, backward=False)
                    else:
                        joystick_direction = math.degrees(math.atan2(rel_y, rel_x)) + self.camera_angle_z
                        if joystick_direction > 180.0:
                            joystick_direction -= 360.0
                        if joystick_direction < -180.0:
                            joystick_direction += 360.0
                        hero_direction = self.scene.hero.get_unit().direction
                        # if _Debug:
                        #     print(f'            on_run_units hero joystick={joystick_direction} camera={self.camera_angle_z} hero={hero_direction}')
                        direction_diff = joystick_direction - hero_direction
                        if direction_diff > 180.0:
                            direction_diff -= 360.0
                        if direction_diff < -180.0:
                            direction_diff += 360.0
                        if direction_diff > 0.0:
                            self.scene.hero.turn(right=True, turn_max=abs(direction_diff))
                        else:
                            self.scene.hero.turn(left=True, turn_max=abs(direction_diff))
                        self.scene.hero.move(forward=True)
                self.scene.update_land(new_position=(u.w, u.h, u.shift_w, u.shift_h))
        elif self.camera_capital_lock > 0:
            capital = self.scene.land.capitals.get(self.camera_capital_lock, None)
            if capital:
                if self.scene.area_center_w != capital['x'] or self.scene.area_center_h != capital['y']:
                    self.scene.update_land(new_position=(capital['x'], capital['y'], 0, 0))
        for unit in self.scene.units.values():
            if not unit.static:
                unit.run(self.scene)
