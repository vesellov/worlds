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
    Color, Translate, Rotate, Mesh, Line,
    # UpdateNormalMatrix,
)

import const
import mth


def ignore_undertouch(func):
    def wrap(self, touch):
        glst = touch.grab_list
        if len(glst) == 0 or (self is glst[0]()):
            return func(self, touch)
    return wrap


class Renderer(Widget):

    def __init__(self, app_root, scene, **kwargs):
        EventLoop.ensure_window()  # Make sure OpenGL context exists
        self.app_root = app_root
        self.scene = scene
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.source = resource_find('assets/shader.glsl')
        self.camera_distance_scale_factor = const.SCALE_INITIAL
        self.camera_distance_to_center = const.CAMERA_DISTANCE_TO_CENTER_INITIAL
        self.camera_angle_y = float(const.ROTATE_VERTICAL_INITIAL)
        self.camera_angle_z = 180.0
        self.camera_unit_lock = None
        self.camera_move_mode = 2
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
        self.brightness = 0.0
        self.contrast = 1.0
        self.this_template_name = None
        self.this_template_variant_index = None
        self.this_template_variant_coefs_index = None
        self.this_template_model_kind = 0
        self.this_template_coefs = [0.0, 0.0, 0.0]
        self.this_template_scale = [1.0, 1.0, 1.0]
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.on_setup_gl_context)
            PushMatrix()
            self.create_sky_background()
            self.scene.create_container()
            PopMatrix()
            self.cb = Callback(self.on_reset_gl_context)
        self.canvas['texture_id'] = 1
        self.keyboard_handler = Window.request_keyboard(self.on_keyboard_closed, self)
        self.keyboard_handler.bind(on_key_down=self.on_keyboard_down)
        Clock.schedule_interval(self.on_update_glsl, 1 / 60)
        Clock.schedule_interval(self.scene.on_update_animations, 1 / 25)
        Clock.schedule_interval(self.scene.on_run_units, 1 / 60)

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
        sky_background_image = Image(source='assets/Cloudy_Sky-Night_04-1024x512.png')
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
        asp = self.width / float(self.height)
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

    def _show_unit(self, template_data, scale=[1.0, 1.0, 1.0]):
        animated_units_onstage = []
        for unit in self.scene.units.values():
            if unit.static:
                continue
            animated_units_onstage.append(unit.name)
        for name in animated_units_onstage:
            self.scene.remove_unit_from_stage(container=self.scene.container_animated_objects, unit_name=name)
        self.scene.meshes_index.clear()
        coefs = [float(c) for c in (template_data['c'].split(' ')[self.this_template_variant_coefs_index]).split(':')]
        unit = self.scene.place_animated_unit_on_land(
            template=self.this_template_name,
            map_w=self.scene.area_center_w,
            map_h=self.scene.area_center_h,
            shift_w=0.5,
            shift_h=0.5,
            direction=0, # random.randint(0, 360),
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
            print(f'    showing template {self.this_template_name} variant {self.this_template_variant_index} with {len(unit.parts)} parts coefs={coefs} scale={scale}:\n    {d}')

    def on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
            App.get_running_app().stop()
        elif keycode[1] == 'u':
            # self.contrast += 0.1
            # self.this_template_coefs[0] += 0.1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
            self.this_template_variant_coefs_index = 0
            self.this_template_scale[0] += 0.1
            self._show_unit(template_data, scale=self.this_template_scale)
        elif keycode[1] == 'i':
            # self.contrast -= 0.1
            # self.this_template_coefs[0] -= 0.1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
            self.this_template_variant_coefs_index = 0
            self.this_template_scale[0] -= 0.1
            self._show_unit(template_data, scale=self.this_template_scale)
        elif keycode[1] == 'o':
            # self.brightness += 0.1
            # self.this_template_coefs[1] += 0.1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
            self.this_template_variant_coefs_index = 0
            self.this_template_scale[1] += 0.1
            self._show_unit(template_data, scale=self.this_template_scale)
        elif keycode[1] == 'p':
            # self.brightness -= 0.1
            # self.this_template_coefs[1] -= 0.1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
            self.this_template_variant_coefs_index = 0
            self.this_template_scale[1] -= 0.1
            self._show_unit(template_data, scale=self.this_template_scale)
        elif keycode[1] == 'l':
            # self.this_template_coefs[2] += 0.1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
            self.this_template_variant_coefs_index = 0
            self.this_template_scale[2] += 0.1
            self._show_unit(template_data, scale=self.this_template_scale)
        elif keycode[1] == 'k':
            # self.this_template_coefs[2] -= 0.1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            # template_data['c'] = ':'.join(map(str, self.this_template_coefs))
            self.this_template_variant_coefs_index = 0
            self.this_template_scale[2] -= 0.1
            self._show_unit(template_data, scale=self.this_template_scale)
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
            if self.this_template_name is None:
                self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
                self.this_template_variant_index = 0
            if self.this_template_variant_index is None:
                self.this_template_variant_index = 0
                self.this_template_variant_coefs_index = 0
            if self.this_template_variant_coefs_index is None:
                self.this_template_variant_coefs_index = 0
            else:
                self.this_template_variant_coefs_index += 1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            if self.this_template_variant_coefs_index >= len(template_data['c'].split(' ')):
                self.this_template_variant_coefs_index = 0
            self._show_unit(template_data)
        elif keycode[1] == 'h':
            if self.this_template_name is None:
                self.this_template_name = sorted(self.app_root.known_templates.keys())[0]
                self.this_template_variant_index = 0
            if self.this_template_variant_index is None:
                self.this_template_variant_index = 0
                self.this_template_variant_coefs_index = 0
            if self.this_template_variant_coefs_index is None:
                self.this_template_variant_coefs_index = 0
            else:
                if self.this_template_variant_coefs_index > 0:
                    self.this_template_variant_coefs_index -= 1
            template_data = self.app_root.known_templates[self.this_template_name][self.this_template_variant_index]
            self._show_unit(template_data)
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
        elif keycode[1] == 'e':
            self.camera_move_mode = 3 - self.camera_move_mode
        elif keycode[1] == 'a':
            if self.camera_move_mode == 1:
                self.scene.land_shift(0, const.LAND_MOVE_SPEED)
            else:
                self.scene.land_move(self.camera_angle_z + 90, const.LAND_MOVE_SPEED)
        elif keycode[1] == 'd':
            if self.camera_move_mode == 1:
                self.scene.land_shift(0, -const.LAND_MOVE_SPEED)
            else:
                self.scene.land_move(self.camera_angle_z - 90, const.LAND_MOVE_SPEED)
        elif keycode[1] == 's':
            if self.camera_move_mode == 1:
                self.scene.land_shift(-const.LAND_MOVE_SPEED, 0)
            else:
                self.scene.land_move(self.camera_angle_z, -const.LAND_MOVE_SPEED)
        elif keycode[1] == 'w':
            if self.camera_move_mode == 1:
                self.scene.land_shift(const.LAND_MOVE_SPEED, 0)
            else:
                self.scene.land_move(self.camera_angle_z, const.LAND_MOVE_SPEED)
        return True

    @ignore_undertouch
    def on_touch_down(self, touch):
        touch.grab(self)
        self.touches.append(touch)
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
        self.update_canvas()
