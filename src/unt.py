import math
import random

from kivy.graphics.transformation import Matrix  # @UnresolvedImport

import const
import mth


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
        self.container_unit = None
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
        self.animation_next = None
        self.animation_frame = 0
        self.animation_length = 0
        self.action_types = {}
        self.direction = 0.0
        self.elevation_correction = None

    def __repr__(self):
        return f'Unit[{self.name}/{self.w}:{self.h}/{round(self.shift_w,2)}:{round(self.shift_h,2)}/{self.area_w}:{self.area_h}]'

    def set_position(self, scene, w, h, shift_w, shift_h, direction):
        """
        Places unit on the scene where the server says it is, the movement itself is calculated on the server.
        """
        self.direction = direction
        self.rotate_vertical.angle = self.direction + 90
        w_diff = w - self.w
        h_diff = h - self.h
        self.w = w
        self.h = h
        self.shift_w = shift_w
        self.shift_h = shift_h
        shift_vector = scene.coords_map2xyz(self.w, self.h, self.shift_w, self.shift_h, elevation_correction=self.elevation_correction)
        self.translate_shift.xyz = shift_vector
        if w_diff != 0 or h_diff != 0:
            self.area_w = self.w - int(scene.area_center_w)
            self.area_h = self.h - int(scene.area_center_h)
            segment_angle_x, segment_angle_z = scene.coords_area2angles(self.area_w, self.area_h)
            self.rotate_axis_x.angle = segment_angle_x
            self.rotate_axis_z.angle = segment_angle_z
            _w = int(self.w) - int(scene.area_center_w)
            _h = int(self.h) - int(scene.area_center_h)
            if self.onstage:
                if (_w, _h) in scene.visible_area_mask:
                    pass
                else:
                    scene.hide_unit(container=scene.container_animated_objects, unit_name=self.name)
            else:
                if (_w, _h) in scene.visible_area_mask:
                    scene.show_unit(container=scene.container_animated_objects, unit_name=self.name)

    def animate(self, scene, delta):
        if not self.animations_list:
            return False
        ao = scene.animated_objects[self.object_name]
        animation = ao.animations[self.animation_playing]
        root_part_name = ao.parts[0]
        root_part_animation = animation.parts.get(root_part_name)
        self.animation_length = root_part_animation.frames - 1
        if self.animation_frame >= self.animation_length:
            if self.animation_next:
                self.animation_playing = self.animation_next
                self.animation_next = None
                animation = ao.animations[self.animation_playing]
                root_part_animation = animation.parts.get(root_part_name)
                self.animation_length = root_part_animation.frames - 1
            self.animation_frame = 0
        # self.animation_frame = self.animation_frame % (root_part_animation.frames - 1)
        # if self.animation_frame >= root_part_animation.frames:
            # if _Debug:
            #     print(f'restarting unit ({self.name}) animation {self.animation_playing} after frame {self.animation_frame}')
        #     self.animation_frame = 0
            # current_animation = self.animations_list.index(self.animation_playing)
            # # current_animation += 1
            # if current_animation >= len(self.animations_list):
            #     current_animation = 0
            # self.animation_playing = self.animations_list[current_animation]
            # animation = ao.animations[self.animation_playing]
        frame = self.animation_frame
        for part_name in ao.parts:
            if part_name not in animation.parts:
                continue
            part_animation = animation.parts.get(part_name)
            if not part_animation:
                continue
            if part_name not in self.meshes_transforms:
                continue
            r = part_animation.rotation_frames[frame]
            t = part_animation.translation_frames[frame]
            mesh_transform = self.meshes_transforms[part_name]
            translate_mat = Matrix()
            translate_mat.translate(t[0], t[1], t[2])
            mesh_transform.part_translate.matrix = translate_mat
            rotate_mat = Matrix()
            rotate_mat.set(array=mth.quaternion_to_matrix(r[0], r[1], r[2], r[3]))
            mesh_transform.part_rotate.matrix = rotate_mat.inverse()
        self.animation_frame += 1
        return True
