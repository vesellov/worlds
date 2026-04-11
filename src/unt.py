import math
import random

from kivy.graphics.transformation import Matrix  # @UnresolvedImport

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
            self.max_speed = float(random.randint(1, 50)) / 1000.0
            self.acceleration = float(random.randint(1, 5)) / 1000.0
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

    def animate(self, scene, delta):
        if not self.animations_list:
            return False
        ao = scene.animated_objects[self.object_name]
        animation = ao.animations[self.animation_playing]
        root_part_name = ao.parts[0]
        root_part_animation = animation.parts.get(root_part_name)
        if self.animation_frame >= root_part_animation.frames:
            # if _Debug:
            #     print(f'restarting unit ({self.name}) animation {self.animation_playing} after frame {self.animation_frame}')
            self.animation_frame = 0
            current_animation = self.animations_list.index(self.animation_playing)
            # current_animation += 1
            if current_animation >= len(self.animations_list):
                current_animation = 0
            self.animation_playing = self.animations_list[current_animation]
            animation = ao.animations[self.animation_playing]
        frame = self.animation_frame
        for part_name in ao.parts:
            if part_name not in animation.parts:
                continue
            part_animation = animation.parts.get(part_name)
            if not part_animation:
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
