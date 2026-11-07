"""
Client side view of the unit controlled by the server: the 3D model, its parts and textures.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

from lib import const
from lib import figures
from lib import mth
from lib import models
from client import loader

#------------------------------------------------------------------------------


class Hero(object):
    """
    Client side view of a unit which is controlled by the server: builds 3D model of the unit from the template
    received from the server and applies the state of the unit received from the server.
    """

    def __init__(self, scene, model_name, skin=0, hair=None, wears=[], weapon=None, texture=None, store_id=None):
        self.scene = scene
        # the model and its textures are taken from that store, see loader.StoreLoader
        self.store_id = store_id or models.DEFAULT_STORE_ID
        self.template_data = None
        self.unit_name = None
        self.model_name = model_name
        self.skin = skin
        self.hair = hair
        self.wears = wears
        self.weapon = weapon
        self.texture = texture

    def build_template(self):
        if self.template_data is None:
            self.template_data = self.scene.catalog.build_template_data(
                model_name=self.model_name,
                skin=self.skin,
                hair=self.hair,
                wears=self.wears,
                weapon=self.weapon,
                texture=self.texture,
            )
        return self.template_data

    def required_files(self):
        """
        Files of the store which must be downloaded before the unit can be created, see loader.missing_files().
        """
        template = self.build_template()
        textures = template['textures'].values() if not self.texture else [self.texture]
        return loader.missing_files(
            templates=[(template['model_name'], self.store_id)],
            textures=[(t, self.store_id) for t in textures],
        )

    def create_unit(self, map_w, map_h, elevation_correction=None, coefs=[0.5, 0.5, 0.75], scale=[1.0, 1.0, 1.0], shift_w=0.5, shift_h=0.5, direction=0):
        template = self.build_template()
        selected_parts = [p.replace('.hidden', '') for p in template['parts']]
        hidden_parts = [p.replace('.hidden', '') for p in template['parts'] if p.endswith('.hidden')]
        unit = self.scene.place_animated_unit_on_land(
            template=template['model_name'],
            coefs=coefs,
            scale=scale,
            map_w=map_w,
            map_h=map_h,
            shift_w=shift_w,
            shift_h=shift_h,
            direction=direction,
            elevation_correction=elevation_correction,
            selected_parts=selected_parts,
            hidden_parts=hidden_parts,
            textures=template['textures'] if not self.texture else {'*': self.texture},
            single_texture=False if self.texture else True,
            selected_animations=template['animations'],
            store_id=self.store_id,
        )
        unit.animations_list = template['animations']
        unit.action_types = template['action_types']
        unit.animation_playing = unit.action_types['idle'][0]['name']
        unit.animation_next = None
        if elevation_correction is None and not figures.is_flying(self.model_name):
            # the lowest point of the figure stands on the ground, flying figures stay as high as they fly
            elevation_correction = self.scene.animated_objects[unit.object_name].ground_offset
        unit.elevation_correction = elevation_correction
        unit.coefs = coefs
        unit.scale = scale
        self.unit_name = unit.name
        if _Debug:
            print(f'created hero unit {self.unit_name} from generated template for model {self.model_name}')
        return unit

    def get_unit(self):
        return self.scene.units.get(self.unit_name)

    def label_text(self):
        """
        Name of the figure and of its texture, shown above the unit, see scr.UnitLabels.
        """
        texture = self.texture or self.build_template()['textures'].get('*') or ''
        # the layer of the texture after ":" is not interesting
        return f'{self.model_name}\n{texture.split(":")[0]}'.strip()

    def label_height(self):
        """
        Height above the root part of the unit where its label is shown, in the units of the scene.
        """
        u = self.get_unit()
        if not u:
            return 0.0
        return self.scene.animated_objects[u.object_name].top_offset

    def set_animation(self, animation_name):
        """
        Server decides which animation the unit is playing, client only plays frames of it.
        """
        u = self.get_unit()
        if not u or not animation_name:
            return
        if u.animation_playing == animation_name:
            u.animation_next = None
            return
        if animation_name not in self.scene.animated_objects[u.object_name].animations:
            if _Debug:
                print(f'animation {animation_name} is not loaded for unit {u.name}')
            return
        idle_animation_name = u.action_types['idle'][0]['name'] if 'idle' in u.action_types else None
        if animation_name == idle_animation_name and u.animation_frame > u.animation_length * 0.7:
            # let the current animation cycle finish smoothly before switching to idle
            u.animation_next = animation_name
        else:
            u.animation_playing = animation_name
            u.animation_frame = 0
            u.animation_next = None
