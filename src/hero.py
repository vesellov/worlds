import const
import mth

_Debug = True


class Hero(object):

    def __init__(self, scene, model_name, skin=0, hair=None, wears=[], weapon=None, texture=None):
        self.scene = scene
        self.unit_name = None
        self.model_name = model_name
        self.skin = skin
        self.hair = hair
        self.wears = wears
        self.weapon = weapon
        self.texture = texture

    def create_unit(self, map_w, map_h, elevation_correction=None, coefs=[0.5, 0.5, 0.75]):
        template = self.scene.catalog.build_template_data(
            model_name=self.model_name,
            skin=self.skin,
            hair=self.hair,
            wears=self.wears,
            weapon=self.weapon,
            texture=self.texture,
        )
        selected_parts = [p.replace('.hidden', '') for p in template['parts']]
        hidden_parts = [p.replace('.hidden', '') for p in template['parts'] if p.endswith('.hidden')]        
        unit = self.scene.place_animated_unit_on_land(
            template=template['model_name'],
            coefs=coefs,
            # coefs=[0.15, 0.11, 0.81],
            # coefs=[0.5, 0.5, 0.75],
            # coefs=[1.0, 1.0, 1.0],
            # scale=scale,
            map_w=map_w,
            map_h=map_h,
            shift_w=0.5,
            shift_h=0.5,
            direction=0, # random.randint(0, 360),
            elevation_correction=elevation_correction,
            selected_parts=selected_parts,
            hidden_parts=hidden_parts,
            textures=template['textures'] if not self.texture else {'*': self.texture},
            single_texture=False if self.texture else True,
            selected_animations=template['animations'],
        )
        unit.animations_list = template['animations']
        unit.action_types = template['action_types']
        unit.max_speed = 0.04
        unit.acceleration_value = 0.002
        unit.turn_speed = 5.0
        unit.animation_playing = unit.action_types['idle'][0]['name']
        unit.animation_next = None
        unit.elevation_correction = elevation_correction
        unit.coefs = coefs
        self.unit_name = unit.name
        if _Debug:
            print(f'created hero unit {self.unit_name} from generated template for model {self.model_name}')
        return unit

    def get_unit(self):
        return self.scene.units.get(self.unit_name)

    def move(self, forward=False, backward=False):
        u = self.get_unit()
        if not u:
            return
        if forward:
            u.acceleration_up = u.acceleration_value
            u.acceleration_down = 0.0
            u.is_walking = True
        elif backward:
            u.acceleration_up = -u.acceleration_value
            u.acceleration_down = 0.0
            u.is_walking = True
        else:
            u.acceleration_up = 0.0
            u.acceleration_down = u.acceleration_value * 5
            u.is_walking = False
        if u.is_walking:
            if u.animation_playing != u.action_types['run'][0]['name']:
                u.animation_frame = 0
            u.animation_playing = u.action_types['run'][0]['name']
        else:
            if u.animation_playing != u.action_types['idle'][0]['name']:
                if u.animation_frame > u.animation_length * 0.7:
                    u.animation_next = u.action_types['idle'][0]['name']
                else:
                    u.animation_playing = u.action_types['idle'][0]['name']
                    u.animation_frame = 0
            else:
                u.animation_playing = u.action_types['idle'][0]['name']
                u.animation_frame = 0

    def turn(self, left=False, right=False):
        u = self.get_unit()
        if not u:
            return
        u.direction += u.turn_speed if left else -u.turn_speed if right else 0.0
