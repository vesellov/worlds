import const
import mth


class Hero(object):

    def __init__(self, scene, model_name, skin=0, hair=None, weapon=None, wears=[]):
        self.scene = scene
        self.unit_name = None
        self.model_name = model_name
        self.skin = skin
        self.hair = hair
        self.weapon = weapon
        self.wears = wears

    def create_unit(self):
        template = self.scene.catalog.build_template_data(
            model_name=self.model_name,
            skin=self.skin,
            hair=self.hair,
            wears=self.wears,
            weapon=self.weapon,
        )
        selected_parts = [p.replace('.hidden', '') for p in template['parts']]
        hidden_parts = [p.replace('.hidden', '') for p in template['parts'] if p.endswith('.hidden')]        
        unit = self.scene.place_animated_unit_on_land(
            template=template['model_name'],
            coefs=[0.5, 0.5, 0.5],
            # scale=scale,
            map_w=140,
            map_h=244,
            shift_w=0.5,
            shift_h=0.5,
            direction=0, # random.randint(0, 360),
            # elevation_correction=-5.0,
            selected_parts=selected_parts,
            hidden_parts=hidden_parts,
            textures=template['textures'],
            single_texture=True,
            selected_animations=template['animations'],
        )
        unit.action_types = template['action_types']
        unit.max_speed = 0.007
        unit.animation_playing = unit.action_types['idle'][0]
        self.unit_name = unit.name
        return unit

    def get_unit(self):
        return self.scene.units.get(self.unit_name)

    def move(self, forward=False, backward=False):
        u = self.get_unit()
        if not u:
            return
        if forward:
            u.acceleration_up = 0.0005
            u.acceleration_down = 0.0
            u.is_walking = True
        elif backward:
            u.acceleration_up = -0.0005
            u.acceleration_down = 0.0
            u.is_walking = True
        else:
            u.acceleration_up = 0.0
            u.acceleration_down = 0.001
            u.is_walking = False

    def turn(self, left=False, right=False):
        u = self.get_unit()
        if not u:
            return
        turn_speed = 2.0
        u.direction += turn_speed if left else -turn_speed if right else 0.0
