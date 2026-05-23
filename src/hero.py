import const
import mth


class Hero(object):

    def __init__(self):
        self.unit_name = None
        self.race = None
        self.culture = None
        self.weapon = None
        self.armor = None

    def create_unit(self):
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
        return unit
