import os
import json
import math

from PIL import Image

import const
import models

_Debug = True


DEFAULT_HERO_TEMPLATE = dict(
    model_name='unhuma',
    skin=41,
    hair=0,
    wears=[
        'hadagan brigand pants.thin',
        'hadagan brigand boots.thin',
        'hadagan brigand gloves.thin',
        'hadagan brigand leggins.thick',
        'hadagan brigand helm.thick',
        'hadagan brigand plate.thick',
    ],
    weapon='cheat dagger.bronze',
    coefs=[0.5, 0.5, 0.75],
)

DEFAULT_HERO_POSITION = (261, 337)

# radius in segments of the area around a building where units can not walk
BUILDING_RADIUS = {
    'capital': 0.6,
    'tower': 0.4,
    'flag': 0.15,
}


def normalize_angle(a):
    while a > 360.0:
        a -= 360.0
    while a < 0.0:
        a += 360.0
    return a


def normalize_position(w, h, shift_w, shift_h):
    # same as in unt.Unit.run() before: integer part of the shift moves unit to another segment
    if shift_w > 1.0:
        w += int(shift_w)
        shift_w = float(shift_w) - float(int(shift_w))
    elif shift_w < 0.0:
        w += int(shift_w) - 1
        shift_w = float(shift_w) - float(int(shift_w)) + 1.0
    if shift_h > 1.0:
        h += int(shift_h)
        shift_h = float(shift_h) - float(int(shift_h))
    elif shift_h < 0.0:
        h += int(shift_h) - 1
        shift_h = float(shift_h) - float(int(shift_h)) + 1.0
    return w, h, shift_w, shift_h


def point_line_left_or_right(xp, yp, xa, ya, xb, yb):
    cross_product = (xb - xa) * (yp - ya) - (yb - ya) * (xp - xa)
    if cross_product > 0:
        return -1
    elif cross_product < 0:
        return 1
    return 0


def get_z_in_triangle(x, y, p1, p2, p3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
    B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
    C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    D = -(x1 * (y2 * z3 - y3 * z2) + x2 * (y3 * z1 - y1 * z3) + x3 * (y1 * z2 - y2 * z1))
    if C == 0:
        return (z1 + z2 + z3) / 3.0
    return -(A * x + B * y + D) / C


class Simulation(object):
    """
    Authoritative state of the game world.
    All changes of the world happen here: clients only request actions by sending their controls.
    """

    def __init__(self, root_path):
        self.root_path = root_path
        self.map = None
        self.info = None
        self.elevation_map_data = {}
        self.buildings = []
        self.animations_catalog = {}
        self.action_types_cache = {}
        self.objects = {}
        self.controls = {}
        self.tick_number = 0
        self.next_id = 0

    def load(self):
        self.load_heightmap(os.path.join(self.root_path, 'assets', 'heightmap.png'))
        self.load_buildings(os.path.join(self.root_path, 'assets', 'buildings.json'))
        self.animations_catalog = json.loads(open(os.path.join(self.root_path, 'catalog', 'animations.json'), 'rt').read())
        self.map = models.Map(width=self.width, height=self.height, water_level=const.WATER_LEVEL / 100.0)
        self.info = models.World(name='worlds', width=self.width, height=self.height)
        if _Debug:
            print(f'world loaded, map is {self.width}x{self.height} with {len(self.buildings)} buildings')

    #--- MAP ---

    def elevation_unpack(self, h):
        # same as dat.LandData.elevation_unpack() on the client
        if h > const.INPUT_WATER_LEVEL:
            return pow(h - 18, const.ELEVATION_UNPACK_EXPONENT)
        if h <= 0:
            return -1 * (const.INPUT_WATER_LEVEL - 1) * const.ELEVATION_UNPACK_UNDERWATER_FACTOR
        return (float(h - const.INPUT_WATER_LEVEL) / float(h)) * float(const.ELEVATION_UNPACK_UNDERWATER_FACTOR)

    def load_heightmap(self, heightmap_file_name):
        e_min_unpacked = self.elevation_unpack(1)
        e_max_unpacked = self.elevation_unpack(100)
        unpacked_delta = e_max_unpacked - e_min_unpacked
        im = Image.open(heightmap_file_name).convert('RGB')
        self.width, self.height = im.size
        pixels = im.load()
        for w in range(self.width):
            for h in range(self.height):
                e_unpacked = self.elevation_unpack(float(pixels[w, h][0]))
                self.elevation_map_data[(w, h)] = float(e_unpacked - e_min_unpacked) / unpacked_delta

    def load_buildings(self, buildings_file_name):
        for building_info in json.loads(open(buildings_file_name, 'rt').read()):
            radius = BUILDING_RADIUS.get(building_info['k'], 0.3)
            # buildings are placed at the corner of the segment, see scen.Scene.add_land_segment()
            self.buildings.append((float(building_info['x']), float(building_info['y']), radius))

    def get_elevation(self, w, h):
        return self.elevation_map_data[(w % self.width, h % self.height)]

    def calculate_elevation(self, w, h, shift_w, shift_h):
        # same as scen.Scene.calculate_elevation() on the client, but in the 0.0 - 1.0 elevation scale
        p00 = (0, 0, self.get_elevation(w, h))
        p01 = (0, 1, self.get_elevation(w, h + 1))
        p10 = (1, 0, self.get_elevation(w + 1, h))
        p11 = (1, 1, self.get_elevation(w + 1, h + 1))
        if point_line_left_or_right(shift_w, shift_h, p01[0], p01[1], p10[0], p10[1]) == 1:
            return get_z_in_triangle(shift_w, shift_h, p00, p01, p10)
        return get_z_in_triangle(shift_w, shift_h, p01, p10, p11)

    def is_blocked(self, w, h, shift_w, shift_h, ignore=None):
        if self.calculate_elevation(w, h, shift_w, shift_h) < self.map.water_level:
            return 'water'
        x = w + shift_w
        y = h + shift_h
        for bx, by, radius in self.buildings:
            if (x - bx) ** 2 + (y - by) ** 2 < radius ** 2:
                return 'building'
        for o in self.objects.values():
            if o is ignore or not isinstance(o, models.Unit):
                continue
            if (x - o.w - o.shift_w) ** 2 + (y - o.h - o.shift_h) ** 2 < 0.2 ** 2:
                return 'unit'
        return None

    #--- OBJECTS ---

    def action_types(self, model_name):
        # same as action_types in dat.CatalogData.build_template_data() on the client
        if model_name not in self.action_types_cache:
            action_types = {}
            for action in self.animations_catalog[model_name]['actions']:
                if action['animation_stage'] != 'cycle':
                    continue
                action_type = action['action_type'].split(':')[0]
                action_types.setdefault(action_type, []).append(action['action_name'])
            self.action_types_cache[model_name] = action_types
        return self.action_types_cache[model_name]

    def animation_for(self, unit):
        names = self.action_types(unit.template.model_name).get(unit.action)
        return names[0] if names else None

    def find_free_position(self, w, h):
        for distance in range(0, 10):
            for dw in range(-distance, distance + 1):
                for dh in range(-distance, distance + 1):
                    if max(abs(dw), abs(dh)) != distance:
                        continue
                    if not self.is_blocked(w + dw, h + dh, 0.5, 0.5):
                        return w + dw, h + dh
        return w, h

    def spawn_hero(self, owner, template=None):
        self.next_id += 1
        w, h = self.find_free_position(*DEFAULT_HERO_POSITION)
        hero = models.Hero(
            id=self.next_id,
            name=f'hero{self.next_id}',
            owner=owner,
            template=models.Template(**(template or DEFAULT_HERO_TEMPLATE)),
            w=w,
            h=h,
        )
        hero.animation = self.animation_for(hero)
        self.objects[hero.id] = hero
        self.controls[hero.id] = models.Controls()
        if _Debug:
            print(f'spawned {hero.name} for {owner} at {w},{h}')
        return hero

    def remove_object(self, object_id):
        self.controls.pop(object_id, None)
        return self.objects.pop(object_id, None)

    def set_controls(self, unit_id, controls):
        if unit_id in self.controls:
            self.controls[unit_id] = controls

    #--- GAME LOOP ---

    def tick(self):
        """
        Process all requests of all clients, returns list of objects which were changed.
        """
        self.tick_number += 1
        changed = []
        for o in list(self.objects.values()):
            if isinstance(o, models.Unit):
                before = o.to_state()
                self.run_unit(o, self.controls.get(o.id) or models.Controls())
                if o.to_state() != before:
                    changed.append(o)
        return changed

    def run_unit(self, u, controls):
        # turning, same as it was in rend.Renderer.on_run_units() and hero.Hero.turn() on the client
        if controls.target_direction is not None:
            direction_diff = controls.target_direction - u.direction
            if direction_diff > 180.0:
                direction_diff -= 360.0
            if direction_diff < -180.0:
                direction_diff += 360.0
            turn = min(u.turn_speed, abs(direction_diff))
            u.direction = normalize_angle(u.direction + (-turn if direction_diff > 0.0 else turn))
        elif controls.turn:
            u.direction = normalize_angle(u.direction + (u.turn_speed if controls.turn > 0 else -u.turn_speed))
        # acceleration, same as it was in hero.Hero.move() on the client
        if controls.move > 0:
            acceleration_up, acceleration_down = u.acceleration_value, 0.0
        elif controls.move < 0:
            acceleration_up, acceleration_down = -u.acceleration_value, 0.0
        else:
            acceleration_up, acceleration_down = 0.0, u.acceleration_value * 5
        # speed, same as it was in unt.Unit.run() on the client
        speed = u.speed + acceleration_up
        if acceleration_up >= 0:
            speed = min(speed, u.max_speed)
        else:
            speed = max(speed, -u.max_speed)
        if speed > 0:
            speed = max(0.0, speed - acceleration_down)
        elif speed < 0:
            speed = min(0.0, speed + acceleration_down)
        u.speed = round(speed, 6)
        if u.speed:
            new_position = normalize_position(
                u.w,
                u.h,
                u.shift_w + u.speed * math.cos(math.radians(u.direction)),
                u.shift_h + u.speed * math.sin(math.radians(u.direction)),
            )
            # a unit which somehow is already inside of a blocked area must be able to walk out of it
            if self.is_blocked(*new_position, ignore=u) and not self.is_blocked(u.w, u.h, u.shift_w, u.shift_h, ignore=u):
                u.speed = 0.0
            else:
                u.w, u.h, u.shift_w, u.shift_h = new_position
        u.action = 'run' if controls.move else 'idle'
        u.animation = self.animation_for(u)
