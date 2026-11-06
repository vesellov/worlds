"""
Simulation of one biome on the server: the authoritative positions and movement of the objects.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import os
import io
import json
import math
import random

from PIL import Image

#------------------------------------------------------------------------------

from lib import const
from lib import flora
from lib import models
from lib import physics
from lib import zone
from server import showcase

#------------------------------------------------------------------------------

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
    coefs=[1.0, 1.0, 1.0],
    scale=[1.0, 1.0, 1.0],
    store_id=2,
)

DEFAULT_HERO_POSITION = (261, 337)

# controls requested for a tick further than that in the future are applied earlier, at that many ticks from now
MAX_INPUT_LEAD = 40

# distance in segments between the centers of two units of the usual size, they do not walk closer
UNIT_KEEP_DISTANCE = 0.2

# animations played for the action of the unit, the first one its figure has: not every figure can run
ACTION_ANIMATIONS = {
    'run': ('run', 'walk', 'idle'),
}

# neutral units added by the clients: not more than that in one biome, they appear that many segments
# in front of the hero of the client, up to NEUTRAL_UNIT_ANGLE degrees to the left or to the right of its direction,
# so the next one does not appear right on the previous one, see Simulation.add_neutral_unit()
MAX_NEUTRAL_UNITS = 100
NEUTRAL_UNIT_DISTANCE = 2.0
NEUTRAL_UNIT_ANGLE = 60.0

#------------------------------------------------------------------------------


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


def push_out_of_circle(dx, dy, radius2, heading=None):
    """
    (dx, dy) goes from the center of a round obstacle to the position. Returns the shortest movement
    which takes the position out of the circle, or None when it is outside already.
    """
    distance2 = dx * dx + dy * dy
    if distance2 >= radius2:
        return None
    distance = math.sqrt(distance2)
    if distance == 0.0:
        if not heading:
            return math.sqrt(radius2) + physics.SLIDE_GAP, 0.0
        # right in the center, back to where the unit came from
        dx, dy, distance = -heading[0], -heading[1], 1.0
    k = (math.sqrt(radius2) - distance + physics.SLIDE_GAP) / distance
    return dx * k, dy * k

#------------------------------------------------------------------------------


class Simulation(object):
    """
    Authoritative state of one biome of the game world, see server.biomes.
    All changes of the world happen here: clients only request actions by sending their controls.
    """

    def __init__(self, biome, catalog_path):
        # server.biomes.BiomeArchive, files of the biome are read from its archive
        self.biome = biome
        self.catalog_path = catalog_path
        self.map = None
        self.info = None
        self.elevation_map_data = {}
        # lib.zone.ZoneMap: mountains and deep water of zonemap.png, trees and buildings where units can not walk
        self.zonemap = None
        # [template, store_id] of all models used in the world, clients download them before rendering the scene
        self.templates = []
        self.animations_catalog = {}
        self.weapons_catalog = {}
        self.action_types_cache = {}
        # showcase.Looks, random looks of the neutral units, loaded when the first one is added
        self.looks = None
        self.neutral_units_count = 0
        self.objects = {}
        # holds various info about particular land segment, key is cell coordinates tuple [(w,h)]
        self.land_store = {}
        # unit id -> list of (Controls, slack) received from the client and waiting for their tick
        self.pending_controls = {}
        self.tick_number = 0
        self.next_id = 0

    def load(self):
        buildings_data = self.biome.read_json('buildings.json')
        plants_data = flora.decode(self.biome.read('plants.bin'))
        self.load_heightmap(self.biome.read('heightmap.png'))
        # trees and buildings are added to the closed land of zonemap.png, units walk around them as around any closed area,
        # the client builds the same map, see dat.LandData.load_zonemap_file()
        self.zonemap = zone.ZoneMap.build(self.biome.read('zonemap.png'), self.width, self.height, plants_data, buildings_data)
        self.load_templates(buildings_data, plants_data)
        with open(os.path.join(self.catalog_path, 'animations.json'), 'rt') as f:
            self.animations_catalog = json.loads(f.read())
        with open(os.path.join(self.catalog_path, 'weapons.json'), 'rt') as f:
            self.weapons_catalog = json.loads(f.read())
        self.map = models.Map(width=self.width, height=self.height, water_level=const.WATER_LEVEL / 100.0)
        self.info = models.World(name=self.biome.id, width=self.width, height=self.height)
        if _Debug:
            print(f'biome {self.biome.id} loaded, map is {self.width}x{self.height}, zonemap has {self.zonemap.scale} pixels per segment, {len(self.templates)} templates')

    #--- MAP ---

    def elevation_unpack(self, h):
        # same as dat.LandData.elevation_unpack() on the client
        if h > const.INPUT_WATER_LEVEL:
            return pow(h - 18, const.ELEVATION_UNPACK_EXPONENT)
        if h <= 0:
            return -1 * (const.INPUT_WATER_LEVEL - 1) * const.ELEVATION_UNPACK_UNDERWATER_FACTOR
        return (float(h - const.INPUT_WATER_LEVEL) / float(h)) * float(const.ELEVATION_UNPACK_UNDERWATER_FACTOR)

    def load_heightmap(self, heightmap_data):
        e_min_unpacked = self.elevation_unpack(1)
        e_max_unpacked = self.elevation_unpack(100)
        unpacked_delta = e_max_unpacked - e_min_unpacked
        im = Image.open(io.BytesIO(heightmap_data)).convert('RGB')
        self.width, self.height = im.size
        pixels = im.load()
        for w in range(self.width):
            for h in range(self.height):
                e_unpacked = self.elevation_unpack(float(pixels[w, h][0]))
                self.elevation_map_data[(w, h)] = float(e_unpacked - e_min_unpacked) / unpacked_delta

    def load_templates(self, buildings_data, plants_data):
        templates = set()
        for building_info in buildings_data:
            templates.add(building_info['m'])
        # keys of the plants are "template#texture#parts", see dat.LandData.load_plants_data() on the client
        for plant_key in plants_data.keys():
            templates.add(plant_key.split('#')[0])
        # buildings and plants are taken from the default store, every hero says where its model is
        world_templates = set((t.lower(), models.DEFAULT_STORE_ID) for t in templates)
        world_templates.add((DEFAULT_HERO_TEMPLATE['model_name'].lower(), DEFAULT_HERO_TEMPLATE.get('store_id', models.DEFAULT_STORE_ID)))
        self.templates = [list(t) for t in sorted(world_templates)]

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

    def is_blocked(self, w, h, shift_w, shift_h, ignore=None, size=1.0):
        """
        A unit can not stand at the position: the zone is closed there or another unit is too close.
        size is the result of zone.unit_size(), ignore is the unit itself.
        """
        return physics.is_blocked(self.zonemap, w + shift_w, h + shift_h, zone.unit_radius(size), self.units_obstacle(ignore, size))

    def units_obstacle(self, u, size):
        """
        Returns the function which finds other units around a position for physics.move(): it returns the push
        out of the nearest of them, or None. Only the server knows where all units are, so the client
        does not check them and finds out from the next state when its hero was blocked by another unit.
        """

        def _obstacle(x, y, heading, from_x, from_y):
            for o in self.objects.values():
                if o is u or not isinstance(o, models.Unit):
                    continue
                # each of the two units keeps half of the distance
                keep = UNIT_KEEP_DISTANCE * (size + zone.unit_size(o)) / 2.0
                push = push_out_of_circle(x - o.w - o.shift_w, y - o.h - o.shift_h, keep * keep, heading)
                if push:
                    return push
            return None

        return _obstacle

    #--- OBJECTS ---

    def weapon_type(self, weapon):
        # weapon of the template is "<weapon name>.<material name>"
        if not weapon:
            return None
        return self.weapons_catalog[weapon.split('.')[0]]['type']

    def action_types(self, model_name, weapon_type=None):
        # same as action_types in dat.CatalogData.build_template_data() on the client, but only the animations
        # which the client loads for a figure with that weapon
        key = (model_name, weapon_type)
        if key not in self.action_types_cache:
            action_types = {}
            for action in self.animations_catalog[model_name]['actions']:
                if action['animation_stage'] != 'cycle':
                    continue
                weapons = action['weapons'].split(',')
                if action['weapons'] and weapons != ['all', ] and weapon_type not in weapons:
                    continue
                action_type = action['action_type'].split(':')[0]
                action_types.setdefault(action_type, []).append(action['action_name'])
            self.action_types_cache[key] = action_types
        return self.action_types_cache[key]

    def animation_for(self, unit):
        action_types = self.action_types(unit.template.model_name, self.weapon_type(unit.template.weapon))
        for action_type in ACTION_ANIMATIONS.get(unit.action, (unit.action, )):
            names = action_types.get(action_type)
            if names:
                return names[0]
        return None

    def find_free_position(self, w, h, size=1.0):
        # the nearest open cell for a unit of that size, see zone.unit_size(), the whole map is searched when the cells around are closed
        for distance in range(0, max(self.width, self.height) // 2 + 1):
            for dw in range(-distance, distance + 1):
                for dh in range(-distance, distance + 1):
                    if max(abs(dw), abs(dh)) != distance:
                        continue
                    if not self.is_blocked(w + dw, h + dh, 0.5, 0.5, size=size):
                        return w + dw, h + dh
        return w, h

    def spawn_hero(self, owner, template=None):
        self.next_id += 1
        hero = models.Hero(
            id=self.next_id,
            name=f'hero{self.next_id}',
            owner=owner,
            template=models.Template(**(template or DEFAULT_HERO_TEMPLATE)),
            w=0,
            h=0,
        )
        hero.w, hero.h = self.find_free_position(*DEFAULT_HERO_POSITION, size=zone.unit_size(hero))
        hero.animation = self.animation_for(hero)
        hero.input_tick = self.tick_number
        self.objects[hero.id] = hero
        self.pending_controls[hero.id] = []
        if _Debug:
            print(f'spawned {hero.name} for {owner} at {hero.w},{hero.h}')
        return hero

    def add_neutral_unit(self, near_id):
        """
        Adds a unit of a random figure with a random look in front of the object near_id, the unit runs in a circle:
        it always moves forward and turns, the clients calculate its movement with the same rules, see lib.physics.
        Returns the unit or None.
        """
        near = self.objects.get(near_id)
        if not near or self.neutral_units_count >= MAX_NEUTRAL_UNITS:
            return None
        if self.looks is None:
            self.looks = showcase.Looks(self.catalog_path)
        self.next_id += 1
        unit = models.Unit(
            id=self.next_id,
            name=f'unit{self.next_id}',
            template=models.Template(**self.looks.random_template()),
            w=0,
            h=0,
            direction=float(random.randint(0, 359)),
            move=1,
            turn=1,
        )
        angle = math.radians(near.direction + random.uniform(-NEUTRAL_UNIT_ANGLE, NEUTRAL_UNIT_ANGLE))
        x = near.w + near.shift_w + NEUTRAL_UNIT_DISTANCE * math.cos(angle)
        y = near.h + near.shift_h + NEUTRAL_UNIT_DISTANCE * math.sin(angle)
        unit.w, unit.h = self.find_free_position(int(math.floor(x)) % self.width, int(math.floor(y)) % self.height, size=zone.unit_size(unit))
        unit.action = 'run'
        unit.animation = self.animation_for(unit)
        self.objects[unit.id] = unit
        self.neutral_units_count += 1
        if _Debug:
            print(f'added neutral {unit.name} {unit.template.model_name} at {unit.w},{unit.h}, {self.neutral_units_count} in biome {self.biome.id}')
        return unit

    def remove_object(self, object_id):
        self.pending_controls.pop(object_id, None)
        return self.objects.pop(object_id, None)

    def set_controls(self, unit_id, controls):
        """
        Client asks to apply the controls at the tick controls.tick, so it knows in advance how its hero is going to move.
        Controls which came too late are applied at the next tick, the client finds it out from Unit.input_slack.
        """
        pending = self.pending_controls.get(unit_id)
        if pending is None:
            return
        next_tick = self.tick_number + 1
        slack = controls.tick - next_tick
        tick = max(next_tick, min(controls.tick, self.tick_number + MAX_INPUT_LEAD))
        # controls are applied in the order they were sent
        if pending:
            tick = max(tick, pending[-1][0].tick)
        controls.tick = tick
        pending.append((controls, slack))

    def apply_controls(self, u):
        pending = self.pending_controls.get(u.id)
        while pending and pending[0][0].tick <= self.tick_number:
            controls, slack = pending.pop(0)
            u.move = controls.move
            u.turn = controls.turn
            u.target_direction = controls.target_direction
            u.input_seq = controls.seq
            u.input_tick = self.tick_number
            u.input_slack = slack

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
                self.apply_controls(o)
                self.run_unit(o)
                if o.to_state() != before:
                    changed.append(o)
        return changed

    def run_unit(self, u):
        # the client calculates the movement with the same function and the same zonemap, see cli.Prediction,
        # only other units are not known there
        size = zone.unit_size(u)
        u.w, u.h, u.shift_w, u.shift_h, u.direction, u.speed = physics.move(
            self.zonemap, u.w, u.h, u.shift_w, u.shift_h, u.direction, u.speed, u.move, u.turn, u.target_direction,
            u.max_speed, u.acceleration_value, u.deceleration_value, u.turn_speed,
            zone.unit_radius(size), self.units_obstacle(u, size),
        )
        u.action = 'run' if u.move else 'idle'
        u.animation = self.animation_for(u)
