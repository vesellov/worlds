"""
Simulation of one biome on the server: the authoritative positions and movement of the objects.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import os
import io
import json

from PIL import Image

#------------------------------------------------------------------------------

from lib import const
from lib import models
from lib import physics

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
    coefs=[0.5, 0.5, 0.75],
    store_id=2,
)

DEFAULT_HERO_POSITION = (261, 337)

# controls requested for a tick further than that in the future are applied earlier, at that many ticks from now
MAX_INPUT_LEAD = 40

# zonemap.png of the biome says where units can walk: white cells are open, black cells are closed,
# a pixel brighter than that is a white one
ZONE_OPEN_THRESHOLD = 128

# radius in segments of the area around a building where units can not walk
BUILDING_RADIUS = {
    'capital': 0.6,
    'tower': 0.4,
    'flag': 0.15,
}

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
        # one byte per cell of zonemap.png, row by row, see is_open()
        self.zonemap_data = b''
        self.buildings = []
        # [template, store_id] of all models used in the world, clients download them before rendering the scene
        self.templates = []
        self.animations_catalog = {}
        self.action_types_cache = {}
        self.objects = {}
        # unit id -> list of (Controls, slack) received from the client and waiting for their tick
        self.pending_controls = {}
        self.tick_number = 0
        self.next_id = 0

    def load(self):
        buildings_data = self.biome.read_json('buildings.json')
        plants_data = self.biome.read_json('plants.json')
        self.load_heightmap(self.biome.read('heightmap.png'))
        self.load_zonemap(self.biome.read('zonemap.png'))
        self.load_buildings(buildings_data)
        self.load_templates(buildings_data, plants_data)
        with open(os.path.join(self.catalog_path, 'animations.json'), 'rt') as f:
            self.animations_catalog = json.loads(f.read())
        self.map = models.Map(width=self.width, height=self.height, water_level=const.WATER_LEVEL / 100.0)
        self.info = models.World(name=self.biome.id, width=self.width, height=self.height)
        if _Debug:
            print(f'biome {self.biome.id} loaded, map is {self.width}x{self.height} with {len(self.buildings)} buildings and {len(self.templates)} templates')

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

    def load_zonemap(self, zonemap_data):
        im = Image.open(io.BytesIO(zonemap_data)).convert('L')
        if im.size != (self.width, self.height):
            print(f'biome {self.biome.id}: WARNING, zonemap.png is {im.size[0]}x{im.size[1]} but the map is {self.width}x{self.height}, it is scaled')
            im = im.resize((self.width, self.height), Image.NEAREST)
        self.zonemap_data = im.tobytes()

    def is_open(self, w, h):
        """
        Units can walk on the cell when it is white on zonemap.png, black cells are closed.
        """
        return self.zonemap_data[(h % self.height) * self.width + (w % self.width)] >= ZONE_OPEN_THRESHOLD

    def load_buildings(self, buildings_data):
        for building_info in buildings_data:
            radius = BUILDING_RADIUS.get(building_info['k'], 0.3)
            # buildings are placed at the corner of the segment, see scen.Scene.add_land_segment()
            self.buildings.append((float(building_info['x']), float(building_info['y']), radius))

    def load_templates(self, buildings_data, plants_data):
        templates = set()
        for building_info in buildings_data:
            templates.add(building_info['m'])
        # keys of plants.json are "template#texture#parts", see dat.LandData.load_plants_data() on the client
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

    def is_blocked(self, w, h, shift_w, shift_h, ignore=None):
        if not self.is_open(w, h):
            return 'zone'
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
        # the nearest open cell, the whole map is searched when the cells around are closed
        for distance in range(0, max(self.width, self.height) // 2 + 1):
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
        hero.input_tick = self.tick_number
        self.objects[hero.id] = hero
        self.pending_controls[hero.id] = []
        if _Debug:
            print(f'spawned {hero.name} for {owner} at {w},{h}')
        return hero

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
        # the client calculates the movement with the same function, see cli.Prediction
        w, h, shift_w, shift_h, direction, speed = physics.step(
            u.w, u.h, u.shift_w, u.shift_h, u.direction, u.speed, u.move, u.turn, u.target_direction,
            u.max_speed, u.acceleration_value, u.deceleration_value, u.turn_speed,
        )
        u.direction = direction
        # a unit which somehow is already inside of a blocked area must be able to walk out of it
        if speed and self.is_blocked(w, h, shift_w, shift_h, ignore=u) and not self.is_blocked(u.w, u.h, u.shift_w, u.shift_h, ignore=u):
            u.speed = 0.0
        else:
            u.w, u.h, u.shift_w, u.shift_h = w, h, shift_w, shift_h
            u.speed = speed
        u.action = 'run' if u.move else 'idle'
        u.animation = self.animation_for(u)
