"""
Simulation of one biome on the server: the authoritative positions and movement of the objects.
"""

#------------------------------------------------------------------------------

_Debug = True

#------------------------------------------------------------------------------

import os
import io
import math
import random

from PIL import Image

#------------------------------------------------------------------------------

from lib import const
from lib import dress
from lib import figures
from lib import models
from lib import physics
from lib import plants
from lib import stores
from lib import zone
from server import behavior
from server import roads

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

# heroes appear that many segments from the first capital of the biome, or in the middle of the map when it has no capitals
HERO_START_SHIFT = (0, 3)

# controls requested for a tick further than that in the future are applied earlier, at that many ticks from now
MAX_INPUT_LEAD = 40

# animations played for the action of the unit, the first one its figure has: not every figure can run
ACTION_ANIMATIONS = {
    'run': ('run', 'walk', 'idle'),
    'walk': ('walk', 'run', 'idle'),
}

# neutral units added by the clients: not more than that in one biome. They appear at the first capital
# of the biome and walk along the roads to the second one, see Simulation.add_neutral_unit().
# In a biome without roads or with less than two capitals they appear that many segments in front of the hero of the client,
# up to NEUTRAL_UNIT_ANGLE degrees to the left or to the right of its direction, so the next one does not appear
# right on the previous one, and walk around in a circle of NEUTRAL_CIRCLE_RADIUS segments, see behavior.UnitWalkInCircle
MAX_NEUTRAL_UNITS = 100
NEUTRAL_UNIT_DISTANCE = 2.0
NEUTRAL_UNIT_ANGLE = 60.0
NEUTRAL_CIRCLE_RADIUS = 1.5

# when the server starts one unit of every figure of every store is added to every biome, see Simulation.add_showcase_units():
# they stand at equal distances along all roads of the biome, or in rows that many segments apart around the place
# where the heroes appear when the biome has no roads, and walk around in a small circle of SHOWCASE_CIRCLE_RADIUS segments
SHOWCASE_SPACING = 3.0
SHOWCASE_CIRCLE_RADIUS = 1.0

# two moving units which are going to touch each other in less than that many ticks turn aside and slow down,
# see Simulation.avoidance()
AVOID_HORIZON = 12.0

# units turn aside a bit earlier than they touch: the space between them, in segments
AVOID_MARGIN = 0.03

# the part of the speed towards another unit which a unit keeps going aside, the rest is lost:
# the units slow down while they go around each other
AVOID_SIDE = 0.8

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


def collision_time(px, py, vx, vy, reach):
    """
    (px, py) goes from one unit to another one, (vx, vy) is how the second unit moves relatively to the first one
    every tick. Returns in how many ticks the distance between them becomes reach, 0.0 if they are closer already
    and still come closer, None when they do not come that close.
    """
    b = px * vx + py * vy
    if b >= 0.0:
        # they go apart
        return None
    c = px * px + py * py - reach * reach
    if c <= 0.0:
        return 0.0
    a = vx * vx + vy * vy
    discriminant = b * b - a * c
    if discriminant < 0.0:
        return None
    return (-b - math.sqrt(discriminant)) / a

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
        # store id -> animations.json and weapons.json of the store, the template of the unit says its store, see lib.stores
        self.animations_catalog = {}
        self.weapons_catalog = {}
        self.action_types_cache = {}
        # (model name, weapon, action, size) -> speed, see unit_speed()
        self.unit_speeds_cache = {}
        # store id -> dress.Looks, random looks of the neutral units, loaded when the first one of the store is added
        self.looks = {}
        self.neutral_units_count = 0
        # figures of the neutral units go one by one in the order of their names, see add_neutral_unit()
        self.next_figure_index = 0
        # unit id -> behavior.UnitBehavior of the neutral units, units which are controlled by nobody
        self.behaviors = {}
        # number of the capital ("i" of buildings.json) -> (x, y) of the capitals of the biome, see nearest_capital()
        self.capitals = {}
        # roads.Roads of roads.json, None when the biome has no roads
        self.roads = None
        self.objects = {}
        # holds various info about particular land segment, key is cell coordinates tuple [(w,h)]
        self.land_store = {}
        # unit id -> list of (Controls, slack) received from the client and waiting for their tick
        self.pending_controls = {}
        self.tick_number = 0
        self.next_id = 0
        # (w, h) where the heroes appear, see HERO_START_SHIFT
        self.start_position = None

    def load(self):
        buildings_data = self.biome.read_json('buildings.json')
        plants_data = plants.decode(self.biome.read('plants.bin'))
        self.load_heightmap(self.biome.read('heightmap.png'))
        # trees and buildings are added to the closed land of zonemap.png, units walk around them as around any closed area,
        # the client builds the same map, see dat.LandData.load_zonemap_file()
        self.zonemap = zone.ZoneMap.build(self.biome.read('zonemap.png'), self.width, self.height, plants_data, buildings_data)
        self.load_templates(buildings_data, plants_data)
        self.start_position = self.find_start_position(buildings_data)
        self.capitals = dict((b['i'], (float(b['x']), float(b['y']))) for b in buildings_data if b['k'] == 'capital')
        # biomes built before roads.json was added have no roads
        if self.biome.has('roads.json'):
            self.roads = roads.Roads(self.biome.read_json('roads.json'), self.width, self.height)
        self.animations_catalog = stores.read(os.path.join(self.catalog_path, 'animations.json'))
        self.weapons_catalog = stores.read(os.path.join(self.catalog_path, 'weapons.json'))
        self.map = models.Map(width=self.width, height=self.height, water_level=const.WATER_LEVEL / 100.0)
        self.info = models.World(name=self.biome.id, width=self.width, height=self.height)
        if _Debug:
            print(f'biome {self.biome.id} loaded, map is {self.width}x{self.height}, zonemap has {self.zonemap.scale} pixels per segment, {len(self.templates)} templates, {len(self.roads or [])} roads')

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

    def find_start_position(self, buildings_data):
        capitals = sorted((b for b in buildings_data if b['k'] == 'capital'), key=lambda b: b['i'])
        if not capitals:
            return self.width // 2, self.height // 2
        return (int(capitals[0]['x']) + HERO_START_SHIFT[0]) % self.width, (int(capitals[0]['y']) + HERO_START_SHIFT[1]) % self.height

    def nearest_capital(self, x, y):
        best = None
        best_distance2 = None
        for cx, cy in self.capitals.values():
            dx = behavior.nearest_shift(x, cx, self.width)
            dy = behavior.nearest_shift(y, cy, self.height)
            distance2 = dx * dx + dy * dy
            if best is None or distance2 < best_distance2:
                best, best_distance2 = (cx, cy), distance2
        return best

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
                # units do not overlap: the distance between their centers is not smaller than their two radiuses
                keep = zone.unit_radius(size) + zone.unit_radius(zone.unit_size(o))
                # the map is wrapped around, units at the opposite edges of the map are next to each other
                dx = -behavior.nearest_shift(x, o.w + o.shift_w, self.width)
                dy = -behavior.nearest_shift(y, o.h + o.shift_h, self.height)
                push = push_out_of_circle(dx, dy, keep * keep, heading)
                if push:
                    return push
            return None

        return _obstacle

    #--- OBJECTS ---

    def weapon_type(self, weapon, store_id=None):
        # weapon of the template is "<weapon name>.<material name>"
        if not weapon:
            return None
        return stores.of(self.weapons_catalog, store_id)[weapon.split('.')[0]]['type']

    def action_types(self, model_name, weapon_type=None, store_id=None):
        # same as action_types in dat.CatalogData.build_template_data() on the client, but only the animations
        # which the client loads for a figure with that weapon
        key = (model_name, weapon_type, store_id)
        if key not in self.action_types_cache:
            action_types = {}
            for action in stores.of(self.animations_catalog, store_id)[model_name]['actions']:
                if action['animation_stage'] != 'cycle':
                    continue
                weapons = action['weapons'].split(',')
                if action['weapons'] and weapons != ['all', ] and weapon_type not in weapons:
                    continue
                action_type = action['action_type'].split(':')[0]
                action_types.setdefault(action_type, []).append(action['action_name'])
            self.action_types_cache[key] = action_types
        return self.action_types_cache[key]

    def animation_for(self, unit, action=None):
        # the animation the unit plays for that action, the current action of the unit by default
        action = action or unit.action
        store_id = unit.template.store_id
        action_types = self.action_types(unit.template.model_name, self.weapon_type(unit.template.weapon, store_id), store_id)
        for action_type in ACTION_ANIMATIONS.get(action, (action, )):
            names = action_types.get(action_type)
            if names:
                return names[0]
        return None

    def unit_speed(self, unit, action):
        """
        Segments per tick the unit moves while it plays the animation of that action ('walk' or 'run'),
        a zombie is slower than a tiger, see lib.figures.
        """
        size = zone.unit_size(unit)
        key = (unit.template.model_name, unit.template.weapon, action, size, unit.template.store_id)
        if key not in self.unit_speeds_cache:
            animations = stores.of(self.animations_catalog, unit.template.store_id)
            self.unit_speeds_cache[key] = figures.max_speed(animations, unit.template.model_name, self.animation_for(unit, action), size)
        return self.unit_speeds_cache[key]

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
        hero.w, hero.h = self.find_free_position(*self.start_position, size=zone.unit_size(hero))
        hero.animation = self.animation_for(hero)
        hero.max_speed = self.unit_speed(hero, 'run')
        hero.input_tick = self.tick_number
        self.objects[hero.id] = hero
        self.pending_controls[hero.id] = []
        if _Debug:
            print(f'spawned {hero.name} for {owner} at {hero.w},{hero.h}')
        return hero

    def add_neutral_unit(self, near_id):
        """
        Adds a unit with a random look, the figures go one by one in the order of their names. The unit appears
        at the first capital of the biome and walks along the roads to the second one, see behavior.UnitRoadToCapitalWalk.
        In a biome without roads or with less than two capitals it appears in front of the object near_id and walks around in a circle,
        see behavior.UnitWalkInCircle. Returns the unit or None.
        """
        near = self.objects.get(near_id)
        if not near or self.neutral_units_count >= MAX_NEUTRAL_UNITS:
            return None
        looks = self.store_looks()
        model_name = looks.models[self.next_figure_index % len(looks.models)]
        self.next_figure_index += 1
        capital_numbers = sorted(self.capitals.keys())
        if len(capital_numbers) >= 2 and self.roads:
            x, y = self.capitals[capital_numbers[0]]
            unit = self.create_neutral_unit(model_name, x, y)
            self.behaviors[unit.id] = behavior.UnitRoadToCapitalWalk(capital_numbers[1])
        else:
            angle = math.radians(near.direction + random.uniform(-NEUTRAL_UNIT_ANGLE, NEUTRAL_UNIT_ANGLE))
            x = near.w + near.shift_w + NEUTRAL_UNIT_DISTANCE * math.cos(angle)
            y = near.h + near.shift_h + NEUTRAL_UNIT_DISTANCE * math.sin(angle)
            unit = self.create_neutral_unit(model_name, x, y)
            self.behaviors[unit.id] = self.walk_in_circle(unit, NEUTRAL_CIRCLE_RADIUS)
        return unit

    def add_showcase_units(self):
        """
        Adds one unit of every figure of every store, so all of them can be seen in the biome. They stand at equal distances
        along all roads of the biome, see showcase_positions(), and walk around in a small circle there.
        """
        # (store id, model name) of the figures of all stores, the same figure of two stores is shown twice
        showcase = []
        for store_id in sorted(self.animations_catalog.keys()):
            showcase.extend((store_id, model_name) for model_name in self.store_looks(store_id).models)
        positions = self.showcase_positions(len(showcase))
        for (store_id, model_name), (x, y) in zip(showcase, positions):
            unit = self.create_neutral_unit(model_name, x, y, store_id=store_id)
            self.behaviors[unit.id] = self.walk_in_circle(unit, SHOWCASE_CIRCLE_RADIUS)
        if _Debug:
            print(f'added {len(positions)} showcase units in biome {self.biome.id}')

    def showcase_positions(self, count):
        """
        Points (x, y) at equal distances along all roads of the biome, one after another.
        A biome without roads gets rows SHOWCASE_SPACING segments apart around the place where the heroes appear.
        """
        if count <= 0:
            return []
        # pieces of all roads between their neighbour points: (x, y, dx, dy, length)
        pieces = []
        if self.roads:
            for road_id in sorted(self.roads.roads.keys()):
                points = self.roads.points(road_id)
                for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
                    dx, dy = self.roads.shift(ax, ay, bx, by)
                    length = math.sqrt(dx * dx + dy * dy)
                    if length > 0.0:
                        pieces.append((ax, ay, dx, dy, length))
        total = sum(piece[4] for piece in pieces)
        if total <= 0.0:
            columns = int(math.ceil(math.sqrt(count)))
            start_w, start_h = self.start_position
            return [(
                start_w + 0.5 + SHOWCASE_SPACING * (n % columns - (columns - 1) / 2.0),
                start_h + 0.5 + SHOWCASE_SPACING * (n // columns - (columns - 1) / 2.0),
            ) for n in range(count)]
        step = total / count
        positions = []
        piece_index = 0
        piece_start = 0.0
        for n in range(count):
            # in the middle of its part of the roads, so the units do not stand at the very ends of the roads
            distance = step * (n + 0.5)
            while piece_index < len(pieces) - 1 and piece_start + pieces[piece_index][4] < distance:
                piece_start += pieces[piece_index][4]
                piece_index += 1
            ax, ay, dx, dy, length = pieces[piece_index]
            t = min(1.0, (distance - piece_start) / length)
            positions.append((ax + dx * t, ay + dy * t))
        return positions

    def store_looks(self, store_id=None):
        # dress.Looks of the figures of that store, the default store when store_id is None
        if store_id is None:
            store_id = models.DEFAULT_STORE_ID
        if store_id not in self.looks:
            self.looks[store_id] = dress.Looks(self.catalog_path, store_id=store_id)
        return self.looks[store_id]

    def create_neutral_unit(self, model_name, x, y, store_id=None):
        """
        Adds a unit with a random look of that figure of that store at the open cell nearest to (x, y),
        see find_free_position(). The figure is of the default store when store_id is None.
        """
        self.next_id += 1
        unit = models.Unit(
            id=self.next_id,
            name=f'unit{self.next_id}',
            template=models.Template(**self.store_looks(store_id).random_template(model_name)),
            w=0,
            h=0,
            direction=float(random.randint(0, 359)),
        )
        unit.w, unit.h = self.find_free_position(int(math.floor(x)) % self.width, int(math.floor(y)) % self.height, size=zone.unit_size(unit))
        unit.animation = self.animation_for(unit)
        self.objects[unit.id] = unit
        self.neutral_units_count += 1
        if _Debug:
            print(f'added neutral {unit.name} {unit.template.model_name} at {unit.w},{unit.h}, {self.neutral_units_count} in biome {self.biome.id}')
        return unit

    def walk_in_circle(self, unit, radius):
        """
        The unit walks around in a circle of that radius from where it stands, see behavior.UnitWalkInCircle.
        """
        # the unit stands on the circle, the point it follows goes around from there
        angle = random.uniform(0.0, 2.0 * math.pi)
        return behavior.UnitWalkInCircle(
            unit.w + unit.shift_w - radius * math.cos(angle),
            unit.h + unit.shift_h - radius * math.sin(angle),
            radius,
            angle=angle,
            clockwise=random.choice((True, False)),
        )

    def remove_object(self, object_id):
        self.pending_controls.pop(object_id, None)
        self.behaviors.pop(object_id, None)
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
        units = [o for o in self.objects.values() if isinstance(o, models.Unit)]
        states = {}
        for u in units:
            states[u.id] = u.to_state()
            # heroes are controlled by their clients, other units by their behavior
            self.apply_controls(u)
            self.run_behavior(u)
        steers = self.avoidance(units)
        changed = []
        for u in units:
            self.run_unit(u, steers.get(u.id))
            if u.to_state() != states[u.id]:
                changed.append(u)
        return changed

    def run_behavior(self, u):
        b = self.behaviors.get(u.id)
        if b is None:
            return
        next_b = b.think(self, u)
        if next_b is not b:
            if _Debug:
                print(f'{u.name} {b!r} -> {next_b!r}')
            self.behaviors[u.id] = next_b

    def intended_movement(self, u):
        # (dx, dy) the unit is going to move in this tick without obstacles, see physics.step()
        w, h, shift_w, shift_h, _, _ = physics.step(
            u.w, u.h, u.shift_w, u.shift_h, u.direction, u.speed, u.move, u.turn, u.target_direction,
            u.max_speed, u.acceleration_value, u.deceleration_value, u.turn_speed,
        )
        return (w + shift_w) - (u.w + u.shift_w), (h + shift_h) - (u.h + u.shift_h)

    def avoidance(self, units):
        """
        Moving units which are going to run into each other soon turn aside and slow down, both of them, heroes too.
        Each one loses the part of its movement towards the other one, the closer the touch is the more it loses,
        and goes aside instead, the same way as a unit slides along a wall, see physics.slide():
        two units on crossing courses go around each other and keep going. Units which come right at each other
        both go aside to their own side of the line between them, so they pass instead of stopping.
        Returns {unit id: (dx, dy)}, see the steer of physics.move().
        """
        entries = []
        for u in units:
            dx, dy = self.intended_movement(u)
            entries.append((u, u.w + u.shift_w, u.h + u.shift_h, dx, dy, zone.unit_radius(zone.unit_size(u))))
        steers = {}
        for i in range(len(entries)):
            ui, xi, yi, vxi, vyi, ri = entries[i]
            for j in range(i + 1, len(entries)):
                uj, xj, yj, vxj, vyj, rj = entries[j]
                rvx = vxj - vxi
                rvy = vyj - vyi
                if not rvx and not rvy:
                    continue
                px = behavior.nearest_shift(xi, xj, self.width)
                py = behavior.nearest_shift(yi, yj, self.height)
                reach = ri + rj + AVOID_MARGIN
                limit = reach + AVOID_HORIZON * math.sqrt(rvx * rvx + rvy * rvy)
                if px * px + py * py > limit * limit:
                    continue
                t = collision_time(px, py, rvx, rvy, reach)
                if t is None or t > AVOID_HORIZON:
                    continue
                # direction from the first unit to the second one at the moment they touch
                nx = px + rvx * t
                ny = py + rvy * t
                length = math.sqrt(nx * nx + ny * ny)
                if length == 0.0:
                    continue
                nx /= length
                ny /= length
                # each one goes aside to its own side of that line: where it already passes the other one
                tx, ty = -ny, nx
                side = rvx * tx + rvy * ty
                side_i = -1.0 if side > 0.0 else 1.0
                urgency = 1.0 - t / AVOID_HORIZON
                approach_i = (vxi * nx + vyi * ny) * urgency
                if approach_i > 0.0:
                    self.add_steer(steers, ui.id, -nx * approach_i + side_i * tx * approach_i * AVOID_SIDE, -ny * approach_i + side_i * ty * approach_i * AVOID_SIDE)
                approach_j = -(vxj * nx + vyj * ny) * urgency
                if approach_j > 0.0:
                    self.add_steer(steers, uj.id, nx * approach_j - side_i * tx * approach_j * AVOID_SIDE, ny * approach_j - side_i * ty * approach_j * AVOID_SIDE)
        # a unit does not go faster than it wanted, even when several units turn it aside
        for u, _, _, vx, vy, _ in entries:
            steer = steers.get(u.id)
            if not steer:
                continue
            new_vx = vx + steer[0]
            new_vy = vy + steer[1]
            speed = math.sqrt(vx * vx + vy * vy)
            new_speed = math.sqrt(new_vx * new_vx + new_vy * new_vy)
            if new_speed > speed:
                k = speed / new_speed
                steers[u.id] = (new_vx * k - vx, new_vy * k - vy)
        return steers

    def add_steer(self, steers, unit_id, dx, dy):
        sx, sy = steers.get(unit_id, (0.0, 0.0))
        steers[unit_id] = (sx + dx, sy + dy)

    def run_unit(self, u, steer=None):
        # the client calculates the movement with the same function and the same zonemap, see cli.Prediction,
        # only other units are not known there and it does not know the steer, see avoidance()
        size = zone.unit_size(u)
        u.w, u.h, u.shift_w, u.shift_h, u.direction, u.speed = physics.move(
            self.zonemap, u.w, u.h, u.shift_w, u.shift_h, u.direction, u.speed, u.move, u.turn, u.target_direction,
            u.max_speed, u.acceleration_value, u.deceleration_value, u.turn_speed,
            zone.unit_radius(size), self.units_obstacle(u, size), steer,
        )
        if not u.speed:
            u.action = 'idle'
        elif u.id in self.behaviors:
            u.action = self.behaviors[u.id].action
        else:
            u.action = 'run'
        u.animation = self.animation_for(u)
