"""
Builds the map for "make land_build" from a zone of the original game "Evil Islands" instead of the map of
the Fantasy Map Generator, so the biome of the world looks like that zone: the same hills, rivers, sea, rocks,
forests, houses and trees.

A zone of the game is maps/<zone>.mpr, a .res archive: <zone>.mp says how many sectors the zone has and the kind
of every tile of its textures (grass, ground, stone, ...), every <zone>XXXYYY.sec is the sector XXX, YYY of 32 x 32
squares: the heights of the 33 x 33 corners of the squares, the texture of every tile of 2 x 2 squares,
and the water of the sector: the heights of its surface and its material. The textures of the tiles are
<zone>000.mmp, <zone>001.mmp, ... in res/textures.res. The objects of the zone (trees, houses, fences, bridges ...)
are in maps/<zone>.mob, their models in res/figures.res.

The axes of the zone are kept: the client turns the models of the game into its scene the same way as the land,
see lib.mth.ei2xyz(), so the zone is not mirrored in the world. The axis Y of the game goes up the map
and the rows of the images of the biome go down, so minimap.png shows the zone upside down compared to <zone>map.mmp.

server.land reads only a part of the map of the Fantasy Map Generator, this tool writes that part:
- one square cell for every tile of the zone, its height "h" and its biome, a cell of a big zone takes a few tiles;
- the zone is put into the middle of a square map, the ocean is around it: the land at the edges of the zone
  slopes down into the ocean;
- the heights are in the same scale as the land: one tile of the zone takes as much of the land as one cell takes
  of the biome of that size (see --size), the heights go up and down the same way. The world has one level of water,
  the zone has a few: the sea and the rivers higher up, so the land is measured from the water nearest to it;
- the tiles: water where the land is under the water of the zone, the rocks get the "cliff2" tile, the ground gets
  the biome with the closest color of the tile, the grass gets a biome by the number of trees of the zone around it;
- the biome has no bridges to walk on, the water around the bridges of the zone is shallow, so the units cross
  the rivers there;
- no capitals, routes or rivers of the Fantasy Map Generator, instead "land" has:
  "exact_heights": true, so server.land keeps the heights as they are;
  "buildings": the objects of the zone which are not plants, as in buildings.json, the positions are in cells
  of the map, "a" are the circles (x, y, radius) which cover the model, units can not walk there, see lib.zone;
  "plants": the trees and bushes of the zone, the name of every plant is from catalog/plants.json.

usage: ei_map.py <game data folder> <zone> <output map.json> [--size 128] [--vertical 1.0] [--catalog catalog/]
run inside repack/EIrepack with Formats/ and src/ on the path, see "make ei_map" in the Makefile
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import io
import os
import math
import json
import argparse
import tempfile
import contextlib

import numpy as np
import cv2

#------------------------------------------------------------------------------

import res
import mp
import sec
import mob
import mmp
import lnk
import fig

#------------------------------------------------------------------------------

from lib import const
from lib import mth
from server import land

#------------------------------------------------------------------------------

# kinds of the tiles of the textures, see Formats/mp.py
TILE_KINDS = (
    'grass', 'ground', 'stone', 'sand', 'rock', 'field', 'water', 'road',
    'empty', 'snow', 'ice', 'drygrass', 'snowballs', 'lava', 'swamp', 'highrock',
)

# biomes of the Fantasy Map Generator in its order, server.land takes the tiles and the plants of the cells by them
BIOMES = (
    ('Marine', '#466eab'),
    ('Hot desert', '#fbe79f'),
    ('Cold desert', '#b5b887'),
    ('Savanna', '#d2d082'),
    ('Grassland', '#c8d68f'),
    ('Tropical seasonal forest', '#b6d95d'),
    ('Temperate deciduous forest', '#29bc56'),
    ('Tropical rainforest', '#7dcb35'),
    ('Temperate rainforest', '#409c43'),
    ('Taiga', '#4b6b32'),
    ('Tundra', '#96784b'),
    ('Glacier', '#d5e7eb'),
    ('Wetland', '#0b9131'),
)

# biomes of the kinds of the tiles, the other kinds get the biome of the ground with the closest color of the texture
KIND_BIOMES = {
    'field': 'Grassland',
    'drygrass': 'Cold desert',
    'road': 'Cold desert',
    'sand': 'Wetland',
    'swamp': 'Wetland',
    'snow': 'Glacier',
    'ice': 'Glacier',
    'snowballs': 'Glacier',
    'lava': 'Hot desert',
}

# biomes of the ground, the color of the texture of the tile picks one of them
GROUND_BIOMES = ('Cold desert', 'Tundra', 'Wetland')

# rocks get this tile, it borders on most of the tiles of the land
ROCK_KINDS = ('stone', 'rock', 'highrock')
ROCK_TILE = 'cliff2'
ROCK_BIOME = 'Tundra'

# grass without trees around keeps the tile of the grass and gets no plants, more trees around make a forest
GRASS_TILE = 'grass2'
GRASS_BIOME = 'Grassland'
SPARSE_FOREST_BIOME = 'Temperate rainforest'
DENSE_FOREST_BIOME = 'Taiga'

# trees of the zone in a tile, smoothed over FOREST_RADIUS tiles around, the grass is a forest from that many
FOREST_RADIUS = 2.0
SPARSE_FOREST_DENSITY = 0.03
DENSE_FOREST_DENSITY = 0.12

# the objects of the zone which are trees and bushes, by the start of the name of the model: a tree counts as one,
# a bush as half
TREE_MODELS = ('nafltr', )
BUSH_MODELS = ('naflbu', )

# units cross the rivers of the zone by the bridges, the biome has no bridges: the water around a bridge
# is that shallow instead, in the units of the world, so the units can walk there, see server.land.ZONE_MAX_DEPTH
BRIDGE_MODELS = ('stbr', )
FORD_RADIUS = 4.0
FORD_DEPTH = 0.4

# land lower than its water by that much is under the water, in the heights of the zone
MIN_WATER_DEPTH = 0.05

# the water nearest to the land is smoothed over that many tiles, so the land between the sea and the rivers
# higher up goes up smoothly
WATER_LEVEL_SMOOTHING = 6.0

# the land of the zone slopes down into the ocean over that many cells around the zone, down to that depth
EDGE_CELLS = 8
OCEAN_DEPTH = 6.0

# server.land keeps the index of a cell in two bytes
MAX_CELLS = 256 * 256

CATALOG_DIR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(const.__file__)))), 'catalog')

# records of the .mob which are units, those are not put on the map
UNIT_NODES = ('UNIT', )

# objects of the zone which are plants, by the start of the name of the model, and also the templates
# of catalog/plants.json; the season of a plant by its texture when catalog/plants.json does not know it
PLANT_MODELS = ('nafl', )
TEXTURE_SEASONS = {
    'tree01': 'autumn',
    'tree02': 'temperate',
    'tree03': 'winter',
    'tree04': 'desert',
    'tree05': 'autumn',
    'tree06': 'autumn',
}

# "k" of the other objects in buildings.json, by the start of the name of the model
OBJECT_KINDS = (
    ('stbr', 'bridge'),
    ('stwa', 'wall'),
    ('stga', 'gate'),
    ('stbu', 'house'),
    ('stto', 'torch'),
    ('nast', 'stone'),
    ('stst', 'structure'),
)
DEFAULT_OBJECT_KIND = 'object'

# units walk over the bridges and through the gates, those do not close the land
OPEN_KINDS = ('bridge', 'gate')

# smallest circle which covers a part of a model where units can not walk, in squares of the zone:
# a fence is thin, a row of such circles goes along it
MIN_AREA_RADIUS = 0.3

# a part of a model bigger than that is not real, in squares
MAX_PART_SIZE = 100.0

#------------------------------------------------------------------------------


def read_archive(file_path, names=None):
    """
    Files of a .res archive: lower case name -> bytes, only the files of those lower case names when given.
    """
    result = {}
    with open(file_path, 'rb') as f:
        for name, offset, size in res.read_filetree(f):
            name = name.lower()
            if names is not None and name not in names:
                continue
            f.seek(offset)
            result[name] = f.read(size)
    return result


def write_temp_file(temp_dir_path, file_name, data):
    file_path = os.path.join(temp_dir_path, file_name)
    with open(file_path, 'wb') as f:
        f.write(data)
    return file_path


def read_zone(game_dir_path, zone_name, temp_dir_path):
    """
    The land of the zone in the axes of the game, the row of an array is Y:
    "land": heights of the corners of the squares, "liquid": heights of the surface of the water at the corners
    (nan where there is no water), "material": material of the water of every tile (-1 where there is none),
    "texture": texture of every tile, "kinds": kind of every texture, "colors": color of every texture.
    """
    files = read_archive(os.path.join(game_dir_path, 'maps', zone_name + '.mpr'))
    info = mp.read_info(write_temp_file(temp_dir_path, 'zone.mp', files[zone_name + '.mp']))
    altitude, sectors_x, sectors_y = info[0], info[1], info[2]
    textures_count, texture_size, tile_size = info[3], info[4], info[6]
    land_heights = np.zeros((sectors_y * 32 + 1, sectors_x * 32 + 1))
    liquid_heights = np.full(land_heights.shape, np.nan)
    materials = np.full((sectors_y * 16, sectors_x * 16), -1)
    textures = np.zeros(materials.shape, dtype=int)
    for i in range(sectors_x):
        for j in range(sectors_y):
            sector_file_name = '%s%03d%03d.sec' % (zone_name, i, j)
            sector = sec.read_info(write_temp_file(temp_dir_path, sector_file_name, files[sector_file_name]))
            for k, vertex in enumerate(sector[1]):
                land_heights[j * 32 + k // 33, i * 32 + k % 33] = vertex[2] * altitude / 65535.0
            if sector[0]:
                for k, vertex in enumerate(sector[2]):
                    liquid_heights[j * 32 + k // 33, i * 32 + k % 33] = vertex[2] * altitude / 65535.0
                for k, material in enumerate(sector[5]):
                    if material != 65535:
                        materials[j * 16 + k // 16, i * 16 + k % 16] = material
            for k, texture in enumerate(sector[3] if sector[0] else sector[2]):
                textures[j * 16 + k // 16, i * 16 + k % 16] = texture[1] * 64 + texture[0]
    texture_names = ['%s%03d.mmp' % (zone_name, i) for i in range(textures_count)]
    texture_files = read_archive(os.path.join(game_dir_path, 'res', 'textures.res'), names=set(texture_names))
    tiles_in_row = texture_size // tile_size
    colors = {}
    for texture_index, texture_name in enumerate(texture_names):
        image = mmp.read_image(write_temp_file(temp_dir_path, texture_name, texture_files[texture_name]))
        pixels = np.array(image.convert('RGB')).astype(float)
        for tile_index in range(64):
            x = (tile_index % tiles_in_row) * tile_size
            y = (tile_index // tiles_in_row) * tile_size
            colors[texture_index * 64 + tile_index] = pixels[y:y + tile_size, x:x + tile_size].reshape(-1, 3).mean(axis=0)
    return {
        'land': land_heights,
        'liquid': liquid_heights,
        'material': materials,
        'texture': textures,
        'kinds': [TILE_KINDS[kind] for kind in info[10]],
        'colors': colors,
    }


def read_objects(game_dir_path, zone_name):
    """
    Objects of the zone which are not units: trees, bushes, houses, fences, bridges, torches and so on.
    Every object is a dict: "model", "texture", "parts" (None when the object has all parts of its model),
    "coefs" (the complection of the model), "x", "y", "z" in squares of the zone, z is over the land,
    "yaw": turn around the vertical axis in radians.
    """
    # the reader prints the records it does not know
    with contextlib.redirect_stdout(io.StringIO()):
        info = mob.read_info(os.path.join(game_dir_path, 'maps', zone_name + '.mob'))
    result = []
    collect_objects(info or [], result)
    return result


def collect_objects(nodes, result):
    for node in nodes:
        if not isinstance(node, list) or len(node) != 2 or not isinstance(node[1], list):
            continue
        fields = dict((n[0], n[1]) for n in node[1] if isinstance(n, list) and len(n) == 2 and not isinstance(n[1], list))
        if node[0] not in UNIT_NODES and fields.get('OBJTEMPLATE') and fields.get('OBJPOSITION'):
            x, y, z = fields['OBJPOSITION'][:3]
            # (w, x, y, z), the objects lean a little on the slopes, only the turn around the vertical axis is kept
            qw, qx, qy, qz = fields.get('OBJROTATION') or (1.0, 0.0, 0.0, 0.0)
            coefs = list(fields.get('OBJCOMPLECTION') or (0.0, 0.0, 0.0))
            coefs[2] = abs(coefs[2])
            parts = sorted(p.lower() for p in fields.get('OBJBODYPARTS') or () if p and p != 'None')
            result.append({
                'model': fields['OBJTEMPLATE'].lower(),
                'texture': str(fields.get('OBJPRIMTXTR') or '').lower(),
                'parts': parts or None,
                'coefs': coefs,
                'x': x,
                'y': y,
                'z': z,
                'yaw': math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)),
            })
        collect_objects(node[1], result)
    return result


def read_models(game_dir_path, model_names, temp_dir_path):
    """
    Parts of the models from res/figures.res: model name -> list of (part name, parent part name or None,
    offsets, centers, half sizes), every one of the three is 8 x 3: the variants of the model by its coefficients,
    see lib.mth.trilinear(). The offset of a part is from its parent, see .bon, the center and the half size
    are of the box around the part, see the header of .fig.
    """
    names = set(model_name + suffix for model_name in model_names for suffix in ('.mod', '.bon'))
    files = read_archive(os.path.join(game_dir_path, 'res', 'figures.res'), names=names)
    result = {}
    for model_name in model_names:
        if model_name + '.mod' not in files:
            continue
        parts_files = read_archive(write_temp_file(temp_dir_path, model_name + '.mod', files[model_name + '.mod']))
        bones = {}
        if model_name + '.bon' in files:
            bones = read_archive(write_temp_file(temp_dir_path, model_name + '.bon', files[model_name + '.bon']))
        if model_name not in parts_files:
            continue
        parents = {}
        collect_parents(lnk.read_info(write_temp_file(temp_dir_path, model_name + '.lnk', parts_files[model_name])), None, parents)
        parts = []
        for part_name, parent_name in parents.items():
            part_name_lower = part_name.lower()
            if part_name_lower not in parts_files:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                info = fig.read_info(write_temp_file(temp_dir_path, part_name_lower + '.fig', parts_files[part_name_lower]))
            if not info or info[0] != 8:
                continue
            offsets = np.zeros((8, 3))
            if part_name_lower in bones and len(bones[part_name_lower]) >= 96:
                offsets = np.frombuffer(bones[part_name_lower][:96], dtype='<f4').reshape(8, 3).astype(float)
            parts.append((part_name_lower, parent_name.lower() if parent_name else None, offsets, np.array(info[9]), np.array(info[11])))
        result[model_name] = parts
    return result


def collect_parents(tree, parent_name, parents):
    """
    Parent of every part of the tree of the parts of a model, see Formats/lnk.py.
    """
    if not tree:
        return parents
    name, children = tree[0], tree[1]
    parents[name] = parent_name
    for child in children:
        collect_parents(child, name, parents)
    return parents


def model_box(model_parts, coefs, selected_parts):
    """
    Box around the model with those coefficients from above: min x, min y, max x, max y in squares,
    only the selected parts when given. None when the model is not known.
    """
    if not model_parts:
        return None
    offsets = dict((name, mth.trilinear(part_offsets, coefs)) for name, _, part_offsets, _, _ in model_parts)
    parents = dict((name, parent_name) for name, parent_name, _, _, _ in model_parts)
    box = None
    for name, _, _, centers, halves in model_parts:
        if selected_parts and name not in selected_parts:
            continue
        position = np.zeros(3)
        part_name = name
        while part_name is not None:
            position = position + offsets.get(part_name, np.zeros(3))
            part_name = parents.get(part_name)
        center = position + mth.trilinear(centers, coefs)
        half = np.abs(mth.trilinear(halves, coefs))
        # some parts have no box, their header is garbage
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half)) or half.max() > MAX_PART_SIZE:
            continue
        part_box = (center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1])
        if box is None:
            box = part_box
        else:
            box = (min(box[0], part_box[0]), min(box[1], part_box[1]), max(box[2], part_box[2]), max(box[3], part_box[3]))
    return box


def box_circles(box, x, y, yaw):
    """
    Circles which cover the box of an object at (x, y) turned by yaw: (x, y, radius) in squares of the zone.
    A long box (a fence) is covered by a row of circles along it.
    """
    min_x, min_y, max_x, max_y = box
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_x = (max_x - min_x) / 2.0
    half_y = (max_y - min_y) / 2.0
    radius = max(min(half_x, half_y), MIN_AREA_RADIUS)
    length = max(half_x, half_y)
    count = max(1, int(round(length / radius)))
    along = np.linspace(-(length - radius), length - radius, count) if count > 1 else [0.0]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    circles = []
    for a in along:
        local_x = center_x + (a if half_x >= half_y else 0.0)
        local_y = center_y + (0.0 if half_x >= half_y else a)
        circles.append((x + local_x * cos_yaw - local_y * sin_yaw, y + local_x * sin_yaw + local_y * cos_yaw, radius))
    return circles


def object_kind(model_name):
    for prefix, kind in OBJECT_KINDS:
        if model_name.startswith(prefix):
            return kind
    return DEFAULT_OBJECT_KIND


def is_plant(obj, plants_names):
    return obj['model'].startswith(PLANT_MODELS) or template_key(obj) in plants_names


def template_key(obj):
    """
    The key of the template of the object in catalog/plants.json and plants.bin: "model#texture#parts".
    """
    return '%s#%s#%s' % (obj['model'], obj['texture'], ':'.join(obj['parts']) if obj['parts'] else 'null')


def read_plants_names(catalog_dir_path):
    """
    Name of the plant of every template of catalog/plants.json: "model#texture#parts" -> "temperate_spruce".
    The trees and stumps are solid, see lib.zone.SOLID_PLANT_NAME_SUFFIXES.
    """
    with open(os.path.join(catalog_dir_path, 'plants.json'), 'rt') as f:
        catalog_plants = json.loads(f.read())
    result = {}
    for plant_name, templates in catalog_plants.items():
        for key in templates:
            result.setdefault(key.lower(), plant_name)
    return result


def plant_name(obj, plants_names):
    """
    The name of the plant from catalog/plants.json, or made of the season of its texture and the kind of its model.
    """
    key = template_key(obj)
    if key in plants_names:
        return plants_names[key]
    for name, plants_name in plants_names.items():
        if name.startswith(obj['model'] + '#' + obj['texture'] + '#'):
            return plants_name
    season = TEXTURE_SEASONS.get(obj['texture'], 'generic')
    return f'{season}_{"tree" if obj["model"].startswith(TREE_MODELS) else "bush"}'


def tiles_heights(corners):
    """
    Height of every tile of 2 x 2 squares: the middle of the tile counts most, then its sides, then its corners.
    """
    weights = np.array([1.0, 2.0, 1.0])
    result = np.zeros(((corners.shape[0] - 1) // 2, (corners.shape[1] - 1) // 2))
    for dy in range(3):
        for dx in range(3):
            result += corners[dy:dy + corners.shape[0] - 2:2, dx:dx + corners.shape[1] - 2:2] * weights[dy] * weights[dx]
    return result / 16.0


def tiles_water_levels(liquid, materials):
    """
    Height of the surface of the water of every tile which has water, nan for the others.
    """
    result = np.full(materials.shape, np.nan)
    for row, col in zip(*np.where(materials >= 0)):
        levels = liquid[row * 2:row * 2 + 3, col * 2:col * 2 + 3]
        if not np.all(np.isnan(levels)):
            result[row, col] = np.nanmax(levels)
    return result


def nearest_water_levels(water_levels, fallback):
    """
    Height of the water nearest to every tile, smoothed: the tiles take the levels of their neighbors step by step.
    """
    known = ~np.isnan(water_levels)
    if not known.any():
        return np.full(water_levels.shape, fallback)
    levels = np.where(known, water_levels, 0.0)
    rows, cols = levels.shape
    while not known.all():
        padded_levels = np.pad(np.where(known, levels, 0.0), 1)
        padded_known = np.pad(known, 1).astype(float)
        sums = np.zeros(levels.shape)
        counts = np.zeros(levels.shape)
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            sums += padded_levels[dy:dy + rows, dx:dx + cols]
            counts += padded_known[dy:dy + rows, dx:dx + cols]
        grown = ~known & (counts > 0)
        levels[grown] = sums[grown] / counts[grown]
        known = known | grown
    return cv2.GaussianBlur(levels, (0, 0), WATER_LEVEL_SMOOTHING, borderType=cv2.BORDER_REPLICATE)


def forest_density(objects, shape):
    """
    Trees of the zone in every tile, smoothed over FOREST_RADIUS tiles around.
    """
    trees = np.zeros(shape)
    for obj in objects:
        weight = 1.0 if obj['model'].startswith(TREE_MODELS) else (0.5 if obj['model'].startswith(BUSH_MODELS) else 0.0)
        row, col = int(obj['y'] / 2.0), int(obj['x'] / 2.0)
        if weight and 0 <= row < shape[0] and 0 <= col < shape[1]:
            trees[row, col] += weight
    return cv2.GaussianBlur(trees, (0, 0), FOREST_RADIUS, borderType=cv2.BORDER_CONSTANT)


def fords(objects, shape):
    """
    Tiles near the bridges of the zone.
    """
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    result = np.zeros(shape, dtype=bool)
    for obj in objects:
        if obj['model'].startswith(BRIDGE_MODELS):
            result |= np.hypot(rows + 0.5 - obj['y'] / 2.0, cols + 0.5 - obj['x'] / 2.0) <= FORD_RADIUS
    return result


def color_distance(c1, c2):
    return sum(abs(float(a) - float(b)) for a, b in zip(c1, c2))


def ground_biome(color):
    """
    The biome of the ground with the closest color of its tile.
    """
    return min(GROUND_BIOMES, key=lambda biome: color_distance(color, land.tiles_colors[land.biomes_mapping[biome][0][0]]))


def tile_biome(kind, color, trees):
    """
    The biome of a tile of the land and the tile it keeps, None when it takes the tile of the biome.
    """
    if kind in ROCK_KINDS:
        return ROCK_BIOME, ROCK_TILE
    if kind == 'grass':
        if trees >= DENSE_FOREST_DENSITY:
            return DENSE_FOREST_BIOME, None
        if trees >= SPARSE_FOREST_DENSITY:
            return SPARSE_FOREST_BIOME, None
        return GRASS_BIOME, GRASS_TILE
    if KIND_BIOMES.get(kind):
        return KIND_BIOMES[kind], None
    return ground_biome(color), None


def height_levels():
    """
    Heights of the cells of the map ("h", 0 - 100) and the elevations of the land they make in the units
    of the world, see server.land.unpack_elevation(), and the elevation of the water.
    """
    levels = np.arange(0, 101)
    elevations = land.unpack_elevation(levels) * const.ELEVATION_FACTOR
    water_elevation = const.WATER_LEVEL / 100.0 * const.ELEVATION_FACTOR
    return levels, elevations, water_elevation


def pick_level(levels, elevations, elevation):
    return int(levels[np.argmin(np.abs(elevations - elevation))])


def merge_tiles(values, factor):
    """
    Average of every block of factor x factor tiles, the last blocks repeat the tiles at the edges.
    """
    rows = -(-values.shape[0] // factor) * factor
    cols = -(-values.shape[1] // factor) * factor
    values = np.pad(values, ((0, rows - values.shape[0]), (0, cols - values.shape[1])), mode='edge')
    return values.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


def build_map(zone, objects, models, plants_names, size, vertical):
    tile_rows, tile_cols = zone['material'].shape
    # server.land keeps the index of a cell in two bytes, a cell of a big zone takes a few tiles
    factor = 1
    while (-(-max(tile_rows, tile_cols) // factor) + 2 * EDGE_CELLS) ** 2 > MAX_CELLS:
        factor += 1
    rows = -(-tile_rows // factor)
    cols = -(-tile_cols // factor)
    side = max(rows, cols) + 2 * EDGE_CELLS
    top = (side - rows) // 2
    left = (side - cols) // 2
    # one cell of the map is that many units of the world, see server.land.xy2draw()
    cell_size = const.SEGMENT_SIZE * size * land.GLOBAL_SCALE / side
    # the heights of the zone are in squares, a tile is 2 x 2 squares
    height_scale = cell_size / (2.0 * factor) * vertical
    heights = tiles_heights(zone['land'])
    water_levels = tiles_water_levels(zone['liquid'], zone['material'])
    under_water = ~np.isnan(water_levels) & (heights < water_levels - MIN_WATER_DEPTH)
    nearest_levels = nearest_water_levels(water_levels, heights.min())
    levels, elevations, water_elevation = height_levels()
    land_levels = levels > const.INPUT_WATER_LEVEL
    water_levels_mask = (levels >= 1) & (elevations < water_elevation)
    min_land = elevations[land_levels].min() - water_elevation
    # elevation over the water in the units of the world: positive is the land, negative is under the water
    real_depths = (np.nan_to_num(water_levels) - heights) * height_scale
    depths = np.where(fords(objects, (tile_rows, tile_cols)), np.minimum(real_depths, FORD_DEPTH), real_depths)
    # the bottom near the bridges is that much higher than in the zone, the bridges stand higher by that
    raised = np.where(under_water, real_depths - depths, 0.0)
    over_water = np.where(
        under_water,
        -depths,
        np.maximum((heights - nearest_levels) * height_scale, min_land),
    )
    trees = forest_density(objects, (tile_rows, tile_cols))
    over_water = merge_tiles(over_water, factor)
    textures = zone['texture'][::factor, ::factor]
    trees = merge_tiles(trees, factor)
    padding = ((top, side - rows - top), (left, side - cols - left))
    over_water = np.pad(over_water, padding, mode='edge')
    textures = np.pad(textures, padding, mode='edge')
    trees = np.pad(trees, padding, mode='edge')
    # distance from the zone in tiles: the land slopes down into the ocean there
    row_distance = np.maximum(np.maximum(top - np.arange(side), np.arange(side) - (top + rows - 1)), 0)
    col_distance = np.maximum(np.maximum(left - np.arange(side), np.arange(side) - (left + cols - 1)), 0)
    distance = np.hypot(row_distance[:, None], col_distance[None, :])
    t = np.clip(distance / EDGE_CELLS, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    over_water = over_water * (1.0 - t) - OCEAN_DEPTH * t
    biome_ids = dict((name, i) for i, (name, _) in enumerate(BIOMES))
    vertices = []
    for row in range(side + 1):
        for col in range(side + 1):
            vertices.append({'i': len(vertices), 'p': [col, row]})
    cells = []
    for row in range(side):
        for col in range(side):
            i = len(cells)
            elevation = over_water[row, col]
            cell = {
                'i': i,
                'v': [row * (side + 1) + col, row * (side + 1) + col + 1, (row + 1) * (side + 1) + col + 1, (row + 1) * (side + 1) + col],
                'p': [col + 0.5, row + 0.5],
                'f': 1,
            }
            if elevation < 0:
                cell['h'] = pick_level(levels[water_levels_mask], elevations[water_levels_mask], water_elevation + elevation)
                cell['biome'] = biome_ids['Marine']
                cell['f'] = 2
            else:
                texture = int(textures[row, col])
                cell['h'] = pick_level(levels[land_levels], elevations[land_levels], water_elevation + elevation)
                biome, tile = tile_biome(zone['kinds'][texture], zone['colors'][texture], trees[row, col])
                cell['biome'] = biome_ids[biome]
                if tile:
                    cell['tile'] = tile
            cells.append(cell)

    def _map_xy(x, y):
        # squares of the zone -> cells of the map
        return round(x / (2.0 * factor) + left, 3), round(y / (2.0 * factor) + top, 3)

    # the models are drawn lib.const.MODELS_SCALE_FACTOR times bigger than they are in squares of the zone
    scale = round(cell_size / (2.0 * factor) / const.MODELS_SCALE_FACTOR, 3)
    buildings = []
    plants = []
    for obj in objects:
        x, y = _map_xy(obj['x'], obj['y'])
        # the models turn the same way as in the game, a unit of the client looks along "d" + 90, see unt.Unit
        direction = (math.degrees(obj['yaw']) - 90.0) % 360.0
        coefs = ':'.join(str(round(c, 1)) for c in obj['coefs'])
        if is_plant(obj, plants_names):
            plants.append({
                'n': plant_name(obj, plants_names),
                'm': obj['model'],
                't': obj['texture'],
                'p': ':'.join(obj['parts']) if obj['parts'] else None,
                'c': coefs,
                'd': int(round(direction)) % 360,
                'x': x,
                'y': y,
            })
            continue
        kind = object_kind(obj['model'])
        row = min(max(int(obj['y'] / 2.0), 0), tile_rows - 1)
        col = min(max(int(obj['x'] / 2.0), 0), tile_cols - 1)
        areas = []
        box = model_box(models.get(obj['model']), obj['coefs'], obj['parts'])
        if box and kind not in OPEN_KINDS:
            for area_x, area_y, radius in box_circles(box, obj['x'], obj['y'], obj['yaw']):
                areas.append(list(_map_xy(area_x, area_y)) + [round(radius / (2.0 * factor), 3)])
        buildings.append({
            'k': kind,
            'n': f'{obj["model"]}#{obj["texture"]}',
            'm': obj['model'],
            't': obj['texture'],
            'p': ':'.join(obj['parts']) if obj['parts'] else None,
            'c': coefs,
            's': f'{scale}:{scale}:{scale}',
            'd': round(direction, 1),
            # the client puts the model that much lower than the land, see scen.Scene.coords_map2xyz()
            'e': round(float(raised[row, col]) - obj['z'] * height_scale, 3),
            'x': x,
            'y': y,
            'a': areas,
        })
    if _Debug:
        print(f'{side} x {side} cells of {factor} x {factor} tiles, one cell is {cell_size:.2f} units of the world, '
              f'heights are {height_scale:.2f} units for one square')
        print(f'{int(under_water.sum())} tiles under the water, {len(buildings)} buildings, {len(plants)} plants')
    return {
        'info': {'mapName': zone['name'], 'width': side, 'height': side},
        'land': {
            'exact_heights': True,
            'buildings': buildings,
            'plants': plants,
        },
        'biomesData': {
            'i': list(range(len(BIOMES))),
            'name': [name for name, _ in BIOMES],
            'color': [color for _, color in BIOMES],
        },
        'grid': {
            'cells': [],
        },
        'pack': {
            'cells': cells,
            'vertices': vertices,
            'features': [0, {'i': 1, 'type': 'island', 'land': True}, {'i': 2, 'type': 'ocean', 'land': False}],
            'burgs': [0],
            'states': [],
            'cultures': [],
            'routes': [],
            'rivers': [],
        },
    }


def main():
    parser = argparse.ArgumentParser(description='builds the map for server.land from a zone of the original game')
    parser.add_argument('game_dir_path', metavar='GAME_DIR', help='folder of the game with the maps/ and res/ folders')
    parser.add_argument('zone_name', metavar='ZONE', help='name of the zone, for example zone1 is maps/zone1.mpr and maps/zone1.mob')
    parser.add_argument('map_file_path', metavar='MAP_JSON', help='output JSON file, assets/map.json for "make land_build"')
    parser.add_argument('--size', dest='size', type=int, default=128, help='width and height of the biome, the same as for server.land')
    parser.add_argument('--vertical', dest='vertical', type=float, default=1.0, help='the heights of the zone are that many times higher')
    parser.add_argument('--catalog', dest='catalog_dir_path', default=CATALOG_DIR_PATH, metavar='CATALOG_DIR',
                        help='folder with plants.json, the names of the plants are taken from there')
    args = parser.parse_args()
    zone_name = args.zone_name.lower()
    with tempfile.TemporaryDirectory() as temp_dir_path:
        zone = read_zone(args.game_dir_path, zone_name, temp_dir_path)
        zone['name'] = zone_name
        objects = read_objects(args.game_dir_path, zone_name)
        plants_names = read_plants_names(args.catalog_dir_path)
        building_models = sorted(set(obj['model'] for obj in objects if not is_plant(obj, plants_names)))
        models = read_models(args.game_dir_path, building_models, temp_dir_path)
    data = build_map(zone, objects, models, plants_names, args.size, args.vertical)
    with open(args.map_file_path, 'wt') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    main()
