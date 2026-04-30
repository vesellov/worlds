import os
import sys
import json
import pprint
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from shapely.geometry import Point, Polygon
import numpy as np

import mth


min_x = 0
min_y = 0
input_width = 0
input_height = 0
width = 0
height = 0
min_elevation = 0
max_elevation = 0
# min_elevation_unpacked = 0
# max_elevation_unpacked = 0
# water_level_unpacked = 0
# elevation_data = {}


INPUT_WATER_LEVEL = 20  # 20 is the water level in the input heightmap, 100 is the input max height
# ELEVATION_UNPACK_EXPONENT = 2.1
# ELEVATION_UNPACK_UNDERWATER_FACTOR = 80.0
CLIFFS_HEIGHT_MARGIN = 60
CLIFFS_HEIGHT_DROP = 10
TRANSFORM_CYCLES = 20
RIVERS_MAX_ELEVATION_LEVEL = 4
LAKES_MAX_ELEVATION_LEVEL = 2
ROUTES_MIN_ELEVATION_LEVEL = 23

# see: https://github.com/vesellov/worlds/blob/main/catalog.json
tiles_colors = {
    'cliff1': (108, 107, 94),
    'cliff10': (33, 47, 70),
    'cliff11': (122, 122, 122),
    'cliff2': (116, 114, 86),
    'cliff3': (97, 95, 58),
    'cliff5': (80, 79, 61),
    'cliff7': (41, 41, 22),
    'cliff8': (121, 88, 86),
    'cliff9': (59, 77, 42),
    'dirt1': (104, 89, 61),
    'dirt2': (105, 83, 63),
    'dirt3': (99, 73, 46),
    'dirt4': (97, 80, 65),
    'dirt5': (81, 65, 47),
    'dirt6': (99, 101, 98),
    'dirt7': (92, 83, 72),
    'dust1': (115, 95, 71),
    'dust2': (68, 54, 39),
    'dust3': (124, 110, 96),
    'grass1': (24, 39, 0),
    'grass2': (33, 55, 2),
    'grass3': (80, 104, 3),
    'grass5': (95, 98, 2),
    'grass6': (46, 69, 1),
    'grass7': (40, 62, 17),
    'lava1': (155, 35, 20),
    'mud1': (170, 99, 28),
    'mud2': (130, 64, 32),
    'rock1': (79, 77, 74),
    'rock2': (56, 58, 50),
    'sand1': (177, 172, 113),
    'sand2': (176, 158, 104),
    'sand4': (151, 130, 91),
    'sand5': (163, 142, 94),
    'sand6': (163, 142, 94),
    'sand7': (139, 114, 92),
    'snow1': (209, 220, 250),
    'snow2': (208, 215, 228),
    'snow3': (242, 243, 252),
    'snow4': (227, 229, 239),
    'soil1': (116, 112, 76),
    'soil2': (103, 102, 49),
    'soil3': (87, 78, 47),
    'soil4': (61, 58, 29),
    'soil5': (72, 62, 1),
    'soil6': (146, 139, 82),
    'stone1': (78, 72, 62),
    'stone2': (100, 85, 72),
    'stone3': (118, 116, 88),
    'stone5': (176, 165, 130),
    'stone7': (142, 85, 54),
    'stone8': (52, 43, 27),
    'tile1': (80, 84, 68),
    'tile2': (92, 100, 74),
    'water1': (43, 125, 193),
    'water2': (65, 126, 113),
    'water3': (96, 157, 139),
    'water4': (15, 96, 81),
    'water5': (49, 70, 93),
    'water6': (40, 64, 89),
    'water7': (54, 72, 99),
    'water8': (11, 35, 56),
}
tiles_colors_reversed = {v: k for k, v in tiles_colors.items()}
tiles_colors_reversed[(0, 0, 0)] = 'water5'

# see: https://github.com/Azgaar/Fantasy-Map-Generator/blob/master/src/modules/biomes.ts#L12
biomes_colors = {
    "466eab": "Marine",
    "fbe79f": "Hot desert",
    "b5b887": "Cold desert",
    "d2d082": "Savanna",
    "c8d68f": "Grassland",
    "b6d95d": "Tropical seasonal forest",
    "29bc56": "Temperate deciduous forest",
    "7dcb35": "Tropical rainforest",
    "409c43": "Temperate rainforest",
    "4b6b32": "Taiga",
    "96784b": "Tundra",
    "d5e7eb": "Glacier",
    "0b9131": "Wetland",
}
biomes_colors = {tuple(int(k[i:i+2], 16) for i in (0, 2, 4)):v for k, v in biomes_colors.items()}

biomes_mapping = {
    # https://en.wikipedia.org/wiki/Marines
    'Marine':                       [('water5', 1.0), ],
    # https://en.wikipedia.org/wiki/Desert_climate#Hot_desert_climates
    'Hot desert':                   [('mud2', 0.75), ('sand2', 1.0), ],
    # https://en.wikipedia.org/wiki/Desert_climate#Cold_desert_climates
    'Cold desert':                  [('soil6', 0.75), ('sand2', 1.0), ],
    # https://en.wikipedia.org/wiki/Savanna
    'Savanna':                      [('soil3', 0.5), ('dust1', 0.75), ('dirt2', 1.0),],
    # https://en.wikipedia.org/wiki/Grassland
    'Grassland':                    [('soil3', 0.3), ('soil5', 0.6), ('grass2', 1.0), ],
    # https://en.wikipedia.org/wiki/Seasonal_tropical_forest
    'Tropical seasonal forest':     [('soil5', 0.5), ('grass2', 1.0), ],
    # https://en.wikipedia.org/wiki/Temperate_deciduous_forest
    'Temperate deciduous forest':   [('soil4', 0.5), ('grass1', 1.0), ],
    # https://en.wikipedia.org/wiki/Tropical_rainforest
    'Tropical rainforest':          [('grass3', 0.5), ('grass2', 1.0), ],
    # https://en.wikipedia.org/wiki/Temperate_rainforest
    'Temperate rainforest':         [('grass1', 0.5), ('grass2', 1.0), ],
    # https://en.wikipedia.org/wiki/Taiga
    'Taiga':                        [('soil5', 0.5), ('grass1', 6.0), ('grass2', 1.0), ],
    # https://en.wikipedia.org/wiki/Tundra
    'Tundra':                       [('soil3', 0.5), ('dirt2', 1.0), ],
    # https://en.wikipedia.org/wiki/Glacier
    'Glacier':                      [('snow3', 1.0), ],
    # https://en.wikipedia.org/wiki/Wetland
    'Wetland':                      [('grass2', 0.5), ('sand4', 1.0), ],                                    
}

trees_biomes_mapping = {
    'Grassland': { # few trees, but more plants
        'soil3': [(0.2, 'grassland:tree'), (0.3, 'grassland:bush'), (0.35, 'grassland:deadwood'), (0.5, 'grassland:mushroom'), ],
        'soil5': [(0.2, 'grassland:tree'), (0.3, 'grassland:bush'), (0.35, 'grassland:deadwood'), (0.5, 'grassland:mushroom'), ],
        'grass2': [(0.2, 'grassland:tree'), (0.3, 'grassland:bush'), (0.35, 'grassland:deadwood'), (0.5, 'grassland:mushroom'), ],
    },
    'Savanna': {    # few trees, few plants
        'soil3': [(0.1, 'savanna:tree'), (0.2, 'savanna:bush'), (0.25, 'savanna:deadwood'), ],
    },  
    'Wetland': {  # few trees, few plants
        'grass2': [(0.1, 'lowland:tree'), (0.2, 'lowland:bush'), (0.25, 'lowland:deadwood'), ],
    },
    'Tropical seasonal forest': {  # trees, bushes and deadwood
        'grass2': [(0.4, 'tropical:tree'), (0.95, 'tropical:bush'), (1.0, 'tropical:deadwood'), ],
    },
    'Temperate deciduous forest': {  # average amount of trees
        'soil4': [(0.2, 'temperate:tree'), (0.95, 'temperate:bush'), (1.0, 'temperate:deadwood'), ],
    },
    'Tropical rainforest': {  # most amount of trees
        'grass2': [(0.4, 'tropical:tree'), (0.95, 'tropical:bush'), (1.0, 'tropical:deadwood'), ],
    },
    'Temperate rainforest': {  # average amount of trees, but more plants
        'grass1': [(0.2, 'temperate:tree'), (0.95, 'temperate:bush'), (1.0, 'temperate:deadwood'), ],
    },
    'Taiga': {  # trees, bushes, deadwood, mushrooms
        'soil5': [(0.4, 'taiga:tree'), (0.8, 'taiga:bush'), (0.95, 'taiga:mushroom'), (1.0, 'taiga:deadwood'), ],
        'grass1': [(0.4, 'taiga:tree'), (0.8, 'taiga:bush'), (0.95, 'taiga:mushroom'), (1.0, 'taiga:deadwood'), ],
    },
    'Tundra': {'soil3': [(0.25, 'taiga:tree'), ], },  # few trees
    'Hot desert': {'mud2': [(0.25, 'hot_desert:bush'), ], },  # few bushes only
    'Cold desert': {'soil6': [(0.25, 'cold_desert:bush'), ], },  # no trees, but plants
    'Glacier': {'snow3': [(0.125, 'glacier:frozen_plant'), (0.125, 'glacier:frozen_tree'), ], },  # few frozen plants or trees
    'Marine': {},  # no trees
}

roads_mapping = {
    'water5': ['dirt6', ],
    'cliff1': ['cliff2', ],
    'cliff2': ['dirt6', ],
    'dust1': ['stone1', ],
    'dirt1': ['stone1', ],
    'dirt2': ['stone1', ],
    'sand2': ['stone1', ],
    'sand2': ['stone1', ],
    'sand4': ['stone1', ],
    'grass1': ['stone1', ],
    'grass2': ['stone1', ],
    'grass3': ['stone1', ],
    'soil3': ['stone1', ],
    'soil4': ['stone1', ],
    'soil5': ['stone1', ],
}

inner_outer_transform_before_borders_list = [
    ('soil1', 'dirt2', None),
    ('soil3', 'dirt2', None),
    ('soil4', 'grass1', None),
    ('soil5', 'grass2', None),
    ('soil6', 'sand2', None),
    ('grass1', 'grass2', None),
    ('grass3', 'grass2', None),
    ('dirt1', 'dirt2', None),
    ('dirt4', 'dirt2', None),
    ('dust1', 'dirt2', None),
    ('snow1', 'snow2', None),
    ('snow2', 'snow3', None),
    ('snow3', 'cliff2', None),
    ('snow3', 'cliff1', None),
    ('mud2', 'sand2', None),
    ('water1', 'water5', None),
    ('water5', 'grass2', 'sand4'),
    ('soil1', 'stone1', 'dirt2'),
    ('soil3', 'stone1', 'dirt2'),
    ('soil5', 'stone1', 'dirt2'),
    ('soil6', 'stone1', 'dirt2'),
    ('grass1', 'stone1', 'grass2'),
    ('grass3', 'stone1', 'grass2'),
    ('dirt1', 'stone1', 'dirt2'),
    ('dust1', 'stone1', 'dirt2'),
    ('mud2', 'stone1', 'dirt2'),
]

inner_outer_transform_borders_list = [
    # ('cliff1', 'grass1', 'dirt2'),
    ('cliff1', 'grass2', 'cliff2'),
    ('cliff1', 'dirt2', 'cliff2'),
    ('cliff1', 'dirt6', 'cliff2'),
    ('cliff1', 'sand2', 'cliff2'),
    ('cliff1', 'sand4', 'cliff2'),
    ('cliff2', 'stone1', 'dirt6'),
    # ('cliff1', 'grass3', 'dirt2'),
    # ('cliff1', 'soil5', 'dirt2'),
    # ('cliff1', 'dirt6', 'sand2'),
    # ('cliff1', 'sand4', 'sand2'),
    # ('cliff1', 'water5', 'sand2'),
    # ('sand1', 'grass2', 'sand2'),
    # ('sand1', 'grass1', 'sand2'),
    # ('sand1', 'cliff2', 'sand2'),
    # ('sand1', 'dirt2', 'sand2'),
    # ('snow1', 'sand2', 'snow2'),
    # ('sand2', 'stone1', 'sand4'),
    # ('cliff2', 'grass2', 'grass1'),
    ('snow3', 'cliff2', 'cliff1'),
    ('snow3', 'dirt2', 'cliff1'),
    ('snow3', 'dirt6', 'cliff1'),
    ('snow3', 'sand4', 'sand2'),
    ('snow3', 'grass2', 'cliff1'),
    ('snow3', 'grass3', 'cliff1'),
    # ('grass1', 'soil5', 'grass2'),
    ('grass1', 'grass3', 'grass2'),
    ('grass1', 'sand4', 'grass2'),
    ('grass1', 'sand2', 'grass2'),
    ('grass1', 'dirt2', 'grass2'),
    ('grass1', 'stone1', 'dirt2'),
    # ('grass1', 'cliff2', 'dirt2'),
    # ('grass2', 'cliff2', 'sand2'),
    # ('grass3', 'sand1', 'dirt2'),
    # ('grass3', 'sand2', 'dirt2'),
    # ('grass2', 'cliff2', 'grass1'),
    # ('grass2', 'stone1', 'dirt6'),
    ('grass3', 'sand4', 'dirt2'),
    ('grass3', 'dirt6', 'grass2'),
    # ('dirt1', 'dirt6', 'sand4'),
    ('dirt2', 'dirt6', 'sand4'),
    # ('dirt4', 'dirt2', 'sand2'),
    # ('dust1', 'grass2', 'dirt2'),
    # ('dust1', 'sand2', 'dirt2'),
    ('dust1', 'sand4', 'dirt2'),
    # ('dust1', 'dirt6', 'dirt2'),
    ('soil3', 'sand2', 'dirt2'),
    ('soil3', 'sand4', 'dirt2'),
    # ('soil3', 'dirt6', 'dirt2'),
    ('soil4', 'grass2', 'grass1'),
    # ('soil4', 'sand2', 'grass1'),
    # ('soil4', 'cliff2', 'grass1'),
    ('soil4', 'dirt6', 'grass1'),
    # ('soil5', 'grass3', 'grass2'),
    ('soil5', 'sand2', 'grass2'),
    ('soil5', 'sand4', 'grass2'),
    ('soil5', 'grass1', 'grass2'),
    # ('soil5', 'sand4', 'grass2'),
    ('soil5', 'dirt6', 'grass2'),
    ('soil5', 'cliff2', 'dirt2'),
    ('soil5', 'stone1', 'dirt2'),
    # ('soil6', 'grass2', 'sand2'),
    # ('soil6', 'sand4', 'sand2'),
    ('soil6', 'cliff2', 'sand2'),
    # ('mud2', 'sand4', 'sand2'),
    ('mud2', 'dirt2', 'sand2'),
    ('mud2', 'sand4', 'sand2'),
    # ('mud2', 'grass2', 'sand2'),
    # ('water5', 'stone1', 'dirt6'),
    ('water5', 'cliff2', 'dirt6'),
    ('water5', 'sand1', 'sand4'),
    ('water5', 'sand2', 'sand4'),
    ('water5', 'grass1', 'grass2'),
    ('water5', 'grass3', 'dirt2'),
    ('water5', 'dirt2', 'sand4'),
    ('water5', 'dust1', 'dirt2'),
    ('water5', 'soil3', 'dirt2'),
    ('water5', 'soil4', 'grass1'),
    ('water5', 'soil5', 'grass2'),
    ('water5', 'mud2', 'sand2'),
    ('water5', 'stone1', 'dirt6'),
    # ('water1', 'sand1', 'water5'),
    # ('water1', 'sand2', 'water5'),
    # ('water1', 'grass1', 'water5'),
    ('water1', 'grass2', 'water5'),
    ('water1', 'sand4', 'water5'),
    ('water1', 'dirt6', 'water5'),
    ('water1', 'stone1', 'water5'),
    # ('water1', 'grass3', 'water5'),
    # ('water1', 'dirt1', 'water5'),
    # ('water1', 'dirt2', 'water5'),
    # ('water1', 'dust1', 'water5'),
    # ('water1', 'soil3', 'water5'),
    # ('water1', 'soil4', 'water5'),
    # ('water1', 'soil5', 'water5'),
    # ('water1', 'mud2', 'water5'),
    ('dirt6', 'sand2', 'sand4'),
]

transform_two_adjacent_diagonal_neighbors = [
    # ('grass1', 'grass2'),
    # ('soil4', 'grass1'),
    # ('snow3', 'snow2'),
    # ('dirt4', 'dirt2'),
    # ('grass2', 'sand2'),
    # ('soil5', 'grass2'),
    # ('dirt2', 'grass2'),
    # ('sand4', 'sand2'),
    # ('dirt2', 'sand2'),
    # ('grass2', 'grass3'),
    # ('grass2', 'water5', ),
    # ('sand2', 'water5', ),
    # ('sand2', 'water5'),
    # ('soil4', 'water5', ),
    # ('sand4', 'water5'),
    # ('water1', 'water5'),
]


def color_distance(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])


def xy2draw(x, y):
    global min_x, min_y, input_width, input_height, width, height
    return float(x - min_x) * float(width) / float(input_width), float(y - min_y) * float(height) / float(input_height)


def random_points_in_polygon(polygon_points, random_points_number):
    polygon = Polygon(polygon_points)
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < random_points_number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            if pnt not in points:
                points.append(pnt)
    return points


def read_full_fantasy_map_generator_json_file(file_path):
    raw = open(file_path, 'rt').read()
    return json.loads(raw)


def detect_bounds(data):
    vertices = data['pack']['vertices']
    min_x = min(v['p'][0] for v in vertices)
    max_x = max(v['p'][0] for v in vertices)
    min_y = min(v['p'][1] for v in vertices)
    max_y = max(v['p'][1] for v in vertices)
    return min_x, max_x, min_y, max_y


def detect_elevation_bounds(data):
    cells = data['pack']['cells']
    min_elevation = min(filter(None, [c['h'] for c in cells]))
    max_elevation = max(c['h'] for c in cells)
    return min_elevation, max_elevation


def enrich_data_with_tiles_mapping(data):
    # biomes_colors_array = data['biomesData']['color']
    biomes_names = data['biomesData']['name']
    tiles_stats = {}
    for i in range(len(data['pack']['cells'])):
        cell = data['pack']['cells'][i]
        h = cell['h']
        # hex_color = biomes_colors_array[cell['biome']].lstrip('#')
        # biome_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # best_color_dist = None
        best_biome = biomes_names[cell['biome']]
        # for c in biomes_colors.keys():
        #     diff_dist = color_distance(biome_color, c)
        #     if best_color_dist is None or diff_dist < best_color_dist:
        #         best_color_dist = diff_dist
        #         best_biome = biomes_colors[c]
        biome_tile = None
        cell_feature = data['pack']['features'][cell['f']]
        if cell_feature['type'] == 'lake':
            h = round(cell_feature['height'])
            if h >= CLIFFS_HEIGHT_MARGIN:
                best_biome = 'Glacier'
            biome_tile = 'water1'
        if not biome_tile:
            possible_tiles = biomes_mapping[best_biome]
            rnd = random.randint(0, 10000) / 10000.0
            for tile, chance in possible_tiles:
                if rnd <= chance:
                    biome_tile = tile
                    break
        # if True:
        #     if biome_tile in ['snow1', 'snow2', 'snow3', ]:
        #         if h < CLIFFS_HEIGHT_MARGIN - CLIFFS_HEIGHT_DROP:
        #             import pdb; pdb.set_trace()
        #             biome_tile = 'mud2'
        data['pack']['cells'][i]['tile'] = biome_tile
        tiles_stats[biome_tile] = tiles_stats.get(biome_tile, 0) + 1
    return tiles_stats


def render_biomes(data, draw):
    cells = data['pack']['cells']
    vertices = data['pack']['vertices']
    biomes_colors_data = data['biomesData']['color']
    count = 0
    for cell in cells:
        points = []
        hex_color = biomes_colors_data[cell['biome']].lstrip('#')
        for v_i in cell['v']:
            v = vertices[v_i]
            x, y = v['p']
            coord = xy2draw(x, y)
            points.append(coord)
        draw.polygon(points, fill=tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))
        count += 1
    return count


def render_cell_index(data, draw):
    cells = data['pack']['cells']
    vertices = data['pack']['vertices']
    count = 0
    for i in range(len(cells)):
        cell = cells[i]
        points = []
        if i // 256 > 255:
            raise Exception(f"Too many cells to encode in RGB")
        encoded_color = (i % 256, i // 256, 0)
        for v_i in cell['v']:
            v = vertices[v_i]
            x, y = v['p']
            coord = xy2draw(x, y)
            points.append(coord)
        draw.polygon(points, fill=encoded_color)
        count += 1
    return count


def capture_biomes_data(biome_image, biomes_colors):
    result_biomes_map = {}
    for x in range(biome_image.width):
        for y in range(biome_image.height):
            biome_pixel = biome_image.getpixel((x, y))
            biome_color = (int(biome_pixel[0]), int(biome_pixel[1]), int(biome_pixel[2]))
            best_color_dist = None
            best_biome = None
            for c in biomes_colors.keys():
                diff_dist = color_distance(biome_color, c)
                if best_color_dist is None or diff_dist < best_color_dist:
                    best_color_dist = diff_dist
                    best_biome = biomes_colors[c]
            result_biomes_map[(x, y)] = best_biome
    return result_biomes_map


def capture_tiles_data(tiles_image):
    tiles_map = {}
    for x in range(tiles_image.width):
        for y in range(tiles_image.height):
            tile_pixel = tiles_image.getpixel((x, y))
            tile_color = (int(tile_pixel[0]), int(tile_pixel[1]), int(tile_pixel[2]))
            tiles_map[(x, y)] = tiles_colors_reversed[tile_color]
    return tiles_map


def capture_cell_index_data(cell_index_image):
    cell_index_map = {}
    for x in range(cell_index_image.width):
        for y in range(cell_index_image.height):
            cell_index_pixel = cell_index_image.getpixel((x, y))
            c1 = int(cell_index_pixel[0])
            c2 = int(cell_index_pixel[1])
            cell_index_map[(x, y)] = c1 + c2 * 256
    return cell_index_map


def render_tiles(data, draw, tiles_colors):
    cells = data['pack']['cells']
    vertices = data['pack']['vertices']
    count = 0
    for cell in cells:
        points = []
        for v_i in cell['v']:
            v = vertices[v_i]
            x, y = v['p']
            coord = xy2draw(x, y)
            points.append(coord)
        biome_tile_color = tiles_colors[cell['tile']]
        draw.polygon(points, fill=biome_tile_color)
        count += 1
    return count


def render_routes(data, draw, routes_draw):
    cells = data['pack']['cells']
    count = 0
    for route in data['pack']['routes']:
        if route['group'] == 'searoutes':
            continue
        segments = []
        prev_x, prev_y, prev_cell_index = route['points'][0]
        for i in range(1, len(route['points'])):
            this_x, this_y, this_cell_index = route['points'][i]
            this_cell = cells[this_cell_index]
            this_tile = this_cell['tile']
            this_road_tiles_possible = roads_mapping.get(this_tile, ['stone1', ])
            this_road_tile = this_road_tiles_possible[0]
            segments.append((this_road_tile, xy2draw(prev_x, prev_y), xy2draw(this_x, this_y)))
            prev_x = this_x
            prev_y = this_y
        for segment in segments:
            road_tile, p1, p2 = segment
            road_color = tiles_colors[road_tile]
            draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=road_color, width=2)
            routes_draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(255, 0, 0) if route['group'] == 'roads' else (0, 0, 255), width=2)
        count += 1
    return count


def capture_routes_data(routes_image):
    routes_map = {}
    for x in range(routes_image.width):
        for y in range(routes_image.height):
            tile_pixel = routes_image.getpixel((x, y))
            is_route = tile_pixel[0] > 0 or tile_pixel[1] > 0 or tile_pixel[2] > 0
            if is_route:
                routes_map[(x, y)] = True
    return routes_map


def update_routes_elevation(heightmap_image, routes_map, min_route_elevation):
    changes = 0
    for x, y in routes_map.keys():
        this_pixel = heightmap_image.getpixel((x, y))
        if this_pixel[0] < min_route_elevation:
            heightmap_image.putpixel((x, y), (min_route_elevation, min_route_elevation, min_route_elevation))
            changes += 1
    return changes


def render_rivers(data, draw, rivers_draw):
    river_shallow_tile = 'dirt6'
    river_deep_tile = 'water5'
    cells = data['pack']['cells']
    count = 0
    for river in data['pack']['rivers']:
        if  river['type'] != 'River':
            continue
        points1 = []
        points2 = []
        points3 = []
        points4 = []
        sz = float(len(river['cells']))
        j = 0
        for c_i in river['cells']:
            j += 1
            if c_i < 0:
                continue
            c = cells[c_i]
            x, y = c['p']
            x_draw, y_draw = xy2draw(x, y)
            points1.append((x_draw, y_draw))
            if j > sz * 0.3:
                points2.append((x_draw, y_draw))
            if j > sz * 0.6:
                points3.append((x_draw, y_draw))
            if j > sz * 0.8:
                points4.append((x_draw, y_draw))
        draw.line(points3, fill=tiles_colors[river_deep_tile], width=2)
        rivers_draw.line(points3, fill=tiles_colors[river_deep_tile], width=2)
        draw.line(points2, fill=tiles_colors[river_shallow_tile], width=3)
        rivers_draw.line(points2, fill=tiles_colors[river_shallow_tile], width=3)
        draw.line(points1, fill=tiles_colors[river_shallow_tile], width=2)
        rivers_draw.line(points1, fill=tiles_colors[river_shallow_tile], width=2)
        draw.line(points4, fill=tiles_colors[river_shallow_tile], width=5)
        rivers_draw.line(points4, fill=tiles_colors[river_shallow_tile], width=5)
        draw.line(points3, fill=tiles_colors[river_shallow_tile], width=4)
        rivers_draw.line(points3, fill=tiles_colors[river_shallow_tile], width=4)
        draw.line(points4, fill=tiles_colors[river_deep_tile], width=3)
        rivers_draw.line(points4, fill=tiles_colors[river_deep_tile], width=3)
        count += 1
    return count


def capture_rivers_data(rivers_image):
    rivers_map = {}
    for x in range(rivers_image.width):
        for y in range(rivers_image.height):
            river_pixel = rivers_image.getpixel((x, y))
            is_river = river_pixel[0] > 0 or river_pixel[1] > 0 or river_pixel[2] > 0
            if is_river:
                rivers_map[(x, y)] = True
    return rivers_map


def update_rivers_elevation(heightmap_image, rivers_map, max_river_elevation):
    changes = 0
    for x, y in rivers_map.keys():
        h = heightmap_image.getpixel((x, y))[0]
        if h > max_river_elevation:
            h = max_river_elevation
            heightmap_image.putpixel((x, y), (h, h, h))
            changes += 1
    return changes


# def elevation_unpack(h):
#     if h > INPUT_WATER_LEVEL:
#         return pow(h - 18, ELEVATION_UNPACK_EXPONENT)
#     if h <= 0:
#         return -1 * (INPUT_WATER_LEVEL - 1) * ELEVATION_UNPACK_UNDERWATER_FACTOR
#     return (float(h - INPUT_WATER_LEVEL) / h) * float(ELEVATION_UNPACK_UNDERWATER_FACTOR)


# def elevation_to_scale_255(e):
#     global min_elevation_unpacked, max_elevation_unpacked, water_level_unpacked
#     delta = float(max_elevation_unpacked - min_elevation_unpacked)
#     return int(float(e - min_elevation_unpacked) * 255.0 / delta)


def render_heightmap(data, draw, packed_draw):
    biomes_names = data['biomesData']['name']
    cells = data['pack']['cells']
    vertices = data['pack']['vertices']
    bellow_water_count = 0
    above_cliffs_count = 0
    for cell in cells:
        points = []
        for v_i in cell['v']:
            v = vertices[v_i]
            x, y = v['p']
            points.append(xy2draw(x, y))
        h = cell['h']
        cell_feature = data['pack']['features'][cell['f']]
        if cell_feature['type'] == 'lake':
            # h = round(cell_feature['height']) - 2
            if round(cell_feature['height']) > LAKES_MAX_ELEVATION_LEVEL:
                h = LAKES_MAX_ELEVATION_LEVEL
        else:
            biome = biomes_names[cell['biome']]
            if biome == 'Marine' and h < INPUT_WATER_LEVEL:
                h = 1
        packed_draw.polygon(points, fill=(h, h, h))
        if True:
            if h > CLIFFS_HEIGHT_MARGIN - CLIFFS_HEIGHT_DROP and h <= CLIFFS_HEIGHT_MARGIN:
                h = CLIFFS_HEIGHT_MARGIN - CLIFFS_HEIGHT_DROP
        if False:
            if h < INPUT_WATER_LEVEL:
                h = 2
        if True:
            if h == INPUT_WATER_LEVEL:
                h = INPUT_WATER_LEVEL + 1
        if h < INPUT_WATER_LEVEL:
            bellow_water_count += 1
        if h > CLIFFS_HEIGHT_MARGIN:
            above_cliffs_count += 1
        # e = elevation_to_scale_255(elevation_unpack(h))
        draw.polygon(points, fill=(h, h, h))
    return bellow_water_count, above_cliffs_count


# def capture_elevation_data(heightmap_image):
#     _elevation_data = {}
#     for x in range(heightmap_image.width):
#         for y in range(heightmap_image.height):
#             heightmap_imagebiome_pixel = heightmap_image.getpixel((x, y))
#             # saceled_e = float(heightmap_imagebiome_pixel[0])
#             _elevation_data[(x, y)] = heightmap_imagebiome_pixel[0]
#     return _elevation_data


def build_beach_area(tiles_image, tiles_map):
    beach_area = set()
    for x in range(1, tiles_image.width-1):
        for y in range(1, tiles_image.height-1):
            if (x, y) in beach_area:
                continue
            if tiles_map[(x, y)] == 'water5':  #  and elevation_data.get((x, y), 0) <= INPUT_WATER_LEVEL:
                for xn, yn in [
                    (x-1, y-1),
                    (x-1, y),
                    (x-1, y+1),
                    (x,   y-1),
                    (x,   y+1),
                    (x+1, y-1),
                    (x+1, y),
                    (x+1, y+1),
                ]:
                    neighbor = tiles_map[(xn, yn)]
                    if neighbor != 'water5' and neighbor not in ['sand4', 'dirt6', 'grass2', 'dirt2']:
                        beach_area.add((x, y))
    for x, y in beach_area:
        tiles_map[(x, y)] = 'dirt6'
    return beach_area


def build_cliffs(tiles_image, tiles_map, heightmap_image):
    cliffs = set()
    for x in range(1, tiles_image.width-1):
        for y in range(1, tiles_image.height-1):
            if (x, y) in cliffs:
                continue
            h = heightmap_image.getpixel((x, y))[0]
            # h = elevation_data.get((x, y), 0)
            if h > CLIFFS_HEIGHT_MARGIN:
                for xn, yn in [
                    (x-1, y-1),
                    (x-1, y),
                    (x-1, y+1),
                    (x,   y-1),
                    (x,   y+1),
                    (x+1, y-1),
                    (x+1, y),
                    (x+1, y+1),
                ]:
                    # neighbor_h = elevation_data.get((xn, yn), 0)
                    neighbor_h = heightmap_image.getpixel((xn, yn))[0]
                    if neighbor_h <= CLIFFS_HEIGHT_MARGIN:
                        cliffs.add((x, y))
    for x, y in cliffs:
        tiles_map[(x, y)] = 'cliff1'
    return cliffs


def build_plants(data, tiles_image, tiles_map, cell_index_map, plants_catalog):
    cells = data['pack']['cells']
    biomes_names = data['biomesData']['name']
    result = []
    for x in range(tiles_image.width):
        for y in range(tiles_image.height):
            cell_index = cell_index_map[(x, y)]
            cell = cells[cell_index]
            biome = biomes_names[cell['biome']]
            if biome not in trees_biomes_mapping:
                continue
            for tile, possible_plants in trees_biomes_mapping[biome].items():
                if tiles_map[(x, y)] == tile:
                    rnd = random.randint(0, 10000) / 10000.0
                    selected_plant = None
                    for chance, plant_name in possible_plants:
                        if rnd <= chance:
                            selected_plant = plant_name
                            break
                    if not selected_plant:
                        continue
                    try:
                        plant_biome, plant_kind = selected_plant.split(':')
                        plant_variants = plants_catalog[plant_biome][plant_kind]
                        plant_variant = random.choice(plant_variants)
                        model_name, texture_name, coefs, _ = plant_variant.split('#')
                        plant_direction = random.randint(0, 360)
                        shift_x = random.randint(0, 100) / 100.0
                        shift_y = random.randint(0, 100) / 100.0
                        plant_x = x + shift_x
                        plant_y = y + shift_y
                        plant_encoded = f'{model_name}:{texture_name}:{coefs} {plant_x} {plant_y} {plant_direction}'
                        result.append(plant_encoded)
                    except Exception as e:
                        print(f"Error processing plant {selected_plant} for biome {biome} at {(x, y)}: {e}")
                        continue
    print(f"Planted {len(result)} trees")
    return result


def flood_bellow_water_tiles(tiles_image, tiles_map, heightmap_image, flood_level):
    flooded = 0
    for x in range(tiles_image.width):
        for y in range(tiles_image.height):
            h = heightmap_image.getpixel((x, y))[0]
            if h <= flood_level:
                if tiles_map[(x, y)] not in ['water1', 'water5', 'sand4', 'dirt6', ]:
                    tiles_map[(x, y)] = 'water5'
                    flooded += 1
    return flooded


def shallow_water_tiles(tiles_image, tiles_map, heightmap_image, shallow_level):
    changes = 0
    for x in range(tiles_image.width):
        for y in range(tiles_image.height):
            if tiles_map[(x, y)] in ['water1', 'water5']:
                h = heightmap_image.getpixel((x, y))[0]
                if h > shallow_level:
                    heightmap_image.putpixel((x, y), (shallow_level, shallow_level, shallow_level))
                    changes += 1
    return changes


def transform_inner_outer_areas_borders(tiles_image, tiles_map, routes_map, rivers_map):
    for inner, outer, transform in inner_outer_transform_before_borders_list:
        replacing_list = set()
        transform_list = set()
        for x in range(1, tiles_image.width-1):
            for y in range(1, tiles_image.height-1):
                center = tiles_map[(x, y)]
                if center == inner:
                    for xn, yn in [
                        (x-1, y-1),
                        (x-1, y),
                        (x-1, y+1),
                        (x,   y-1),
                        (x,   y+1),
                        (x+1, y-1),
                        (x+1, y),
                        (x+1, y+1),
                    ]:
                        neighbor = tiles_map[(xn, yn)]
                        if transform is not None:
                            if neighbor == outer and tiles_map[(x, y)] != transform:
                                transform_list.add((x, y))
                                break
                        else:
                            if neighbor != center and tiles_map[(x, y)] != outer:
                                replacing_list.add((x, y))
                                break
        if transform is not None:
            for x, y in transform_list:
                if (x, y) not in routes_map and (x, y) not in rivers_map:
                    tiles_map[(x, y)] = transform
            if transform_list:
                print(f"  transformed border line conditionally between {inner} and {outer} with {transform} length: {len(transform_list)}")
        else:
            for x, y in replacing_list:
                if (x, y) not in routes_map and (x, y) not in rivers_map:
                    tiles_map[(x, y)] = outer
            if replacing_list:
                print(f"  placed border line between {inner} and {outer} with {outer} length: {len(replacing_list)}")


def transform_neighboring_tiles_conditionally(tiles_image, tiles_map, routes_map, rivers_map, catalog, max_cycles=12):
    cycles = max_cycles
    progress = 1
    attempts = 0
    while progress and cycles:
        if attempts > 0:
            attempt_snapshot_image = Image.new("RGB", tiles_image.size, "black")
            for x in range(tiles_image.width):
                for y in range(tiles_image.height):
                    tile = tiles_map[(x, y)]
                    attempt_snapshot_image.putpixel((x, y), tiles_colors[tile])
            attempt_snapshot_image.save(f'/tmp/attempt{attempts}.png')
        attempts += 1
        cycles -= 1
        progress = 0
        if True:
            changes = 0
            for inner, outer, transform in inner_outer_transform_borders_list:
                transform_list = set()
                for x in range(0, tiles_image.width):
                    for y in range(0, tiles_image.height):
                        center = tiles_map[(x, y)]
                        if center == inner:
                            for xn, yn in [
                                (x-1, y-1),
                                (x-1, y),
                                (x-1, y+1),
                                (x,   y-1),
                                (x,   y+1),
                                (x+1, y-1),
                                (x+1, y),
                                (x+1, y+1),
                            ]:
                                if (xn, yn) not in tiles_map:
                                    continue
                                neighbor = tiles_map[(xn, yn)]
                                if neighbor == outer and center != transform:
                                    transform_list.add((x, y))
                                    break
                for x, y in transform_list:
                    if ( (x, y) not in routes_map and (x, y) not in rivers_map ) or attempts > 5:
                        tiles_map[(x, y)] = transform
                        changes += 1
                        progress += 1
            print(f"  transformed border line conditionally with {changes} changes, finished {attempts} attempt")
        if True:
            changes = 0
            for x in range(0, tiles_image.width-1):
                for y in range(0, tiles_image.height-1):
                    neighbors_counts = {}
                    neighbors_tiles = {}
                    for xd, yd in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        xn = x + xd
                        yn = y + yd
                        neighbor = tiles_map[(xn, yn)]
                        if neighbor not in neighbors_counts:
                            neighbors_counts[neighbor] = 0
                        neighbors_counts[neighbor] += 1
                        neighbors_tiles[(xd, yd)] = neighbor
                    if len(neighbors_counts) != 4:
                        continue
                    neighbors4 = list(neighbors_counts.keys())
                    possible_3_adjacent_tiles = []
                    for i in range(4):
                        neighbors_copy = list(neighbors4)
                        neighbors_copy.pop(i)
                        neighbors3 = sorted(neighbors_copy)
                        t1, t2, t3 = neighbors3
                        if f'{t1}_{t1}_{t2}_{t3}' in catalog:
                            possible_3_adjacent_tiles.append((t1, t2, t3))
                        if f'{t1}_{t2}_{t1}_{t3}' not in catalog:
                            possible_3_adjacent_tiles.append((t1, t2, t3))
                        if f'{t1}_{t1}_{t3}_{t2}' not in catalog:
                            possible_3_adjacent_tiles.append((t1, t2, t3))
                        if f'{t1}_{t3}_{t1}_{t2}' not in catalog:
                            possible_3_adjacent_tiles.append((t1, t2, t3))
                    if not possible_3_adjacent_tiles:
                        raise Exception(f"Did not find possible 3-adjacent tile for {neighbors4} at ({x}, {y})")
                    selected_3_tiles = possible_3_adjacent_tiles[random.randint(0, len(possible_3_adjacent_tiles) - 1)]
                    for xd, yd in neighbors_tiles.keys():
                        neighbor = neighbors_tiles[(xd, yd)]
                        if neighbor not in selected_3_tiles:
                            tiles_map[(x + xd, y + yd)] = selected_3_tiles[0]
                            changes += 1
                            progress += 1
            print(f"  transformed 4-adjacent neighbors with {changes} changes")
        if True:
            changes = 0
            previos_changes = set()
            for x in range(0, tiles_image.width-1):
                for y in range(0, tiles_image.height-1):
                    if (x, y) in previos_changes:
                        continue
                    neighbors_counts = {}
                    neighbors_tiles = {}
                    for xd, yd in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        xn = x + xd
                        yn = y + yd
                        neighbor = tiles_map[(xn, yn)]
                        if neighbor not in neighbors_counts:
                            neighbors_counts[neighbor] = 0
                        neighbors_counts[neighbor] += 1
                        neighbors_tiles[(xd, yd)] = neighbor
                    if len(neighbors_counts) != 2:
                        continue
                    diag1 = set([neighbors_tiles[(0, 0)], neighbors_tiles[(1, 1)]])
                    diag2 = set([neighbors_tiles[(1, 0)], neighbors_tiles[(0, 1)]])
                    if len(diag1) == 1 and len(diag2) == 1:
                        previos_changes.add((x, y))
                        previos_changes.add((x+1, y))
                        previos_changes.add((x, y+1))
                        previos_changes.add((x+1, y+1))
                        t00 = tiles_map[(x, y)]
                        t10 = tiles_map[(x+1, y)]
                        change = None
                        if attempts > 5:
                            change = random.randint(1, 2)
                        else:
                            for c1, c2 in transform_two_adjacent_diagonal_neighbors:
                                if c1 == t00 and c2 == t10:
                                    change = 1
                                elif c2 == t00 and c1 == t10:
                                    change = 2
                        if change is not None:
                            if change == 1:
                                tiles_map[(x, y)] = tiles_map[(x+1, y)]
                            else:
                                tiles_map[(x+1, y)] = tiles_map[(x, y)]
                        else:
                            if tiles_map[(x, y)] in ['sand4', 'sand2', 'dirt2', ]:
                                tiles_map[(x+1, y)] = tiles_map[(x, y)]
                            else:
                                tiles_map[(x, y)] = tiles_map[(x+1, y)]
                        changes += 1
                        progress += 1
            print(f"  transformed 2-adjacent diagonal neighbors with {changes} changes")


def build_tiles_puzzle(tiles_image, tiles_map, catalog):
    tiles = {}
    for xb in range(0, tiles_image.width-1):
        for yb in range(0, tiles_image.height-1):
            square = {}
            for xi, yi in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                xn = xb + xi
                yn = yb + yi
                square[(xi, yi)] = tiles_map[(xn, yn)]
            samples_all = list(set(square.values()))
            samples_sorted = list(sorted(list(set(square.values())), reverse=True))
            counts = {}
            coords = {}
            for xd, yd in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                tile = square[(xd, yd)]
                if tile not in counts:
                    counts[tile] = 0
                counts[tile] += 1
                if tile not in coords:
                    coords[tile] = []
                coords[tile].append((xd, yd))
            counts_sorted = sorted(counts.keys(), key=lambda n: counts[n], reverse=True)
            t1 = counts_sorted[0]
            col1 = set([square[(0, 0)], square[(1, 0)]])
            col2 = set([square[(0, 1)], square[(1, 1)]])
            diag1 = set([square[(0, 0)], square[(1, 1)]])
            diag2 = set([square[(1, 0)], square[(0, 1)]])
            x = xb
            y = yb
            if len(samples_all) == 1:
                k = samples_sorted[0]
                if k not in catalog:
                    raise Exception(f"Did not found {k} at ({x}, {y}) in the catalog")
                catalog_ids = catalog[k]
                tiles[(x, y)] = (catalog_ids[random.randint(0, len(catalog_ids) - 1)], 90 * random.randint(0, 3))
            elif len(samples_all) == 2:
                t2 = counts_sorted[1]
                k = f'{t2}_{t1}'
                if k not in catalog:
                    raise Exception(f"Did not found {k} at ({x}, {y}) in the catalog")
                corner_ids = catalog[k]['c']
                side_ids = catalog[k]['s']
                shape = None
                if counts[t1] == 3:
                    if len(diag1) == 1:
                        if len(col1) == 1:
                            shape = 'topleft_bottomright'
                        else:
                            shape = 'bottomright_topleft'
                    else:
                        if len(col1) == 1:
                            shape = 'topright_bottomleft'
                        else:
                            shape = 'bottomleft_topright'
                else:
                    if len(diag1) == 1 and len(diag2) == 1:
                        raise Exception(f"Found neighboring tiles with 2 different types in diagonal at ({x}, {y})")
                    else:
                        if coords[t1].count((0, 0)):
                            if len(col1) == 1:
                                shape = 'top_bottom'
                            else:
                                if coords[t1].count((0, 1)):
                                    shape = 'left_right'
                                else:
                                    shape = 'right_left'
                        elif coords[t1].count((1, 0)):
                            if len(col1) == 1:
                                shape = 'top_bottom'
                            else:
                                if coords[t1].count((1, 1)):
                                    shape = 'right_left'
                                else:
                                    shape = 'left_right'
                        elif coords[t1].count((0, 1)):
                            if len(col2) == 1:
                                shape = 'bottom_top'
                            else:
                                if coords[t1].count((0, 0)):
                                    shape = 'left_right'
                                else:
                                    shape = 'right_left'
                        elif coords[t1].count((1, 1)):
                            if len(col2) == 1:
                                shape = 'bottom_top'
                            else:
                                if coords[t1].count((1, 0)):
                                    shape = 'right_left'
                                else:
                                    shape = 'left_right'
                if shape == 'top_bottom':
                    tiles[(x, y)] = (side_ids[random.randint(0, len(side_ids) - 1)], 0)
                elif shape == 'right_left':
                    tiles[(x, y)] = (side_ids[random.randint(0, len(side_ids) - 1)], 270)
                elif shape == 'bottom_top':
                    tiles[(x, y)] = (side_ids[random.randint(0, len(side_ids) - 1)], 180)
                elif shape == 'left_right':
                    tiles[(x, y)] = (side_ids[random.randint(0, len(side_ids) - 1)], 90)
                elif shape == 'bottomleft_topright':
                    tiles[(x, y)] = (corner_ids[random.randint(0, len(corner_ids) - 1)], 270)
                elif shape == 'topleft_bottomright':
                    tiles[(x, y)] = (corner_ids[random.randint(0, len(corner_ids) - 1)], 0)
                elif shape == 'topright_bottomleft':
                    tiles[(x, y)] = (corner_ids[random.randint(0, len(corner_ids) - 1)], 90)
                elif shape == 'bottomright_topleft':
                    tiles[(x, y)] = (corner_ids[random.randint(0, len(corner_ids) - 1)], 180)
            elif len(samples_all) == 3:
                t2 = counts_sorted[1]
                t3 = counts_sorted[2]
                k1123 = f'{t1}_{t1}_{t2}_{t3}'
                if k1123 not in catalog:
                    raise Exception(f"Did not found {k1123} at ({x}, {y}) in the catalog")
                k1213 = f'{t1}_{t2}_{t1}_{t3}'
                if k1213 not in catalog:
                    raise Exception(f"Did not found {k1213} at ({x}, {y}) in the catalog")
                k1132 = f'{t1}_{t1}_{t3}_{t2}'
                if k1132 not in catalog:
                    raise Exception(f"Did not found {k1132} at ({x}, {y}) in the catalog")
                k1312 = f'{t1}_{t3}_{t1}_{t2}'
                if k1312 not in catalog:
                    raise Exception(f"Did not found {k1312} at ({x}, {y}) in the catalog")
                k12 = f'{t1}_{t2}'
                if k12 not in catalog:
                    raise Exception(f"Did not found {k12} at ({x}, {y}) in the catalog")
                k13 = f'{t1}_{t3}'
                if k13 not in catalog:
                    raise Exception(f"Did not found {k13} at ({x}, {y}) in the catalog")
                k23 = f'{t2}_{t3}'
                if k23 not in catalog:
                    raise Exception(f"Did not found {k23} at ({x}, {y}) in the catalog")
                catalog1123_ids = catalog[k1123]
                catalog1213_ids = catalog[k1213]
                catalog1132_ids = catalog[k1132]
                catalog1312_ids = catalog[k1312]
                shape = None
                if coords[t1].count((0, 0)):
                    if len(diag1) == 1:
                        if coords[t2].count((1, 0)):
                            shape = 1
                            tiles[(x, y)] = (catalog1213_ids[random.randint(0, len(catalog1213_ids) - 1)], 0)
                        else:
                            shape = 2
                            tiles[(x, y)] = (catalog1312_ids[random.randint(0, len(catalog1312_ids) - 1)], 0)
                    else:
                        if len(col1) == 1:
                            if coords[t2].count((0, 1)):
                                shape = 3
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 0)
                            else:
                                shape = 4
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 0)
                        else:
                            if coords[t2].count((1, 0)):
                                shape = 5
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 90)
                            else:
                                shape = 6
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 90)
                elif coords[t1].count((1, 0)):
                    if len(diag2) == 1:
                        if coords[t2].count((0, 0)):
                            shape = 7
                            tiles[(x, y)] = (catalog1312_ids[random.randint(0, len(catalog1312_ids) - 1)], 270)
                        else:
                            shape = 8
                            tiles[(x, y)] = (catalog1312_ids[random.randint(0, len(catalog1312_ids) - 1)], 270)
                    else:
                        if len(col1) == 1:
                            if coords[t2].count((0, 1)):
                                shape = 9
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 180)
                            else:
                                shape = 10
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 180)
                        else:
                            if coords[t2].count((0, 0)):
                                shape = 11
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 270)
                            else:
                                shape = 12
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 270)
                elif coords[t1].count((0, 1)):
                    if len(diag2) == 1:
                        if coords[t2].count((0, 0)):
                            shape = 13
                            tiles[(x, y)] = (catalog1312_ids[random.randint(0, len(catalog1312_ids) - 1)], 90)
                        else:
                            shape = 14
                            tiles[(x, y)] = (catalog1213_ids[random.randint(0, len(catalog1213_ids) - 1)], 90)
                    else:
                        if len(col2) == 1:
                            if coords[t2].count((0, 0)):
                                shape = 15
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 180)
                            else:
                                shape = 16
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 180)
                        else:
                            if coords[t2].count((1, 0)):
                                shape = 17
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 0)
                            else:
                                shape = 18
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 0)
                elif coords[t1].count((1, 1)):
                    if len(diag1) == 1:
                        if coords[t2].count((1, 0)):
                            shape = 19
                            tiles[(x, y)] = (catalog1213_ids[random.randint(0, len(catalog1213_ids) - 1)], 0)
                        else:
                            shape = 20
                            tiles[(x, y)] = (catalog1312_ids[random.randint(0, len(catalog1312_ids) - 1)], 0)
                    else:
                        if len(col2) == 1:
                            if coords[t2].count((0, 0)):
                                shape = 21
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 0)
                            else:
                                shape = 22
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 0)
                        else:
                            if coords[t2].count((0, 0)):
                                shape = 23
                                tiles[(x, y)] = (catalog1123_ids[random.randint(0, len(catalog1123_ids) - 1)], 270)
                            else:
                                shape = 24
                                tiles[(x, y)] = (catalog1132_ids[random.randint(0, len(catalog1132_ids) - 1)], 270)
            else:
                raise Exception(f"Did not found a shape for {samples_all} at ({x}, {y})")
    return tiles


def main():
    global min_x, min_y, input_width, input_height, width, height
    global min_elevation, max_elevation

    random.seed(1)

    catalog = json.loads(open('assets/catalog.json', 'rt').read())
    catalog_plants = json.loads(open('assets/catalog_plants.json', 'rt').read())

    biomes_map = {}
    tiles_map = {}
    cell_index_map = {}
    routes_map = {}
    rivers_map = {}

    singles = set()
    pairs = set()
    triplets = set()
    for k in catalog.keys():
        parts = k.split('_')
        t = tuple(sorted(list(set(parts))))
        if len(t) == 1:
            singles.add(t)
        elif len(t) == 2:
            pairs.add(t)
        elif len(t) == 3:
            triplets.add(t)
        else:
            raise Exception(f"Unexpected parts count in {t}")
    print(f"Catalog has {len(singles)} single tiles, {len(pairs)} pairs and {len(triplets)} triplets")

    json_file_path = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    data = read_full_fantasy_map_generator_json_file(json_file_path)
    print(f"Read JSON data from {json_file_path}")

    min_x, max_x, min_y, max_y = detect_bounds(data)
    input_width = max_x - min_x
    input_height = max_y - min_y
    print(f"Bounds are x=({min_x}:{max_x}) y=({min_y}:{max_y}) width={input_width} height={input_height}")

    min_elevation, max_elevation = detect_elevation_bounds(data)
    # min_elevation_unpacked = elevation_unpack(min_elevation)
    # max_elevation_unpacked = elevation_unpack(max_elevation)
    # water_level_unpacked = elevation_unpack(INPUT_WATER_LEVEL)
    print(f"Elevations are from {min_elevation} to {max_elevation}, water level is {INPUT_WATER_LEVEL}")

    biomes_colors_data = data['biomesData']['color']
    water_color = biomes_colors_data[0].lstrip('#')
    print(f'Found {len(biomes_colors_data)} biomes colors, water color is #{water_color}')

    # deep = elevation_to_scale_255(elevation_unpack(1))
    heightmap_image = Image.new("RGB", (width, height), (min_elevation, min_elevation, min_elevation))
    heightmap_draw = ImageDraw.Draw(heightmap_image)
    heightmap_packed_image = Image.new("RGB", (width, height), (0, 0, 0))
    heightmap_packed_draw = ImageDraw.Draw(heightmap_packed_image)
    bellow_water_count, above_cliffs_count = render_heightmap(data, heightmap_draw, heightmap_packed_draw)
    heightmap_packed_image.save('assets/heightmap_packed.png')
    # heightmap_image = heightmap_image.filter(ImageFilter.GaussianBlur(radius=1.0))
    # elevation_data = capture_elevation_data(heightmap_image)
    print(f'Rendered heightmap, {bellow_water_count} tiles are bellow water level, {above_cliffs_count} tiles are above cliffs, min elevation is {min_elevation}, max elevation is {max_elevation}')

    tiles_stats = enrich_data_with_tiles_mapping(data)
    different_tiles = list(tiles_stats.keys())
    different_tiles.sort(key=lambda i: tiles_stats[i], reverse=True)
    print('Enriched biomes with tiles mapping:')
    for i in range(len(different_tiles)):
        print(f"  {different_tiles[i]}: {tiles_stats[different_tiles[i]]} cells")

    biome_image = Image.new("RGB", (width, height), tuple(int(water_color[i:i+2], 16) for i in (0, 2, 4)))
    biome_draw = ImageDraw.Draw(biome_image)
    biomes_count = render_biomes(data, biome_draw)
    biome_image.save('assets/biome.png')
    biomes_map = capture_biomes_data(biome_image, biomes_colors)
    print(f'Rendered {biomes_count} biomes')

    # tiles_stats = enrich_data_with_tiles_mapping(data)

    if heightmap_image.size != biome_image.size:
        raise Exception("Height map and biome map sizes do not match")

    tiles_image = None

    if os.path.exists('assets/tiles.png'):
        tiles_image = Image.open('assets/tiles.png')
        tiles_image.load()
        print(f'Loaded existing tiles image')

        rivers_image = Image.open('assets/rivers.png')
        rivers_image.load()
        print(f'Loaded existing rivers image')

        routes_image = Image.open('assets/routes.png')
        routes_image.load()
        print(f'Loaded existing routes image')

    else:
        tiles_image = Image.new("RGB", (width, height), tiles_colors['water5'])
        tiles_draw = ImageDraw.Draw(tiles_image)
        biomes_count = render_tiles(data, tiles_draw, tiles_colors)
        print(f'Rendered {biomes_count} biomes')

        rivers_image = Image.new("RGB", (width, height), 'black')
        rivers_draw = ImageDraw.Draw(rivers_image)
        rivers_count = render_rivers(data, tiles_draw, rivers_draw)
        rivers_image.save('assets/rivers.png')
        print(f'Rendered {rivers_count} rivers')

        routes_image = Image.new("RGB", (width, height), "black")
        routes_draw = ImageDraw.Draw(routes_image)
        routes_count = render_routes(data, tiles_draw, routes_draw)
        print(f'Rendered {routes_count} routes at {len(routes_map)} tiles')

        routes_image.save('assets/routes.png')
        tiles_image.save('assets/tiles_original.png')
        
    routes_map = capture_routes_data(routes_image)
    print(f'Captured routes data with {len(routes_map)} tiles')

    rivers_map = capture_rivers_data(rivers_image)
    print(f'Captured rivers data with {len(rivers_map)} tiles')

    cell_index_image = Image.new("RGB", (width, height), "black")
    cell_index_draw = ImageDraw.Draw(cell_index_image)
    cells_count = render_cell_index(data, cell_index_draw)
    cell_index_image.save('assets/cell_index.png')
    cell_index_map = capture_cell_index_data(cell_index_image)
    print(f'Rendered cell index image and indexed {cells_count} cells coordinates')

    if os.path.exists('assets/tiles.png'):
        tiles_map = capture_tiles_data(tiles_image)
        print(f'Read {len(tiles_map)} tiles from existing image')

    else:
        tiles_map = capture_tiles_data(tiles_image)
        print(f'Read {len(tiles_map)} tiles from the rendered image')

        # scaled_water_level = elevation_to_scale_255(elevation_unpack(INPUT_WATER_LEVEL))
        flooded = flood_bellow_water_tiles(tiles_image, tiles_map, heightmap_image, flood_level=INPUT_WATER_LEVEL)
        print(f"Flooded {flooded} tiles based on heightmap data, water level is {INPUT_WATER_LEVEL}")

        # beach_area = build_beach_area(tiles_image, tiles_map)
        # print(f"Added {len(beach_area)} beach area tiles")

        # cliffs = build_cliffs(tiles_image, tiles_map, heightmap_image)
        # print(f"Added {len(cliffs)} cliff tiles")

        print(f"Transforming inner-outer areas borders based on {len(inner_outer_transform_before_borders_list)} rules:")
        transform_inner_outer_areas_borders(tiles_image, tiles_map, routes_map, rivers_map)

        print(f"Transform neighboring tiles conditionally:")
        transform_neighboring_tiles_conditionally(tiles_image, tiles_map, routes_map, rivers_map, catalog, max_cycles=TRANSFORM_CYCLES)

        stats = {}
        tiles_image_output = Image.new("RGB", biome_image.size, "black")
        for x in range(tiles_image.width):
            for y in range(tiles_image.height):
                tile = tiles_map[(x, y)]
                tiles_image_output.putpixel((x, y), tiles_colors[tile])
                stats[tile] = stats.get(tile, 0) + 1
        tiles_image_output.save('assets/tiles.png')
        print(f"Generated tiles image")

    missing_links = {}
    for x in range(1, tiles_image.width-1):
        for y in range(1, tiles_image.height-1):
            center = tiles_map[(x, y)]
            for xn, yn in [
                (x-1, y-1),
                (x-1, y),
                (x-1, y+1),
                (x,   y-1),
                (x,   y+1),
                (x+1, y-1),
                (x+1, y),
                (x+1, y+1),
            ]:
                neighbor = tiles_map[(xn, yn)]
                if neighbor != center:
                    pair = tuple(sorted([center, neighbor]))
                    k = f"{pair[0]}_{pair[1]}"
                    if k not in catalog:
                        if pair not in missing_links:
                            missing_links[pair] = (x, y)

    if missing_links:
        print(f"Found missing catalog items between different tile types:")
        print('  ' + ('\n  '.join([f'{pair[0]} - {pair[1]} at {coord[0]}:{coord[1]}' for pair, coord in missing_links.items()])))
        raise Exception("Missing links detected")

    for x in range(0, tiles_image.width-1):
        for y in range(0, tiles_image.height-1):
            neighbors_counts = {}
            neighbors_tiles = {}
            for xd, yd in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                xn = x + xd
                yn = y + yd
                neighbor = tiles_map[(xn, yn)]
                if neighbor not in neighbors_counts:
                    neighbors_counts[neighbor] = 0
                neighbors_counts[neighbor] += 1
                neighbors_tiles[(xd, yd)] = neighbor
            if len(neighbors_counts) == 4:
                raise Exception(f"Found neighboring tiles with 4 different types at ({x}, {y})")
            if len(neighbors_counts) == 2:
                diag1 = set([neighbors_tiles[(0, 0)], neighbors_tiles[(1, 1)]])
                diag2 = set([neighbors_tiles[(1, 0)], neighbors_tiles[(0, 1)]])
                if len(diag1) == 1 and len(diag2) == 1:
                    raise Exception(f"Found neighboring tiles with 2 different types in diagonal at ({x}, {y}): {list(neighbors_counts.keys())}")

    # changes = shallow_water_tiles(tiles_image, tiles_map, heightmap_image, shallow_level=INPUT_WATER_LEVEL-1)
    # print(f"Updated heightmap to set shallow of the bellow water tiles with {changes} changes")

    changes = update_rivers_elevation(heightmap_image, rivers_map, max_river_elevation=RIVERS_MAX_ELEVATION_LEVEL)
    print(f"Updated rivers elevation with {changes} changes")

    changes = update_routes_elevation(heightmap_image, routes_map, min_route_elevation=ROUTES_MIN_ELEVATION_LEVEL)
    print(f"Updated routes elevation with {changes} changes")

    heightmap_image.save('assets/heightmap_raw.png')
    # heightmap_image = heightmap_image.filter(ImageFilter.GaussianBlur(radius=0.75))
    heightmap_image = heightmap_image.filter(ImageFilter.BoxBlur(radius=1))

    changes = update_rivers_elevation(heightmap_image, rivers_map, max_river_elevation=RIVERS_MAX_ELEVATION_LEVEL)
    print(f"Updated rivers elevation second time with {changes} changes")

    changes = update_routes_elevation(heightmap_image, routes_map, min_route_elevation=ROUTES_MIN_ELEVATION_LEVEL)
    print(f"Updated routes elevation second time with {changes} changes")

    heightmap_image.save('assets/heightmap.png')
    print(f"Generated heightmap image, elevations are from {min_elevation} to {max_elevation}")

    tiles = build_tiles_puzzle(tiles_image, tiles_map, catalog)
    print(f"Built tiles puzzle with {len(tiles)} tiles")

    water_catalog_id = catalog['water5'][0]
    encoded_image = Image.new("RGB", (tiles_image.size[0], tiles_image.size[1]), "black")
    catalog_stats = {}
    for x in range(0, encoded_image.width):
        for y in range(0, encoded_image.height):
            catalog_id, rotate = tiles[(x, y)] if (x, y) in tiles else (water_catalog_id, 0)
            catalog_id = int(catalog_id)
            rotate = int(rotate)
            if catalog_id is not None:
                encoded_image.putpixel((x, y), (catalog_id % 256, catalog_id // 256, rotate // 90))
                catalog_stats[catalog_id] = catalog_stats.get(catalog_id, 0) + 1
    open('assets/catalog_stats.json', 'w').write(json.dumps(catalog_stats, indent=2))
    encoded_image.save('assets/encoded.png')
    print(f"Generated encoded tiles image")


    data = read_full_fantasy_map_generator_json_file(sys.argv[1])
    plants = build_plants(data, tiles_image, tiles_map, cell_index_map, catalog_plants)
    open('assets/plants.json', 'w').write(json.dumps(plants, indent=2))

    different_biomes = list(stats.keys())
    different_biomes.sort(key=lambda i: stats[i], reverse=True)
    for i in range(len(different_biomes)):
        print(f"  {i+1:02d}. {different_biomes[i]}: {stats[different_biomes[i]]} tiles")


if __name__ == '__main__':
    main()
