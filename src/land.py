import os
import sys
import json
import pprint
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from shapely.geometry import Point, Polygon
import numpy as np

min_x = 0
min_y = 0
input_width = 0
input_height = 0
width = 0
height = 0
max_elevation = 0


INPUT_WATER_LEVEL = 20      # 0 is the water level in the input heightmap, 100 is the input max height
BEACH_COAST_LINE_HEIGHT_MAX = 16  # 0 is the water level after heightmap translation, 255 is max height


def xy2draw(x, y):
    global min_x, min_y, input_width, input_height, width, height
    return float(x - min_x) * float(width) / float(input_width), float(y - min_y) * float(height) / float(input_height)


def quantize_coefs(coefs, quant_size=0.5):
    return [round(round(c / quant_size, 0) * quant_size, 1) for c in coefs]


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


def read_full_json_file(file_path):
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
    min_elevation = min(c['h'] for c in cells)
    max_elevation = max(c['h'] for c in cells)
    return min_elevation, max_elevation


def render_biomes(data, draw):
    cells = data['pack']['cells']
    vertices = data['pack']['vertices']
    biomes_colors = data['biomesData']['color']
    count = 0
    for cell in cells:
        points = []
        hex_color = biomes_colors[cell['biome']].lstrip('#')
        for v_i in cell['v']:
            v = vertices[v_i]
            x, y = v['p']
            coord = xy2draw(x, y)
            points.append(coord)
        draw.polygon(points, fill=tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))
        count += 1
    return count


def render_routes(data, draw):
    count = 0
    for route in data['pack']['routes']:
        points = []
        for p in route['points']:
            x, y, _ = p
            points.append(xy2draw(x, y))
        if route['group'] == 'trails':
            c = (192, 192, 192)
        elif route['group'] == 'searoutes':
            c = (64, 64, 192)
        elif route['group'] == 'roads':
            c = (192, 128, 64)
        draw.line(points, fill=c, width=1)
        count += 1
    return count


def render_rivers(data, draw):
    cells = data['pack']['cells']
    count = 0
    for route in data['pack']['rivers']:
        points = []
        for c_i in route['cells']:
            if c_i < 0:
                continue
            c = cells[c_i]
            x, y = c['p']
            points.append(xy2draw(x, y))
        draw.line(points, fill=(64, 64, 255), width=4)
        count += 1
    return count


def render_heightmap(data, draw, max_elevation, water_level):
    cells = data['pack']['cells']
    vertices = data['pack']['vertices']
    diff = float(max_elevation - water_level)
    for cell in cells:
        points = []
        h = cell['h']
        if h < water_level:
            h = water_level
        for v_i in cell['v']:
            v = vertices[v_i]
            x, y = v['p']
            points.append(xy2draw(x, y))
        e = int((h - water_level) * 255.0 / diff)
        draw.polygon(points, fill=(e, e, e))


def read_json_file(json_file_path, output_width, output_height):
    global min_x, min_y, input_width, input_height, width, height, max_elevation
    width = int(output_width)
    height = int(output_height)
    data = read_full_json_file(json_file_path)
    min_x, max_x, min_y, max_y = detect_bounds(data)
    input_width = max_x - min_x
    input_height = max_y - min_y
    print(f"Bounds are x=({min_x}:{max_x}) y=({min_y}:{max_y}) width={input_width} height={input_height}")
    min_elevation, max_elevation = detect_elevation_bounds(data)
    print(f"Elevations are from {min_elevation} to {max_elevation}")
    biomes_colors = data['biomesData']['color']
    water_color = biomes_colors[0].lstrip('#')
    print(f'Output size is {width}:{height}, water color is #{water_color}')
    biome_image = Image.new("RGB", (width, height), tuple(int(water_color[i:i+2], 16) for i in (0, 2, 4)))
    biome_draw = ImageDraw.Draw(biome_image)
    heightmap_image = Image.new("RGB", (width, height), "black")
    heightmap_draw = ImageDraw.Draw(heightmap_image)
    biomes_count = render_biomes(data, biome_draw)
    print(f'Rendered {biomes_count} biomes')
    # routes_count = render_routes(data, biome_draw)
    # print(f'Rendered {routes_count} routes')
    # rivers_count = render_rivers(data, biome_draw)
    # print(f'Rendered {rivers_count} rivers')
    render_heightmap(data, heightmap_draw, max_elevation, water_level=INPUT_WATER_LEVEL)
    heightmap_image = heightmap_image.filter(ImageFilter.GaussianBlur(radius=1))
    biome_image.save('biome.png')
    heightmap_image.save('heightmap.png')
    return biome_image, heightmap_image, biomes_colors


def plant_trees(data):
    trees_biomes_mapping = {
        'Tropical seasonal forest': (10.0, ['hills',]),
        'Temperate deciduous forest': (10.0, ['hills', 'fields',]),
        'Tropical rainforest': (20.0, ['hills',]),
        'Temperate rainforest': (10.0, ['coast',]),
        'Taiga': (1.0, ['hills']),
        'Hot desert': (1.0, ['desert',]),
        'Glacier': (1.0, ['winter',]),
        'Cold desert': (1.0, ['winter',]),
        'Grassland': (1.0, ['fields',]),
    }
    trees_registry = {}
    trees_variants = {}
    for model_name, variants in json.loads(open('models.json', 'rt').read()).items():
        for variant in variants:
            modl = variant['m']
            tex = variant['t']
            coefs = quantize_coefs(variant['c'])
            kind = variant['k']
            land_types = variant['b']
            if not kind:
                continue
            if kind == 'tree':
                tree_variant_key = f'{modl}:{tex}:{coefs[0]}:{coefs[1]}:{coefs[2]}'
                if tree_variant_key not in trees_variants:
                    trees_variants[tree_variant_key] = {
                        'm': modl,
                        't': tex,
                        'c': coefs,
                        'k': tree_variant_key,
                    }
                for land_type in land_types:
                    if land_type not in trees_registry:
                        trees_registry[land_type] = []
                    if tree_variant_key not in trees_registry[land_type]:
                        trees_registry[land_type].append(tree_variant_key)
                    # trees_registry[land_type].append({
                    #     'm': model_name,
                    #     'c': coefs,
                    #     't': tex,
                    #     'k': tree_variant_key,
                    # })
    cells = data['pack']['cells']
    vertices = data['pack']['vertices']
    biomes_names = data['biomesData']['name']
    trees = []
    for cell in cells:
        points = []
        biome = biomes_names[cell['biome']]
        if biome not in trees_biomes_mapping:
            continue
        for v_i in cell['v']:
            v = vertices[v_i]
            x, y = v['p']
            coord = xy2draw(x, y)
            points.append(list(coord))
        density, land_types = trees_biomes_mapping[biome]
        trees_in_cell_number = int(random.random() * density)
        random_points = random_points_in_polygon(points, random_points_number=trees_in_cell_number)
        tree_variants = []
        for land_type in land_types:
            tree_variants.extend(trees_registry[land_type])
        for p in random_points:
            t_variant = random.choice(tree_variants)
            t = dict(trees_variants[t_variant])
            t['x'] = p.x
            t['y'] = p.y
            trees.append(t)
    print(f"Planted {len(trees)} trees")
    return trees, trees_variants


def color_distance(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])


def main():
    catalog = json.loads(open('catalog.json', 'rt').read())

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

    biome_image, heightmap_image, biomes_colors = read_json_file(
        json_file_path=sys.argv[1],
        output_width=sys.argv[2],
        output_height=sys.argv[3],
    )

    if heightmap_image.size != biome_image.size:
        raise Exception("Height map and biome map sizes do not match")

    print(f"Map size: {heightmap_image.size}")
    # see: https://github.com/vesellov/worlds/blob/main/catalog.json
    avarage_colors = {
        'cliff1': (124, 123, 110),
        'cliff10': (42, 57, 81),
        'cliff2': (131, 129, 103),
        'cliff3': (107, 105, 71),
        'cliff4': (116, 120, 83),
        'cliff5': (89, 88, 72),
        'cliff6': (120, 99, 84),
        'cliff7': (50, 50, 46),
        'cliff8': (122, 90, 87),
        'cliff9': (53, 78, 39),
        'dirt1': (119, 104, 72),
        'dirt2': (119, 95, 72),
        'dirt3': (104, 76, 49),
        'dirt4': (118, 101, 86),
        'dirt5': (82, 66, 49),
        'dirt6': (101, 104, 102),
        'dirt7': (57, 56, 50),
        'dust1': (121, 101, 78),
        'dust2': (78, 65, 51),
        'grass1': (28, 61, 0),
        'grass2': (44, 76, 0),
        'grass3': (66, 76, 0),
        'grass4': (34, 60, 12),
        'lava1': (163, 36, 9),
        'lava2': (99, 0, 0),
        'mud1': (154, 92, 31),
        'mud2': (141, 102, 31),
        'mud3': (111, 68, 43),
        'mud4': (130, 62, 30),
        'rock1': (84, 83, 80),
        'rock2': (59, 60, 52),
        'sand1': (192, 186, 123),
        'sand2': (192, 173, 123),
        'sand4': (165, 142, 105),
        'sand5': (163, 142, 94),
        'snow1': (209, 219, 251),
        'snow2': (210, 217, 230),
        'snow3': (245, 245, 252),
        'snow4': (227, 229, 239),
        'soil1': (116, 113, 76),
        'soil2': (102, 101, 47),
        'soil3': (98, 91, 49),
        'soil4': (56, 61, 26),
        'soil5': (88, 78, 18),
        'stone1': (108, 101, 88),
        'stone2': (115, 99, 84),
        'stone3': (70, 71, 61),
        'stone4': (115, 106, 96),
        'stone5': (177, 166, 132),
        'stone6': (164, 127, 97),
        'stone7': (143, 87, 56),
        'stone8': (50, 45, 31),
        'tile1': (81, 85, 69),
        'tile2': (95, 103, 76),
        'water1': (44, 130, 200),
        'water2': (65, 126, 113),
        'water3': (102, 158, 140),
        'water4': (0, 97, 82),
        'water5': (53, 76, 99),
        'water6': (42, 63, 89),
        'water7': (56, 72, 99),
        'water8': (5, 36, 57),
    }
    # see: https://github.com/Azgaar/Fantasy-Map-Generator/blob/master/src/modules/biomes.ts#L12
    biomes_types = {
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
    biomes_types = {tuple(int(k[i:i+2], 16) for i in (0, 2, 4)):v for k, v in biomes_types.items()}
    biomes_mapping = {
        'Marine': 'water5',
        'Hot desert': 'sand2',
        'Cold desert': 'dirt1',
        'Savanna': 'dirt2',
        'Grassland': 'soil3',
        'Tropical seasonal forest': 'grass3',
        'Temperate deciduous forest': 'grass2',
        'Tropical rainforest': 'grass1',
        'Temperate rainforest': 'soil4',
        'Taiga': 'dirt4',
        'Tundra': 'soil1',
        'Glacier': 'snow3',
        'Wetland': 'soil5',
    }
    inner_outer_transform_before_borders_list = [
        ('soil3', 'dirt2', None),
        ('soil5', 'grass2', None),
        ('soil4', 'dirt2', None),
        ('dirt1', 'dirt2', None),
        ('dirt4', 'dirt2', None),
        ('snow3', 'dirt2', None),
        ('snow3', 'dirt4', None),
        ('water5', 'sand4', None),
        ('soil1', 'dirt2', None),
    ]
    biomes_map = {}
    tiles_map = {}
    coast_line = set()

    for x in range(biome_image.width):
        for y in range(biome_image.height):
            biome_pixel = biome_image.getpixel((x, y))
            biome_color = (int(biome_pixel[0]), int(biome_pixel[1]), int(biome_pixel[2]))
            best_color_dist = None
            best_biome = None
            for c in biomes_types.keys():
                diff_dist = color_distance(biome_color, c)
                if best_color_dist is None or diff_dist < best_color_dist:
                    best_color_dist = diff_dist
                    best_biome = biomes_types[c]
            biomes_map[(x, y)] = best_biome
            tiles_map[(x, y)] = biomes_mapping[best_biome]

    for inner, outer, transform in inner_outer_transform_before_borders_list:
        replacing_list = set()
        transform_list = set()
        for x in range(1, biome_image.width-1):
            for y in range(1, biome_image.height-1):
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
                tiles_map[(x, y)] = transform
            if transform_list:
                print(f"Transformed border line conditionally between {inner} and {outer} with {transform} length: {len(transform_list)}")
        else:
            for x, y in replacing_list:
                tiles_map[(x, y)] = outer
            if replacing_list:
                print(f"Placed border line between {inner} and {outer} with {outer} length: {len(replacing_list)}")

    if False:
        for x in range(1, biome_image.width-1):
            for y in range(1, biome_image.height-1):
                center = tiles_map[(x, y)]
                if center != 'water5':
                    continue
                for neighbor in [
                    tiles_map[(x-1, y-1)],
                    tiles_map[(x-1, y)],
                    tiles_map[(x-1, y+1)],
                    tiles_map[(x, y-1)],
                    tiles_map[(x, y+1)],
                    tiles_map[(x+1, y-1)],
                    tiles_map[(x+1, y)],
                    tiles_map[(x+1, y+1)],                
                ]:
                    if neighbor != center:
                        if (x, y) not in coast_line:
                            coast_line.add((x, y))
        for x, y in coast_line:
            tiles_map[(x, y)] = 'sand4'
        print(f"Coast line length: {len(coast_line)}")

    if False:
        # diff = float(max_elevation - INPUT_WATER_LEVEL)
        coast_max_height = BEACH_COAST_LINE_HEIGHT_MAX
        print(f"Coast line max height: {coast_max_height} (for coast line input height {coast_max_height})")
        beach_area = set()
        cycles = 3
        progress = 1
        while progress and cycles:
            cycles -= 1
            progress = 0
            for x in range(1, biome_image.width-1):
                for y in range(1, biome_image.height-1):
                    tile = tiles_map[(x, y)]
                    if tile == 'water5':
                        continue
                    height = int(tuple(heightmap_image.getpixel((x, y)))[0])
                    if height > coast_max_height:
                        continue
                    for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                        xn = x + dx
                        yn = y + dy
                        if (xn, yn) in coast_line or (xn, yn) in beach_area:
                            if (x, y) not in coast_line and (x, y) not in beach_area:
                                beach_area.add((x, y))
                                progress += 1
                                break
            print(f"Detecting beach area, progress: {progress} pixels added")
    
        for x in range(1, biome_image.width-1):
            for y in range(1, biome_image.height-1):
                if (x, y) not in coast_line:
                    continue
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
                    neighbor_tile = tiles_map[(xn, yn)]
                    if neighbor_tile == 'water5':
                        continue
                    if (xn, yn) in beach_area:
                        continue
                    if (xn, yn) in coast_line:
                        continue
                    beach_area.add((xn, yn))
        for x, y in beach_area:
            tiles_map[(x, y)] = 'sand4'
        print(f"Beach area pixels total: {len(beach_area)}")

    if False:
        progress = 1
        attempts = 0
        while progress:
            attempts += 1
            if attempts > 10:
                break
            progress = 0
            for x in range(1, biome_image.width-1):
                for y in range(1, biome_image.height-1):
                    center = tiles_map[(x, y)]
                    neighbors_counts = {center: 1}
                    neighbors_tiles = {(0, 0): center}
                    for xd, yd in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        xn = x + xd
                        yn = y + yd
                        neighbor = tiles_map[(xn, yn)]
                        if neighbor not in neighbors_counts:
                            neighbors_counts[neighbor] = 0
                        neighbors_counts[neighbor] += 1
                        neighbors_tiles[(xd, yd)] = neighbor
                    if len(neighbors_counts) < 4:
                        continue
                    neighbors_sorted = sorted(neighbors_counts.keys(), key=lambda n: neighbors_counts[n], reverse=True)
                    tile1 = neighbors_sorted[0]
                    if tile1 != center:
                        tiles_map[(x, y)] = tile1
                        progress += 1
            print(f"Smoothing 4-adjacent neighbors, attempt #{attempts} with {progress} changes")

    if False:
        coast_line.clear()
        for x in range(1, biome_image.width-1):
            for y in range(1, biome_image.height-1):
                center = tiles_map[(x, y)]
                if center != 'water5':
                    continue
                for neighbor in [
                    tiles_map[(x-1, y-1)],
                    tiles_map[(x-1, y)],
                    tiles_map[(x-1, y+1)],
                    tiles_map[(x, y-1)],
                    tiles_map[(x, y+1)],
                    tiles_map[(x+1, y-1)],
                    tiles_map[(x+1, y)],
                    tiles_map[(x+1, y+1)],                
                ]:
                    if neighbor != center:
                        if (x, y) not in coast_line:
                            coast_line.add((x, y))
        for x, y in coast_line:
            tiles_map[(x, y)] = 'sand4'
        print(f"Coast line length in second round: {len(coast_line)}")

    cycles = 3
    progress = 1
    attempts = 0
    while progress and cycles:
        attempts += 1
        cycles -= 1
        progress = 0

        changes = 0
        for x in range(0, biome_image.width-1):
            for y in range(0, biome_image.height-1):
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
                    t00 = tiles_map[(x, y)]
                    t10 = tiles_map[(x+1, y)]
                    change = None
                    for c1, c2 in [
                        ('snow3', 'dirt4'),
                        ('dirt4', 'dirt2'),
                    ]:
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
                        if tiles_map[(x, y)] in ['sand4', 'dirt2', 'grass2']:
                            tiles_map[(x+1, y)] = tiles_map[(x, y)]
                        else:
                            tiles_map[(x, y)] = tiles_map[(x+1, y)]
                    changes += 1 
                    progress += 1
        print(f"Smoothing 2-adjacent diagonal neighbors with {changes} changes")

        for inner, outer, transform in [
            ('soil4', 'grass2', 'dirt2'),
            ('dirt4', 'grass2', 'dirt2'),
            ('snow3', 'dirt2', 'dirt4'),
        ]:
            transform_list = set()
            for x in range(1, biome_image.width-1):
                for y in range(1, biome_image.height-1):
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
                            if neighbor == outer and center != transform:
                                transform_list.add((x, y))
                                break
            for x, y in transform_list:
                tiles_map[(x, y)] = transform
                progress += 1
            print(f"Transform border line conditionally between {inner} and {outer} with {transform} length: {len(transform_list)}")

    stats = {}
    tiles_image = Image.new("RGB", biome_image.size, "black")
    for x in range(biome_image.width):
        for y in range(biome_image.height):
            tile = tiles_map[(x, y)]
            tiles_image.putpixel((x, y), avarage_colors[tile])
            stats[tile] = stats.get(tile, 0) + 1
    tiles_image.save('tiles.png')

    missing_links = {}
    for x in range(1, biome_image.width-1):
        for y in range(1, biome_image.height-1):
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
        print(f"Missing links between tiles:")
        print('  ' + ('\n  '.join([f'{pair[0]} - {pair[1]} at {coord[0]}:{coord[1]}' for pair, coord in missing_links.items()])))
        raise Exception("Missing links detected")

    for x in range(0, biome_image.width-1):
        for y in range(0, biome_image.height-1):
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

    fragment_x = 595
    fragment_y = 672
    fragment_width = 64
    fragment_height = 64

    tiles = {}
    for xb in range(0, biome_image.width-1):
        for yb in range(0, biome_image.height-1):
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
                        # if coords[t1].count((0, 0)):
                        #     shape = 'top_left_bottom_right'
                        # else:
                        #     shape = 'top_right_bottom_left'
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

    encoded_image = Image.new("RGB", (biome_image.size[0], biome_image.size[1]), "black")
    for x in range(0, encoded_image.width-1):
        for y in range(0, encoded_image.height-1):
            catalog_id, rotate = tiles[(x, y)] if (x, y) in tiles else (None, None)
            if catalog_id is not None:
                encoded_image.putpixel((x, y), (catalog_id % 256, catalog_id // 256, rotate // 90))
    encoded_image.save('encoded.png')

    fragment_image = Image.new("RGB", (fragment_width * 64, fragment_height * 64), "black")
    for x in range(fragment_x, fragment_x + fragment_width):
        for y in range(fragment_y, fragment_y + fragment_height):
            catalog_id, rotate = tiles[(x, y)] if (x, y) in tiles else (None, None)
            if catalog_id is not None:
                catalog_image = Image.open(os.path.join('textures', 'land', f'{catalog_id:05d}.png'))
                fragment_image.paste(catalog_image.rotate(rotate), ((x - fragment_x) * 64, (y - fragment_y) * 64))
    fragment_image.save('fragment.png')

    data = read_full_json_file(sys.argv[1])
    trees_list, trees_variants = plant_trees(data)
    open('trees.json', 'w').write(json.dumps(trees_list, indent=2))
    open('trees_variants.json', 'w').write(json.dumps(trees_variants, indent=2))

    different_biomes = list(stats.keys())
    different_biomes.sort(key=lambda i: stats[i], reverse=True)
    for i in range(len(different_biomes)):
        print(f"  {i+1:02d}. {different_biomes[i]}: {stats[different_biomes[i]]} tiles")


if __name__ == '__main__':
    main()
