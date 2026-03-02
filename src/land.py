import os
import sys
import json
import pprint
import random

from PIL import Image, ImageChops
import numpy as np


def color_distance(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])


def main():
    catalog = json.loads(open('catalog.json', 'rt').read())
    heightmap_image = Image.open(sys.argv[1])
    biome_image = Image.open(sys.argv[2])
    if heightmap_image.size != biome_image.size:
        raise Exception("Height map and biome map sizes do not match")

    print(f"Map size: {heightmap_image.size}")

    # registry = dict(
    #     cliff = dict(height=None, min_height=0.05, max_height=1.0, min_elevation=0.1, max_elevation=1.0, tiles=['cliff2']),
    #     dirt = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.05, tiles=['dirt2', 'dirt1']),
    #     # dust = dict(min_height=1, max_height=65535, min_elevation=0, max_elevation=100, tiles=),
    #     grass = dict(height=0.5, min_height=0.001, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['grass1', 'grass2', 'grass3']),
    #     # lava = dict(min_height=0, max_height=0, min_elevation=0, max_elevation=0, tiles=['lava1', 'lava2']),
    #     # mud = dict(min_height=1, max_height=65535, min_elevation=0, max_elevation=10, tiles=['mud1', 'mud2', 'mud3', 'mud4']),
    #     rock = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['rock2']),
    #     sand = dict(height=0.1, min_height=0, max_height=0.5, min_elevation=0, max_elevation=0.1, tiles=['dirt6', 'sand4', 'sand2']),
    #     # snow = dict(min_height=10, max_height=65535, min_elevation=0, max_elevation=65535),
    #     soil = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['soil5']),
    #     stone = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['stone1']),
    #     # tile = dict(min_height=1, max_height=65535, min_elevation=0, max_elevation=10),
    #     water = dict(height=0, min_height=0, max_height=0, min_elevation=0, max_elevation=0, tiles=['water5']),
    # )
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
    biome_tiles = {
        'ocean': 'water5',
        'coast': 'dirt6',
        'beach': 'sand2',
        'grassland': 'grass2',
        'temperate_deciduous_forest': 'grass2',
        'temperate_rainforest': 'grass2',
        'tropical_seasonal_forest': 'grass2',
        'tropical_rainforest': 'grass2',
        'wetland': 'grass2',
        'savanna': 'dirt2',
        'hot_desert': 'sand4',
        'cold_desert': 'dirt1',
        'taiga': 'soil5',
        'tundra': 'dirt2',
        'glacier': 'cliff2',
    }
    biomes = {
        (128, 155, 198): 'ocean',
        (182, 217, 93): 'tropical_seasonal_forest',
        (210, 208, 130): 'savanna',
        (41, 188, 86): 'temperate_deciduous_forest',
        (251, 231, 159): 'hot_desert',
        (64, 156, 67): 'temperate_rainforest',
        (125, 203, 53): 'tropical_rainforest',
        (200, 214, 143): 'grassland',
        (11, 145, 49): 'wetland',
        (181, 184, 135): 'cold_desert',
        (75, 107, 50): 'taiga',
        (150, 120, 75): 'tundra',
        (213, 231, 235): 'glacier',
    }
    min_height = None
    max_height = None
    heights = set()
    for x in range(heightmap_image.width):
        for y in range(heightmap_image.height):
            height_pixel = tuple(heightmap_image.getpixel((x, y)))
            height = int(height_pixel[0])
            if height not in heights:
                heights.add(height)
            if min_height is None or height < min_height:
                min_height = height
            if max_height is None or height > max_height:
                max_height = height
    print(f"Min height: {min_height}, Max height: {max_height}, Total unique heights: {len(heights)}")

    land = {}
    for x in range(biome_image.width):
        for y in range(biome_image.height):
            biome_pixel = biome_image.getpixel((x, y))
            biome_color = (int(biome_pixel[0]), int(biome_pixel[1]), int(biome_pixel[2]))
            best_color_dist = None
            best_biome = None
            for c in biomes.keys():
                diff_dist = color_distance(biome_color, c)
                if best_color_dist is None or diff_dist < best_color_dist:
                    best_color_dist = diff_dist
                    best_biome = biomes[c]
            if x not in land:
                land[x] = {}
            land[x][y] = best_biome

    # for biome in [
    #     # 'grassland',
    #     # 'temperate_deciduous_forest',
    #     # 'temperate_rainforest',
    #     # 'tropical_seasonal_forest',
    #     # 'tropical_rainforest',
    #     # 'wetland',
    #     'savanna',
    #     'hot_desert',
    #     'cold_desert',
    #     'taiga',
    #     'tundra',
    #     'glacier',
    # ]:
    #     biome_line = set()
    #     for y in range(1, biome_image.height-1):
    #         for x in range(1, biome_image.width-1):
    #             center = land[x][y]
    #             if center != biome:
    #                 continue
    #             neighbors = [
    #                 (y-1, x-1),
    #                 (y-1, x),
    #                 (y-1, x+1),
    #                 (y,   x-1),
    #                 (y,   x+1),
    #                 (y+1, x-1),
    #                 (y+1, x),
    #                 (y+1, x+1),
    #             ]
    #             for yn, xn in neighbors:
    #                 neighbor = land[xn][yn]
    #                 if neighbor != center:
    #                     if (x, y) not in biome_line:
    #                         biome_line.add((x, y))
    #     for x, y in biome_line:
    #         land[x][y] = 'glacier'
    #     print(f"Biome {biome} borders total length: {len(biome_line)}")

    coast_line = set()
    for x in range(1, biome_image.width-1):
        for y in range(1, biome_image.height-1):
            center = biome_tiles[land[x][y]]
            if center != 'water5':
                continue
            neighbors = [
                biome_tiles[land[x-1][y-1]],
                biome_tiles[land[x-1][y]],
                biome_tiles[land[x-1][y+1]],
                biome_tiles[land[x][y-1]],
                biome_tiles[land[x][y+1]],
                biome_tiles[land[x+1][y-1]],
                biome_tiles[land[x+1][y]],
                biome_tiles[land[x+1][y+1]],
            ]
            for neighbor in neighbors:
                if neighbor != center:
                    if (x, y) not in coast_line:
                        coast_line.add((x, y))
    for x, y in coast_line:
        land[x][y] = 'coast'
    print(f"Coast line length: {len(coast_line)}")

    beach_area = set()
    coast_max_height = 64
    progress = 1
    while progress:
        progress = 0
        for x in range(1, biome_image.width-1):
            for y in range(1, biome_image.height-1):
                tile = biome_tiles[land[x][y]]
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
    for x in range(1, biome_image.width-1):
        for y in range(1, biome_image.height-1):
            if land[x][y] != 'coast':
                continue
            neighbors = [
                (x-1, y-1),
                (x-1, y),
                (x-1, y+1),
                (x,   y-1),
                (x,   y+1),
                (x+1, y-1),
                (x+1, y),
                (x+1, y+1),
            ]
            for xn, yn in neighbors:
                neighbor_tile = biome_tiles[land[xn][yn]]
                if neighbor_tile == 'water5':
                    continue
                if (xn, yn) in beach_area:
                    continue
                if (xn, yn) in coast_line:
                    continue
                beach_area.add((xn, yn))
    for x, y in beach_area:
        land[x][y] = 'beach'
    print(f"Beach area pixels total: {len(beach_area)}")

    progress = 1
    attempts = 0
    while progress:
        attempts += 1
        if attempts > 10:
            break
        progress = 0
        for x in range(1, biome_image.width-1):
            for y in range(1, biome_image.height-1):
                center = biome_tiles[land[x][y]]
                neighbors_counts = {center: 1}
                neighbors_tiles = {(0, 0): center}
                neighbors_lands = {(0, 0): land[x][y]}
                neighbors_lands_counts = {land[x][y]: 1}
                for xd, yd in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    xn = x + xd
                    yn = y + yd
                    neighbor = biome_tiles[land[xn][yn]]
                    if neighbor not in neighbors_counts:
                        neighbors_counts[neighbor] = 0
                    neighbors_counts[neighbor] += 1
                    neighbors_tiles[(xd, yd)] = neighbor
                    neighbors_lands[(xd, yd)] = land[xn][yn]
                    if land[xn][yn] not in neighbors_lands_counts:
                        neighbors_lands_counts[land[xn][yn]] = 0
                    neighbors_lands_counts[land[xn][yn]] += 1
                if len(neighbors_counts) < 4:
                    continue
                neighbors_sorted = sorted(neighbors_lands_counts.keys(), key=lambda n: neighbors_lands_counts[n], reverse=True)
                land1 = neighbors_sorted[0]
                if land1 != land[x][y]:
                    land[x][y] = land1
                    progress += 1
        print(f"Smoothing 4-adjacent neighbors, attempt #{attempts} with {progress} changes")

    coast_line = set()
    for x in range(1, biome_image.width-1):
        for y in range(1, biome_image.height-1):
            center = biome_tiles[land[x][y]]
            if center != 'water5':
                continue
            neighbors = [
                biome_tiles[land[x-1][y-1]],
                biome_tiles[land[x-1][y]],
                biome_tiles[land[x-1][y+1]],
                biome_tiles[land[x][y-1]],
                biome_tiles[land[x][y+1]],
                biome_tiles[land[x+1][y-1]],
                biome_tiles[land[x+1][y]],
                biome_tiles[land[x+1][y+1]],
            ]
            for neighbor in neighbors:
                if neighbor != center:
                    if (x, y) not in coast_line:
                        coast_line.add((x, y))
    for x, y in coast_line:
        land[x][y] = 'coast'
    print(f"Coast line length: {len(coast_line)}")

    progress = 1
    attempts = 0
    while progress:
        attempts += 1
        if attempts > 10:
            break
        progress = 0
        for x in range(0, biome_image.width-1):
            for y in range(0, biome_image.height-1):
                neighbors_counts = {}
                neighbors_tiles = {}
                neighbors_lands = {}
                neighbors_lands_counts = {}
                for xd, yd in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    xn = x + xd
                    yn = y + yd
                    neighbor = biome_tiles[land[xn][yn]]
                    if neighbor not in neighbors_counts:
                        neighbors_counts[neighbor] = 0
                    neighbors_counts[neighbor] += 1
                    neighbors_tiles[(xd, yd)] = neighbor
                    neighbors_lands[(xd, yd)] = land[xn][yn]
                    if land[xn][yn] not in neighbors_lands_counts:
                        neighbors_lands_counts[land[xn][yn]] = 0
                    neighbors_lands_counts[land[xn][yn]] += 1
                if len(neighbors_counts) != 2:
                    continue
                diag1 = set([neighbors_tiles[(0, 0)], neighbors_tiles[(1, 1)]])
                diag2 = set([neighbors_tiles[(1, 0)], neighbors_tiles[(0, 1)]])
                if len(diag1) == 1 and len(diag2) == 1:
                    land[x+1][y+1] = land[x+1][y]
                    progress += 1
        print(f"Smoothing 2-adjacent diagonal neighbors, attempt #{attempts} with {progress} changes")

    stats = {}
    tiles_image = Image.new("RGB", biome_image.size, "black")
    for x in range(biome_image.width):
        for y in range(biome_image.height):
            tile = biome_tiles[land[x][y]]
            tiles_image.putpixel((x, y), avarage_colors[tile])
            stats[tile] = stats.get(tile, 0) + 1
    tiles_image.save(sys.argv[3])

    missing_links = set()
    for x in range(1, biome_image.width-1):
        for y in range(1, biome_image.height-1):
            center_biome = land[x][y]
            center = biome_tiles[center_biome]
            neighbors = [
                (x-1, y-1),
                (x-1, y),
                (x-1, y+1),
                (x,   y-1),
                (x,   y+1),
                (x+1, y-1),
                (x+1, y),
                (x+1, y+1),
            ]
            for xn, yn in neighbors:
                neighbor = biome_tiles[land[xn][yn]]
                if neighbor != center:
                    pair = tuple(sorted([center, neighbor]))
                    k = f"{pair[0]}_{pair[1]}"
                    if k not in catalog:
                        missing_links.add(pair)
    if missing_links:
        print(f"Missing links between tiles:")
        print('  ' + ('\n  '.join([f'{a} - {b}' for a, b in missing_links])))
        raise Exception("Missing links detected")

    for x in range(0, biome_image.width-1):
        for y in range(0, biome_image.height-1):
            neighbors_counts = {}
            neighbors_tiles = {}
            for xd, yd in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                xn = x + xd
                yn = y + yd
                neighbor = biome_tiles[land[xn][yn]]
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
                    raise Exception(f"Found neighboring tiles with 2 different types in diagonal at ({x}, {y})")

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
                square[(xi, yi)] = biome_tiles[land[xn][yn]]
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
    encoded_image.save(sys.argv[4])

    fragment_image = Image.new("RGB", (fragment_width * 64, fragment_height * 64), "black")
    for x in range(fragment_x, fragment_x + fragment_width):
        for y in range(fragment_y, fragment_y + fragment_height):
            catalog_id, rotate = tiles[(x, y)] if (x, y) in tiles else (None, None)
            if catalog_id is not None:
                catalog_image = Image.open(os.path.join('textures', 'land', f'{catalog_id:05d}.png'))
                fragment_image.paste(catalog_image.rotate(rotate), ((x - fragment_x) * 64, (y - fragment_y) * 64))
    fragment_image.save(sys.argv[5])

    different_biomes = list(stats.keys())
    different_biomes.sort(key=lambda i: stats[i], reverse=True)
    for i in range(len(different_biomes)):
        print(f"  {i+1:02d}. {different_biomes[i]}: {stats[different_biomes[i]]} tiles")


if __name__ == '__main__':
    main()
