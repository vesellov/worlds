import sys
import json
import pprint
import random

from PIL import Image, ImageChops
import numpy as np


def color_distance(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])


def main():
    heightmap_image = Image.open(sys.argv[1])
    biome_image = Image.open(sys.argv[2])
    if heightmap_image.size != biome_image.size:
        raise Exception("Height map and biome map sizes do not match")

    print(f"Map size: {heightmap_image.size}")

    tiles_image = Image.new("RGB", biome_image.size, "black")

    registry = dict(
        cliff = dict(height=None, min_height=0.05, max_height=1.0, min_elevation=0.1, max_elevation=1.0, tiles=['cliff2']),
        dirt = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.05, tiles=['dirt2', 'dirt1']),
        # dust = dict(min_height=1, max_height=65535, min_elevation=0, max_elevation=100, tiles=),
        grass = dict(height=0.5, min_height=0.001, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['grass1', 'grass2', 'grass3']),
        # lava = dict(min_height=0, max_height=0, min_elevation=0, max_elevation=0, tiles=['lava1', 'lava2']),
        # mud = dict(min_height=1, max_height=65535, min_elevation=0, max_elevation=10, tiles=['mud1', 'mud2', 'mud3', 'mud4']),
        rock = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['rock2']),
        sand = dict(height=0.1, min_height=0, max_height=0.5, min_elevation=0, max_elevation=0.1, tiles=['dirt6', 'sand4', 'sand2']),
        # snow = dict(min_height=10, max_height=65535, min_elevation=0, max_elevation=65535),
        soil = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['soil5']),
        stone = dict(height=None, min_height=0.01, max_height=1.0, min_elevation=0, max_elevation=0.1, tiles=['stone1']),
        # tile = dict(min_height=1, max_height=65535, min_elevation=0, max_elevation=10),
        water = dict(height=0, min_height=0, max_height=0, min_elevation=0, max_elevation=0, tiles=['water5']),
    )
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
    pairs = [
        ('cliff2', 'dirt1'),
        ('cliff2', 'dirt2'),
        ('cliff2', 'grass1'),
        ('cliff2', 'grass2'),
        ('cliff2', 'grass3'),
        ('cliff2', 'rock2'),
        ('cliff2', 'sand2'),
        ('cliff2', 'sand4'),
        ('cliff2', 'soil5'),
        ('cliff2', 'water5'),
        ('dirt1', 'cliff2'),
        ('dirt1', 'dirt2'),
        ('dirt1', 'grass2'),
        ('dirt1', 'sand2'),
        ('dirt1', 'sand4'),
        ('dirt1', 'soil5'),
        ('dirt2', 'cliff2'),
        ('dirt2', 'dirt1'),
        ('dirt2', 'dust1'),
        ('dirt2', 'grass1'),
        ('dirt2', 'grass2'),
        ('dirt2', 'grass3'),
        ('dirt2', 'rock2'),
        ('dirt2', 'sand1'),
        ('dirt2', 'sand2'),
        ('dirt2', 'sand4'),
        ('dirt2', 'soil3'),
        ('dirt2', 'soil5'),
        ('dirt2', 'stone1'),
        ('dirt2', 'water5'),
        ('dirt6', 'grass1'),
        ('dirt6', 'grass2'),
        ('dirt6', 'sand2'),
        ('dirt6', 'sand4'),
        ('dirt6', 'water5'),
        ('dust1', 'dirt2'),
        ('grass1', 'cliff2'),
        ('grass1', 'dirt2'),
        ('grass1', 'dirt6'),
        ('grass1', 'grass2'),
        ('grass1', 'grass3'),
        ('grass1', 'sand2'),
        ('grass2', 'cliff2'),
        ('grass2', 'dirt1'),
        ('grass2', 'dirt2'),
        ('grass2', 'dirt6'),
        ('grass2', 'grass1'),
        ('grass2', 'grass3'),
        ('grass2', 'sand2'),
        ('grass2', 'sand4'),
        ('grass2', 'soil3'),
        ('grass2', 'soil5'),
        ('grass2', 'stone1'),
        ('grass3', 'cliff2'),
        ('grass3', 'dirt2'),
        ('grass3', 'grass1'),
        ('grass3', 'grass2'),
        ('grass3', 'soil5'),
        ('rock2', 'cliff2'),
        ('rock2', 'dirt2'),
        ('sand1', 'dirt2'),
        ('sand1', 'sand2'),
        ('sand1', 'sand4'),
        ('sand2', 'cliff2'),
        ('sand2', 'dirt1'),
        ('sand2', 'dirt2'),
        ('sand2', 'dirt6'),
        ('sand2', 'grass1'),
        ('sand2', 'grass2'),
        ('sand2', 'sand1'),
        ('sand2', 'sand4'),
        ('sand2', 'stone1'),
        ('sand4', 'cliff2'),
        ('sand4', 'dirt1'),
        ('sand4', 'dirt2'),
        ('sand4', 'dirt6'),
        ('sand4', 'grass2'),
        ('sand4', 'sand1'),
        ('sand4', 'sand2'),
        ('sand4', 'stone1'),
        ('soil3', 'dirt2'),
        ('soil3', 'grass2'),
        ('soil5', 'cliff2'),
        ('soil5', 'dirt1'),
        ('soil5', 'dirt2'),
        ('soil5', 'grass2'),
        ('soil5', 'grass3'),
        ('stone1', 'dirt2'),
        ('stone1', 'grass2'),
        ('stone1', 'sand2'),
        ('stone1', 'sand4'),
        ('stone1', 'water5'),
        ('water5', 'cliff2'),
        ('water5', 'dirt2'),
        ('water5', 'dirt6'),
        ('water5', 'stone1'),
    ]
    quadruples = [
        ('cliff2', 'cliff2', 'dirt2', 'grass1'),
        ('cliff2', 'cliff2', 'dirt2', 'sand2'),
        ('cliff2', 'cliff2', 'grass1', 'dirt2'),
        ('cliff2', 'cliff2', 'grass1', 'sand2'),
        ('cliff2', 'cliff2', 'sand2', 'dirt2'),
        ('cliff2', 'cliff2', 'sand2', 'grass1'),
        ('dirt1', 'dirt1', 'dirt2', 'grass2'),
        ('dirt1', 'dirt1', 'dirt2', 'sand4'),
        ('dirt1', 'dirt1', 'grass2', 'dirt2'),
        ('dirt1', 'dirt1', 'grass2', 'sand4'),
        ('dirt1', 'dirt1', 'sand4', 'dirt2'),
        ('dirt1', 'dirt1', 'sand4', 'grass2'),
        ('dirt2', 'cliff2', 'dirt2', 'grass1'),
        ('dirt2', 'cliff2', 'dirt2', 'grass2'),
        ('dirt2', 'cliff2', 'dirt2', 'grass3'),
        ('dirt2', 'cliff2', 'dirt2', 'sand2'),
        ('dirt2', 'dirt1', 'dirt2', 'sand2'),
        ('dirt2', 'dirt2', 'cliff2', 'grass1'),
        ('dirt2', 'dirt2', 'cliff2', 'grass2'),
        ('dirt2', 'dirt2', 'cliff2', 'rock2'),
        ('dirt2', 'dirt2', 'cliff2', 'sand2'),
        ('dirt2', 'dirt2', 'dirt1', 'grass2'),
        ('dirt2', 'dirt2', 'grass1', 'cliff2'),
        ('dirt2', 'dirt2', 'grass1', 'grass3'),
        ('dirt2', 'dirt2', 'grass2', 'cliff2'),
        ('dirt2', 'dirt2', 'grass2', 'dirt1'),
        ('dirt2', 'dirt2', 'grass2', 'grass3'),
        ('dirt2', 'dirt2', 'grass2', 'sand2'),
        ('dirt2', 'dirt2', 'grass2', 'sand4'),
        ('dirt2', 'dirt2', 'grass2', 'soil3'),
        ('dirt2', 'dirt2', 'grass2', 'soil5'),
        ('dirt2', 'dirt2', 'grass2', 'stone1'),
        ('dirt2', 'dirt2', 'grass3', 'grass1'),
        ('dirt2', 'dirt2', 'grass3', 'grass2'),
        ('dirt2', 'dirt2', 'grass3', 'soil5'),
        ('dirt2', 'dirt2', 'rock2', 'cliff2'),
        ('dirt2', 'dirt2', 'sand1', 'sand2'),
        ('dirt2', 'dirt2', 'sand2', 'cliff2'),
        ('dirt2', 'dirt2', 'sand2', 'grass2'),
        ('dirt2', 'dirt2', 'sand2', 'sand1'),
        ('dirt2', 'dirt2', 'sand2', 'sand4'),
        ('dirt2', 'dirt2', 'sand4', 'grass2'),
        ('dirt2', 'dirt2', 'sand4', 'sand2'),
        ('dirt2', 'dirt2', 'soil3', 'grass2'),
        ('dirt2', 'dirt2', 'soil5', 'grass2'),
        ('dirt2', 'dirt2', 'soil5', 'grass3'),
        ('dirt2', 'dirt2', 'stone1', 'grass2'),
        ('dirt2', 'grass1', 'dirt2', 'cliff2'),
        ('dirt2', 'grass1', 'dirt2', 'sand2'),
        ('dirt2', 'grass2', 'dirt2', 'cliff2'),
        ('dirt2', 'grass2', 'dirt2', 'sand2'),
        ('dirt2', 'grass2', 'dirt2', 'sand4'),
        ('dirt2', 'grass2', 'dirt2', 'soil3'),
        ('dirt2', 'grass3', 'dirt2', 'cliff2'),
        ('dirt2', 'sand2', 'dirt2', 'cliff2'),
        ('dirt2', 'sand2', 'dirt2', 'dirt1'),
        ('dirt2', 'sand2', 'dirt2', 'grass1'),
        ('dirt2', 'sand2', 'dirt2', 'grass2'),
        ('dirt2', 'sand4', 'dirt2', 'grass2'),
        ('dirt2', 'soil3', 'dirt2', 'grass2'),
        ('grass1', 'dirt2', 'grass1', 'grass2'),
        ('grass1', 'grass1', 'grass2', 'grass3'),
        ('grass1', 'grass1', 'grass3', 'grass2'),
        ('grass1', 'grass2', 'grass1', 'dirt2'),
        ('grass2', 'dirt1', 'grass2', 'dirt2'),
        ('grass2', 'dirt1', 'grass2', 'sand2'),
        ('grass2', 'dirt1', 'grass2', 'sand4'),
        ('grass2', 'dirt2', 'grass2', 'dirt1'),
        ('grass2', 'dirt2', 'grass2', 'sand2'),
        ('grass2', 'dirt2', 'grass2', 'sand4'),
        ('grass2', 'grass2', 'dirt1', 'dirt2'),
        ('grass2', 'grass2', 'dirt2', 'dirt1'),
        ('grass2', 'grass2', 'dirt2', 'grass1'),
        ('grass2', 'grass2', 'dirt2', 'sand2'),
        ('grass2', 'grass2', 'dirt2', 'sand4'),
        ('grass2', 'grass2', 'grass1', 'dirt2'),
        ('grass2', 'grass2', 'sand2', 'dirt2'),
        ('grass2', 'grass2', 'sand4', 'dirt2'),
        ('grass2', 'grass3', 'grass2', 'soil5'),
        ('grass2', 'sand2', 'grass2', 'dirt1'),
        ('grass2', 'sand2', 'grass2', 'dirt2'),
        ('grass2', 'sand4', 'grass2', 'dirt1'),
        ('grass2', 'sand4', 'grass2', 'dirt2'),
        ('grass2', 'soil5', 'grass2', 'grass3'),
        ('sand1', 'sand1', 'sand2', 'sand4'),
        ('sand1', 'sand1', 'sand4', 'sand2'),
        ('sand2', 'dirt2', 'sand2', 'sand4'),
        ('sand2', 'sand1', 'sand2', 'sand4'),
        ('sand2', 'sand2', 'dirt2', 'grass1'),
        ('sand2', 'sand2', 'dirt2', 'grass2'),
        ('sand2', 'sand2', 'dirt2', 'sand4'),
        ('sand2', 'sand2', 'grass1', 'dirt2'),
        ('sand2', 'sand2', 'grass2', 'dirt2'),
        ('sand2', 'sand2', 'grass2', 'sand4'),
        ('sand2', 'sand2', 'sand1', 'sand4'),
        ('sand2', 'sand2', 'sand4', 'dirt2'),
        ('sand2', 'sand2', 'sand4', 'grass2'),
        ('sand2', 'sand2', 'sand4', 'sand1'),
        ('sand2', 'sand4', 'sand2', 'dirt2'),
        ('sand2', 'sand4', 'sand2', 'sand1'),
        ('sand4', 'grass2', 'sand4', 'sand2'),
        ('sand4', 'sand2', 'sand4', 'grass2'),
        ('sand4', 'sand4', 'dirt1', 'grass2'),
        ('sand4', 'sand4', 'dirt2', 'grass2'),
        ('sand4', 'sand4', 'dirt2', 'sand2'),
        ('sand4', 'sand4', 'dirt6', 'water5'),
        ('sand4', 'sand4', 'grass2', 'dirt1'),
        ('sand4', 'sand4', 'grass2', 'dirt2'),
        ('sand4', 'sand4', 'grass2', 'sand2'),
        ('sand4', 'sand4', 'sand2', 'dirt2'),
        ('sand4', 'sand4', 'sand2', 'grass2'),
        ('sand4', 'sand4', 'water5', 'dirt6'),
        ('soil3', 'dirt2', 'soil3', 'grass2'),
        ('soil3', 'grass2', 'soil3', 'dirt2'),
        ('soil5', 'cliff2', 'soil5', 'grass2'),
        ('soil5', 'grass2', 'soil5', 'cliff2'),
        ('soil5', 'soil5', 'cliff2', 'dirt1'),
        ('soil5', 'soil5', 'cliff2', 'dirt2'),
        ('soil5', 'soil5', 'dirt1', 'cliff2'),
        ('soil5', 'soil5', 'dirt2', 'cliff2'),
        ('water5', 'water5', 'cliff2', 'dirt2'),
        ('water5', 'water5', 'dirt2', 'cliff2'),
        ('water5', 'water5', 'dirt6', 'sand4'),
        ('water5', 'water5', 'sand4', 'dirt6'),
    ]

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
                if len(neighbors_counts) != 4:
                    continue
                xr1 = random.randint(0, 1)
                yr1 = random.randint(0, 1)
                xr2 = xr1
                yr2 = yr1
                while xr2 == xr1 and yr2 == yr1:
                    xr2 = random.randint(0, 1)
                    yr2 = random.randint(0, 1)
                land[x+xr1][y+yr1] = land[x+xr2][y+yr2]
                progress += 1
        print(f"Smoothing attempt #{attempts} with {progress} changes")

    for x in range(biome_image.width):
        for y in range(biome_image.height):
            tile = biome_tiles[land[x][y]]
            tiles_image.putpixel((x, y), avarage_colors[tile])
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
                    if pair not in pairs:
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

    for x in range(0, biome_image.width-1):
        for y in range(0, biome_image.height-1):
            neighbors_counts = {}
            neighbors_tiles = {}
            neighbors_coords = {}
            order = []
            for xd, yd in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                xn = x + xd
                yn = y + yd
                neighbor = biome_tiles[land[xn][yn]]
                if neighbor not in neighbors_counts:
                    neighbors_counts[neighbor] = 0
                neighbors_counts[neighbor] += 1
                neighbors_tiles[(xd, yd)] = neighbor
                if neighbor not in neighbors_coords:
                    neighbors_coords[neighbor] = []
                neighbors_coords[neighbor].append((xd, yd))
                order.append(neighbor)
            if len(neighbors_counts) != 3:
                continue
            neighbors_sorted = sorted(neighbors_counts.keys(), key=lambda n: neighbors_counts[n], reverse=True)
            t1 = neighbors_sorted[0]
            t23 = sorted([neighbors_sorted[1], neighbors_sorted[2]])
            t1coords = list(sorted(neighbors_coords[t1]))
            if t1coords[0][0] == t1coords[1][0] or t1coords[0][1] == t1coords[1][1]:
                diag = False
            else:
                diag = True
            if diag:
                if t1coords[0] == (0, 0):
                    rotate = 0
                else:
                    rotate = 90
                q = (t1, t23[0], t1, t23[1])
            else:
                if t1coords[0] == (0, 0):
                    if t1coords[1] == (1, 0):
                        rotate = 0
                        q = (t1, t1, t23[0], t23[1])
                    else:
                        rotate = 270
                        q = (t1, t1, t23[1], t23[0])
                elif t1coords[0] == (0, 1):
                    if t1coords[1] == (1, 1):
                        rotate = 90
                        q = (t1, t1, t23[0], t23[1])
                    else:
                        rotate = 180
                        q = (t1, t1, t23[1], t23[0])
                elif t1coords[0] == (1, 0):
                    if t1coords[1] == (1, 1):
                        rotate = 270
                        q = (t1, t1, t23[0], t23[1])
                    else:
                        rotate = 180
                        q = (t1, t1, t23[1], t23[0])
                elif t1coords[0] == (1, 1):
                    if t1coords[1] == (0, 1):
                        rotate = 180
                        q = (t1, t1, t23[0], t23[1])
                    else:
                        rotate = 90
                        q = (t1, t1, t23[1], t23[0])
            if q not in quadruples:
                raise Exception(f"Found neighboring tiles with 3 unexpected types {q} at ({x}, {y})")

    stats = {}
    for x in range(biome_image.width):
        for y in range(biome_image.height):
            tile = biome_tiles[land[x][y]]
            tiles_image.putpixel((x, y), avarage_colors[tile])
    different_biomes = list(stats.keys())
    different_biomes.sort(key=lambda i: stats[i], reverse=True)
    for i in range(len(different_biomes)):
        print(f"  {i+1:02d}. {different_biomes[i]}: {stats[different_biomes[i]]} tiles")


if __name__ == '__main__':
    main()
