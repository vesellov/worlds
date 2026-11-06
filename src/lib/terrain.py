"""
Land tiles of a biome, shared by server.land and the client.

terrain.bin keeps the type of the land (grass1, water5, cliff1 ...) at every corner of the segments of the map.
The tile of a segment is taken from the catalog of the land tiles, catalog/land.json, by the types at its 4 corners,
see build_tiles(). When the catalog has a few variants of a tile, the variant and the turn of the tile are chosen
by the number of the segment, see cell_number(), so the land generator and the client get the same tiles
and only the types are kept in the file of the biome. The types of the land change rarely along the map,
gzip of the .tar.gz archive compresses them into a few kilobytes.

All numbers are little-endian:

    b'WTRB', version: u8, width: u16, height: u16
    types count: u8, every type: length u8 + ascii bytes
    index of the type for every corner, u8, row by row: y from 0 to height - 1, x from 0 to width - 1 in every row
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

MAGIC = b'WTRB'

VERSION = 1

# type of the land of the segments at the last column and row of the map, which have no corners after them
EDGE_TYPE = 'water5'

# makes the number of the turn of a tile different from the number of its variant
ROTATION_SALT = 1

#------------------------------------------------------------------------------


def encode(types, width, height):
    """
    Returns terrain.bin data, types is {(x, y): type of the land} of all corners of the map.
    """
    names = sorted(set(types[(x, y)] for y in range(height) for x in range(width)))
    if len(names) > 255:
        raise ValueError(f'too many types of the land for terrain.bin: {len(names)}')
    index = {name: i for i, name in enumerate(names)}
    parts = [MAGIC, struct.pack('<BHHB', VERSION, width, height, len(names))]
    for name in names:
        data = name.encode('ascii')
        parts.append(struct.pack('<B', len(data)))
        parts.append(data)
    parts.append(bytes(index[types[(x, y)]] for y in range(height) for x in range(width)))
    return b''.join(parts)


def decode(data):
    """
    Returns (types, width, height) from terrain.bin data, see encode().
    """
    if data[:4] != MAGIC:
        raise ValueError('not a terrain.bin file')
    version, width, height, names_count = struct.unpack_from('<BHHB', data, 4)
    if version != VERSION:
        raise ValueError(f'terrain.bin version {version} is not supported, expected {VERSION}')
    offset = 10
    names = []
    for _ in range(names_count):
        length = data[offset]
        names.append(data[offset + 1:offset + 1 + length].decode('ascii'))
        offset += 1 + length
    types = {}
    for y in range(height):
        row = data[offset + y * width:offset + (y + 1) * width]
        for x in range(width):
            types[(x, y)] = names[row[x]]
    return types, width, height


def cell_number(x, y, salt=0):
    """
    Number which looks random but is always the same for the segment (x, y), the same on every device.
    """
    n = (x * 73856093 ^ y * 19349663 ^ salt * 83492791) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return n ^ (n >> 16)


def choose(variants, x, y):
    """
    One of the variants of a tile for the segment (x, y).
    """
    return variants[cell_number(x, y) % len(variants)]


def build_tiles(types, width, height, catalog):
    """
    Tiles of all segments of the map: {(x, y): (catalog id, turn in degrees)}, types is {(x, y): type of the land}
    of the corners, catalog is catalog/land.json. The segments of the last column and row are EDGE_TYPE tiles.
    Raises an exception when the catalog has no tile for the types around a segment.
    """
    tiles = build_puzzle(types, width, height, catalog)
    edge_id = catalog[EDGE_TYPE][0]
    for x in range(width):
        for y in range(height):
            if (x, y) not in tiles:
                tiles[(x, y)] = (edge_id, 0)
    return tiles


def build_puzzle(types, width, height, catalog):
    tiles = {}
    for xb in range(0, width - 1):
        for yb in range(0, height - 1):
            square = {}
            for xi, yi in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                xn = xb + xi
                yn = yb + yi
                square[(xi, yi)] = types[(xn, yn)]
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
                tiles[(x, y)] = (choose(catalog_ids, x, y), 90 * (cell_number(x, y, ROTATION_SALT) % 4))
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
                    tiles[(x, y)] = (choose(side_ids, x, y), 0)
                elif shape == 'right_left':
                    tiles[(x, y)] = (choose(side_ids, x, y), 270)
                elif shape == 'bottom_top':
                    tiles[(x, y)] = (choose(side_ids, x, y), 180)
                elif shape == 'left_right':
                    tiles[(x, y)] = (choose(side_ids, x, y), 90)
                elif shape == 'bottomleft_topright':
                    tiles[(x, y)] = (choose(corner_ids, x, y), 270)
                elif shape == 'topleft_bottomright':
                    tiles[(x, y)] = (choose(corner_ids, x, y), 0)
                elif shape == 'topright_bottomleft':
                    tiles[(x, y)] = (choose(corner_ids, x, y), 90)
                elif shape == 'bottomright_topleft':
                    tiles[(x, y)] = (choose(corner_ids, x, y), 180)
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
                            tiles[(x, y)] = (choose(catalog1213_ids, x, y), 0)
                        else:
                            shape = 2
                            tiles[(x, y)] = (choose(catalog1312_ids, x, y), 0)
                    else:
                        if len(col1) == 1:
                            if coords[t2].count((0, 1)):
                                shape = 3
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 0)
                            else:
                                shape = 4
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 0)
                        else:
                            if coords[t2].count((1, 0)):
                                shape = 5
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 90)
                            else:
                                shape = 6
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 90)
                elif coords[t1].count((1, 0)):
                    if len(diag2) == 1:
                        if coords[t2].count((0, 0)):
                            shape = 7
                            tiles[(x, y)] = (choose(catalog1312_ids, x, y), 270)
                        else:
                            shape = 8
                            tiles[(x, y)] = (choose(catalog1312_ids, x, y), 270)
                    else:
                        if len(col1) == 1:
                            if coords[t2].count((0, 1)):
                                shape = 9
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 180)
                            else:
                                shape = 10
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 180)
                        else:
                            if coords[t2].count((0, 0)):
                                shape = 11
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 270)
                            else:
                                shape = 12
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 270)
                elif coords[t1].count((0, 1)):
                    if len(diag2) == 1:
                        if coords[t2].count((0, 0)):
                            shape = 13
                            tiles[(x, y)] = (choose(catalog1312_ids, x, y), 90)
                        else:
                            shape = 14
                            tiles[(x, y)] = (choose(catalog1213_ids, x, y), 90)
                    else:
                        if len(col2) == 1:
                            if coords[t2].count((0, 0)):
                                shape = 15
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 180)
                            else:
                                shape = 16
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 180)
                        else:
                            if coords[t2].count((1, 0)):
                                shape = 17
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 0)
                            else:
                                shape = 18
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 0)
                elif coords[t1].count((1, 1)):
                    if len(diag1) == 1:
                        if coords[t2].count((1, 0)):
                            shape = 19
                            tiles[(x, y)] = (choose(catalog1213_ids, x, y), 0)
                        else:
                            shape = 20
                            tiles[(x, y)] = (choose(catalog1312_ids, x, y), 0)
                    else:
                        if len(col2) == 1:
                            if coords[t2].count((0, 0)):
                                shape = 21
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 0)
                            else:
                                shape = 22
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 0)
                        else:
                            if coords[t2].count((0, 0)):
                                shape = 23
                                tiles[(x, y)] = (choose(catalog1123_ids, x, y), 270)
                            else:
                                shape = 24
                                tiles[(x, y)] = (choose(catalog1132_ids, x, y), 270)
            else:
                raise Exception(f"Did not found a shape for {samples_all} at ({x}, {y})")
    return tiles
