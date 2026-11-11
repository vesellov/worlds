"""
Reader of the maps: a .mpr archive with one .mp header and .sec sectors of 32 x 32 cells.

A sector has 33 x 33 land vertices: a small shift in x and y, the height and a packed normal,
and 16 x 16 tiles, each covers 2 x 2 cells. A tile is a square of a terrain texture
(texture number, tile number inside the texture and a rotation by 90 degrees).
Sectors with liquids have a second layer of vertices and tiles and a material per tile.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

import numpy as np

#------------------------------------------------------------------------------

MP_MAGIC = b'\x72\xF6\x4A\xCE'

SEC_MAGIC = b'\x74\xF7\x4B\xCF'

SECTOR_CELLS = 32

SECTOR_VERTICES = SECTOR_CELLS + 1

SECTOR_TILES = 16

# the material of a tile of the water layer where the sector has no liquid
NO_LIQUID = 0xffff

TERRAIN_TYPES = ('base', 'water_notexture', 'grass', 'water')

TILE_TYPES = ('grass', 'ground', 'stone', 'sand', 'rock', 'field', 'water', 'road', 'empty', 'snow',
              'ice', 'drygrass', 'snowballs', 'lava', 'swamp', 'highrock')

VERTEX_DTYPE = np.dtype([('dx', 'i1'), ('dy', 'i1'), ('z', '<u2'), ('normal', '<u4')])

#------------------------------------------------------------------------------


class MapHeader(object):

    def __init__(self):
        self.max_altitude = 0.0
        self.sectors_x = 0
        self.sectors_y = 0
        self.textures_count = 0
        self.texture_size = 0
        self.tiles_count = 0
        self.tile_size = 0
        self.materials = []
        self.tile_types = []
        self.animated_tiles = []

    def __repr__(self):
        return 'MapHeader(%dx%d sectors, %d textures, alt=%.2f)' % (
            self.sectors_x, self.sectors_y, self.textures_count, self.max_altitude)

    @property
    def tiles_per_row(self):
        return self.texture_size // self.tile_size


class Sector(object):

    def __init__(self):
        # (33, 33, 3) x, y, z inside the sector, rows go by y
        self.land = None
        self.land_normals = None
        # (16, 16, 3) texture, tile, rotation
        self.land_tiles = None
        self.water = None
        self.water_normals = None
        self.water_tiles = None
        # (16, 16) material number of the water tiles
        self.water_materials = None

    @property
    def has_water(self):
        return self.water is not None

#------------------------------------------------------------------------------


def parse_mp(data):
    data = bytes(data)
    if data[:4] != MP_MAGIC:
        raise ValueError('not an .mp file')
    h = MapHeader()
    (h.max_altitude, h.sectors_x, h.sectors_y, h.textures_count, h.texture_size,
     h.tiles_count, h.tile_size) = struct.unpack_from('<f6I', data, 4)
    materials_count = struct.unpack_from('<H', data, 32)[0]
    animated_count = struct.unpack_from('<I', data, 34)[0]
    offset = 38
    for _ in range(materials_count):
        values = struct.unpack_from('<I7f3I', data, offset)
        offset += 44
        h.materials.append(dict(
            type=TERRAIN_TYPES[values[0]] if values[0] < len(TERRAIN_TYPES) else values[0],
            color=values[1:5],
            self_illumination=values[5],
            wave_multiplier=values[6],
            warp_speed=values[7],
            glow=tuple(min(1.0, c * values[5]) for c in values[1:4]),
        ))
    h.tile_types = list(struct.unpack_from('<%dI' % h.tiles_count, data, offset))
    offset += 4 * h.tiles_count
    for _ in range(animated_count):
        h.animated_tiles.append(struct.unpack_from('<2H', data, offset))
        offset += 4
    return h


def _unpack_normals(packed):
    packed = packed.astype(np.int64)
    x = (((packed >> 11) & 0x7ff) - 1000.0) / 1000.0
    y = ((packed & 0x7ff) - 1000.0) / 1000.0
    z = (packed >> 22) / 1000.0
    return np.stack([x, y, z], axis=-1)


def _vertices(raw, max_altitude):
    grid = raw.reshape(SECTOR_VERTICES, SECTOR_VERTICES)
    iy, ix = np.mgrid[0:SECTOR_VERTICES, 0:SECTOR_VERTICES]
    xyz = np.stack([
        ix + grid['dx'] / 254.0,
        iy + grid['dy'] / 254.0,
        grid['z'] * (max_altitude / 65535.0),
    ], axis=-1)
    return xyz, _unpack_normals(grid['normal'])


def _tiles(raw):
    raw = raw.astype(np.int64).reshape(SECTOR_TILES, SECTOR_TILES)
    return np.stack([(raw >> 6) & 255, raw & 63, (raw >> 14) & 3], axis=-1)


def parse_sec(data, max_altitude):
    data = bytes(data)
    if data[:4] != SEC_MAGIC:
        raise ValueError('not a .sec file')
    s = Sector()
    liquids = data[4]
    offset = 5
    count = SECTOR_VERTICES * SECTOR_VERTICES
    land = np.frombuffer(data, dtype=VERTEX_DTYPE, count=count, offset=offset)
    offset += count * VERTEX_DTYPE.itemsize
    s.land, s.land_normals = _vertices(land, max_altitude)
    if liquids:
        water = np.frombuffer(data, dtype=VERTEX_DTYPE, count=count, offset=offset)
        offset += count * VERTEX_DTYPE.itemsize
        s.water, s.water_normals = _vertices(water, max_altitude)
    tiles_count = SECTOR_TILES * SECTOR_TILES
    s.land_tiles = _tiles(np.frombuffer(data, dtype='<u2', count=tiles_count, offset=offset))
    offset += tiles_count * 2
    if liquids:
        s.water_tiles = _tiles(np.frombuffer(data, dtype='<u2', count=tiles_count, offset=offset))
        offset += tiles_count * 2
        s.water_materials = np.frombuffer(data, dtype='<u2', count=tiles_count, offset=offset).reshape(
            SECTOR_TILES, SECTOR_TILES).astype(np.int64)
    return s


def sector_name(map_name, x, y):
    return '%s%03d%03d.sec' % (map_name, x, y)
