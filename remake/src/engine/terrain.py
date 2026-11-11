"""
Terrain of a map: meshes of the sectors grouped by the texture and the height of the land at any point.

Texture coordinates of a tile, found by matching the edges of the neighbor tiles of the real maps:
the tiles of a texture are counted from its last row of pixels and the rotation turns the tile
clockwise by 90 degrees per step. The rows of the texture are uploaded in the order of the file.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------

from eilib import mapdata

#------------------------------------------------------------------------------

# local coordinates of the 3 x 3 vertices of one tile and its 8 triangles (counter clockwise seen from above)
TILE_GRID = [(i % 3, i // 3) for i in range(9)]

# 6c8d00: the type of the tile out of the map
OUTSIDE_TILE_TYPE = 8

TILE_TRIANGLES = []
for _y in range(2):
    for _x in range(2):
        _a = _y * 3 + _x
        TILE_TRIANGLES += [_a, _a + 1, _a + 4, _a, _a + 4, _a + 3]

#------------------------------------------------------------------------------


class Terrain(object):

    def __init__(self, game_data, map_name):
        self.game_data = game_data
        self.map_name = map_name
        self.header = game_data.map_header(map_name)
        self.sectors = {}
        for sy in range(self.header.sectors_y):
            for sx in range(self.header.sectors_x):
                sector = game_data.map_sector(map_name, sx, sy)
                if sector is not None:
                    self.sectors[(sx, sy)] = sector

    @property
    def width(self):
        return self.header.sectors_x * mapdata.SECTOR_CELLS

    @property
    def height(self):
        return self.header.sectors_y * mapdata.SECTOR_CELLS

    def texture_name(self, texture_index):
        return '%s%03d' % (self.map_name, texture_index)

    def height_at(self, x, y):
        """
        Height of the land under the point, interpolated inside the triangles of the cell.
        """
        height = self.surface_at(x, y, water=False)
        return 0.0 if height is None else height

    def water_height_at(self, x, y):
        """
        Height of the water of the sector of the point, None where the sector has no water.
        """
        return self.surface_at(x, y, water=True)

    def tile_type_at(self, x, y):
        """
        6c8d00: the type of the tile under the point (tileDesc.reg), the one of the tile of the water where
        the water is above the land, 8 out of the map.
        """
        cx, cy = int(round(x)), int(round(y))
        if cx < 0 or cy < 0 or cx >= self.header.sectors_x * mapdata.SECTOR_CELLS or \
                cy >= self.header.sectors_y * mapdata.SECTOR_CELLS:
            return OUTSIDE_TILE_TYPE
        sector = self.sectors.get((cx // mapdata.SECTOR_CELLS, cy // mapdata.SECTOR_CELLS))
        if sector is None:
            return OUTSIDE_TILE_TYPE
        tx, ty = (cx % mapdata.SECTOR_CELLS) // 2, (cy % mapdata.SECTOR_CELLS) // 2
        tiles = sector.land_tiles
        if sector.water is not None and sector.water_materials is not None and sector.water_materials[ty, tx] != mapdata.NO_LIQUID:
            water = self.water_height_at(x, y)
            if water is not None and self.height_at(x, y) < water:
                tiles = sector.water_tiles
        index = int(tiles[ty, tx, 0]) * 64 + int(tiles[ty, tx, 1])
        types = self.header.tile_types
        if index >= len(types):
            return OUTSIDE_TILE_TYPE
        return types[index] & 0x1f

    def surface_at(self, x, y, water=False):
        sx = min(max(int(x // mapdata.SECTOR_CELLS), 0), self.header.sectors_x - 1)
        sy = min(max(int(y // mapdata.SECTOR_CELLS), 0), self.header.sectors_y - 1)
        sector = self.sectors.get((sx, sy))
        if sector is None:
            return None
        grid = sector.water if water else sector.land
        if grid is None:
            return None
        lx = min(max(x - sx * mapdata.SECTOR_CELLS, 0.0), mapdata.SECTOR_CELLS - 1e-6)
        ly = min(max(y - sy * mapdata.SECTOR_CELLS, 0.0), mapdata.SECTOR_CELLS - 1e-6)
        ix, iy = int(lx), int(ly)
        fx, fy = lx - ix, ly - iy
        z = grid[:, :, 2]
        z00, z10, z01, z11 = z[iy, ix], z[iy, ix + 1], z[iy + 1, ix], z[iy + 1, ix + 1]
        # the same split of the cell as TILE_TRIANGLES: by the diagonal from (0, 0) to (1, 1)
        if fx >= fy:
            return float(z00 + (z10 - z00) * fx + (z11 - z10) * fy)
        return float(z00 + (z11 - z01) * fx + (z01 - z00) * fy)

    def sector_meshes(self, sx, sy):
        """
        texture index -> (vertices (n, 8) float32: x, y, z, nx, ny, nz, u, v, indexes uint16) of the land of one sector,
        the liquids are built by engine.water.
        """
        sector = self.sectors.get((sx, sy))
        if sector is None:
            return {}
        positions = sector.land
        normals = sector.land_normals
        tiles = sector.land_tiles
        per_row = self.header.tiles_per_row
        origin = np.array([sx * mapdata.SECTOR_CELLS, sy * mapdata.SECTOR_CELLS, 0.0])
        groups = {}
        for ty in range(mapdata.SECTOR_TILES):
            for tx in range(mapdata.SECTOR_TILES):
                texture_index, tile_index, rotation = (int(v) for v in tiles[ty, tx])
                groups.setdefault(texture_index, []).append((tx, ty, tile_index, rotation))
        result = {}
        for texture_index, items in groups.items():
            vertices = np.empty((len(items) * 9, 8), dtype=np.float32)
            indexes = np.empty(len(items) * len(TILE_TRIANGLES), dtype=np.uint16)
            for n, (tx, ty, tile_index, rotation) in enumerate(items):
                for k, (gx, gy) in enumerate(TILE_GRID):
                    vx, vy = tx * 2 + gx, ty * 2 + gy
                    u, v = tile_uv(tile_index, rotation, gx / 2.0, gy / 2.0, per_row)
                    vertices[n * 9 + k, 0:3] = positions[vy, vx] + origin
                    vertices[n * 9 + k, 3:6] = normals[vy, vx]
                    vertices[n * 9 + k, 6:8] = (u, v)
                indexes[n * len(TILE_TRIANGLES):(n + 1) * len(TILE_TRIANGLES)] = np.array(TILE_TRIANGLES) + n * 9
            result[texture_index] = (vertices, indexes)
        return result

#------------------------------------------------------------------------------


def tile_uv(tile_index, rotation, lx, ly, per_row):
    """
    Texture coordinates of the point (lx, ly) of a tile, both from 0 to 1, x to the east and y to the north.
    """
    col = tile_index % per_row
    row = per_row - 1 - tile_index // per_row
    # column and row fractions inside the tile as it is seen on the map, then back through the rotations
    a, b = lx, 1.0 - ly
    for _ in range(rotation):
        a, b = b, 1.0 - a
    return (col + a) / per_row, (row + b) / per_row
