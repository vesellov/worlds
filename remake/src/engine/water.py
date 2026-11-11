"""
The water of a map: the second layer of the sectors and its waves, as game.exe does it.

A sector with liquids has its own 33 x 33 vertices, tiles and a material per tile (0xffff - no liquid there).
The materials of the .mp (converted by 6c6870 (partly)): color r, g, b, alpha, self illumination, wave multiplier,
warp speed; the type "water" is textured, "water_notexture" is the plain color.

The waves (CTerrain 6c3e40, the wind from WORLD_SET of the .mob through 6c7e70):
- strength s of the wind is clamped to 0..1, amplitude A = 0.3 * s (0x7c8960), per material a = A * wave multiplier;
- step of the phase k = pi / 7 * (wind x, wind y) normalized * (asin(1 - s) + 0.50025 * pi);
- every update of the terrain (vf06 6cb1a0, taken here once per tick) the time phase grows by -pi / 21 * s;
- phase of a vertex = k . (x, y) + random offset of the vertex (6c57a0) + time phase * wave multiplier;
- with w = sin(phase): z += w * a * 0.25, x and y sway by sin(column or row * pi / 4 + g) * a * 3,
  where g grows by 0.05 per update, and a vertex with w > 0 gets the highlight w * s * multiplier * 50 / 255 * color.
The random offsets are 40 round spots of radius 1..5 on the 33 x 33 grid (height (1 - d^2 / r^2) * pi / 2),
the opposite edges are averaged so the grid repeats without seams.

The levels (SetWaterLevel( water, level, ticks ), 0xb8 of the scripts -> 593fa0 -> 6d43d0 on the list CWorld +0x70):
a record of 16 bytes {water, ticks left, offset, step} per water; the command sets step = (level - offset) / ticks
(ticks at least 1; a new record starts with the offset 0, so the level is counted from the level of the .mp),
the tick of the server (6d44a0 from 58dcd0) adds the step while ticks are left and drops a record which came back
to the offset 0; when anything moved, the units are checked again (+0x130 | 0x20). The water is the index of the
material of the liquid: zone8 keeps every lake in its own material 1..5 of the same color, and its scripts lift
the lake 4 by 2.3 and lower the lake 5 (the remake's reading, the code which reads the offsets is not found yet).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

import numpy as np

#------------------------------------------------------------------------------

from eilib import mapdata
from engine import terrain as terrain_module

#------------------------------------------------------------------------------

NO_LIQUID = mapdata.NO_LIQUID

GRID = mapdata.SECTOR_VERTICES

# game.exe static initializers 6c5320, 6c5340, 6c5360
AMPLITUDE = 0.3
PHASE_STEP = math.pi / 7.0
TIME_STEP = PHASE_STEP * -1.0 / 3.0

Z_SCALE = 0.25
SWAY_SCALE = 3.0
SWAY_STEP = math.pi / 4.0
SWAY_SPEED = 0.05
HIGHLIGHT = 50.0 / 255.0

SPOTS = 40

# position 3, normal 3, texture 2, tint 3, alpha 1, wave 4 (phase, multiplier, column, row),
# highlight color 3, glow color 3 (the self illumination of the material, 6c6870 (partly))
VERTEX_SIZE = 22

#------------------------------------------------------------------------------


class Wind(object):

    def __init__(self, direction=(0.0, 1.0, 0.0), strength=0.0):
        self.strength = min(max(float(strength), 0.0), 1.0)
        d = np.asarray(direction, dtype=np.float64)
        length = np.linalg.norm(d)
        d = d / length if length > 0 else np.array([0.0, 1.0, 0.0])
        factor = PHASE_STEP * (math.asin(1.0 - self.strength) + 0.50025016 * math.pi)
        self.step = (d[0] * factor, d[1] * factor)
        self.amplitude = AMPLITUDE * self.strength
        self.time_step = TIME_STEP * self.strength

    def __repr__(self):
        return 'Wind(s=%.2f step=(%.3f, %.3f))' % (self.strength, self.step[0], self.step[1])


class WaveClock(object):
    """
    The phases of the waves, they move once per update of the terrain.
    """

    def __init__(self, wind):
        self.wind = wind
        self.time_phase = 0.0
        self.sway_phase = 0.0

    def update(self, ticks=1):
        # the time phase is multiplied by the wave multipliers of the materials, so it is not wrapped
        self.time_phase += self.wind.time_step * ticks
        self.sway_phase = math.fmod(self.sway_phase + SWAY_SPEED * ticks, 2.0 * math.pi)

#------------------------------------------------------------------------------


class WaterLevels(object):
    """
    The list CWorld +0x70: water (the material of the liquid) -> [ticks left, offset, step].
    """

    def __init__(self):
        self.records = {}

    def set_level(self, water, level, ticks):
        """
        6d43d0.
        """
        ticks = max(int(ticks), 1)
        record = self.records.setdefault(int(water), [0, 0.0, 0.0])
        record[0] = ticks
        record[2] = (float(level) - record[1]) / ticks

    def tick(self):
        """
        6d44a0: True when a level moved.
        """
        moved = False
        for water, record in list(self.records.items()):
            if record[0] < 1:
                if record[1] == 0.0:
                    del self.records[water]
                continue
            moved = True
            record[0] -= 1
            record[1] += record[2]
        return moved

    def offset(self, water):
        record = self.records.get(water)
        return record[1] if record is not None else 0.0

    def offsets(self):
        return {water: record[1] for water, record in self.records.items()}


def random_phases(seed=0):
    """
    (33, 33) offsets of the phase of the vertices, game.exe 6c57a0 with its own random numbers.
    """
    rng = random.Random(seed)
    grid = np.zeros((GRID, GRID))
    iy, ix = np.mgrid[0:GRID, 0:GRID]
    for _ in range(SPOTS):
        cx, cy = rng.randrange(1 << 15) % GRID, rng.randrange(1 << 15) % GRID
        radius_sq = float((rng.randrange(1 << 15) % 5 + 1) ** 2)
        d_sq = (ix - cx) ** 2.0 + (iy - cy) ** 2.0
        inside = d_sq < radius_sq
        grid[inside] += (1.0 - d_sq[inside] / radius_sq) * math.pi * 0.5
    edge = (grid[1:GRID - 1, 0] + grid[1:GRID - 1, GRID - 1]) * 0.5
    grid[1:GRID - 1, 0] = grid[1:GRID - 1, GRID - 1] = edge
    edge = (grid[0, 1:GRID - 1] + grid[GRID - 1, 1:GRID - 1]) * 0.5
    grid[0, 1:GRID - 1] = grid[GRID - 1, 1:GRID - 1] = edge
    corner = (grid[0, 0] + grid[0, GRID - 1] + grid[GRID - 1, 0] + grid[GRID - 1, GRID - 1]) * 0.25
    grid[0, 0] = grid[0, GRID - 1] = grid[GRID - 1, 0] = grid[GRID - 1, GRID - 1] = corner
    return grid


def wave_offsets(x, y, column, row, phase0, multiplier, wind, clock):
    """
    (dx, dy, dz, w) of water vertices, the same formula the water shader uses.
    """
    a = wind.amplitude * multiplier
    w = np.sin(phase0 + clock.time_phase * multiplier)
    dx = np.sin(column * SWAY_STEP + clock.sway_phase) * a * SWAY_SCALE
    dy = np.sin(row * SWAY_STEP + clock.sway_phase) * a * SWAY_SCALE
    return dx, dy, w * a * Z_SCALE, w


def sector_meshes(terrain, sx, sy, wind, phases):
    """
    (texture, material) -> (vertices (n, VERTEX_SIZE) float32, indexes uint16) of the liquid of a sector,
    texture is the texture index or None for the materials without a texture; the material is the index of the liquid
    whose level WaterLevels moves.
    """
    sector = terrain.sectors.get((sx, sy))
    if sector is None or not sector.has_water:
        return {}
    materials = terrain.header.materials
    per_row = terrain.header.tiles_per_row
    origin = np.array([sx * mapdata.SECTOR_CELLS, sy * mapdata.SECTOR_CELLS, 0.0])
    groups = {}
    for ty in range(mapdata.SECTOR_TILES):
        for tx in range(mapdata.SECTOR_TILES):
            material_index = int(sector.water_materials[ty, tx])
            if material_index == NO_LIQUID or material_index >= len(materials):
                continue
            material = materials[material_index]
            textured = material['type'] == 'water'
            texture_index, tile_index, rotation = (int(v) for v in sector.water_tiles[ty, tx])
            key = (texture_index if textured else None, material_index)
            groups.setdefault(key, []).append((tx, ty, tile_index, rotation, material))
    triangles = np.array(terrain_module.TILE_TRIANGLES)
    result = {}
    for key, items in groups.items():
        vertices = np.zeros((len(items) * 9, VERTEX_SIZE), dtype=np.float32)
        indexes = np.empty(len(items) * len(triangles), dtype=np.uint16)
        for n, (tx, ty, tile_index, rotation, material) in enumerate(items):
            r, g, b, alpha = material['color']
            tint = (1.0, 1.0, 1.0) if key[0] is not None else (r, g, b)
            glow = material.get('glow') or tuple(min(1.0, c * material['self_illumination'])
                                                 for c in (r, g, b))
            for k, (gx, gy) in enumerate(terrain_module.TILE_GRID):
                vx, vy = tx * 2 + gx, ty * 2 + gy
                position = sector.water[vy, vx] + origin
                row = vertices[n * 9 + k]
                row[0:3] = position
                row[3:6] = sector.water_normals[vy, vx]
                if key[0] is not None:
                    row[6:8] = terrain_module.tile_uv(tile_index, rotation, gx / 2.0, gy / 2.0, per_row)
                row[8:11] = tint
                row[11] = alpha
                row[12] = wind.step[0] * position[0] + wind.step[1] * position[1] + phases[vy, vx]
                row[13] = material['wave_multiplier']
                row[14] = vx
                row[15] = vy
                row[16:19] = (r, g, b)
                row[19:22] = glow
            indexes[n * len(triangles):(n + 1) * len(triangles)] = triangles + n * 9
        result[key] = (vertices, indexes)
    return result
