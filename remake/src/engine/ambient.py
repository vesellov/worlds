"""
The sounds of the land around the camera (CTerrain::vf06 6cb1a0 -> 6cb350 on every tick of the world).

The 3 x 3 sectors around the sector of the camera each get a material: the tiles of the sector (land and
water, 6c1bc0 (partly) counts them by the type of tileDesc.reg) are weighed by MATERIALS and the heaviest type wins.
A sector adds ((48^2 - min(d^2, 48^2)) / 48^2 * 100)^2 to its material, d is the distance from the camera to
the middle of the sector in the plane. The three materials with the biggest sums (above 0.001) sound:
- the loop "ambient\\<set>\\<material>\\circle.wav" at the volume round(sqrt(sum)), at most 100; a material
  which stays keeps its loop (6b90c0 sets the volume and the pan), the others start (6b8210) or stop (6b9460);
- with the chance of the material in percents per tick, a random sector of the material plays
  "ambient\\<set>\\<material>\\<day time>\\<n>.wav" (CSoundSystem::PlayAt with the distances 32 and 48)
  at its middle, shifted by up to 16 in x and y; n is random among the files of the "day" folder,
  counted when the map loads (6c6870, partly), so the other times of the day may miss a file and stay silent.

The set is the allod of the zone (Ingos, Suslanger, else Gipat; Dungeon for the caves, 6c6870 (partly)). The day time
is 6cbbc0: 6..18 day, 2..6 morning, 18..22 evening, else night. game.exe also pans the loops by the angle
between the camera and the sounding sectors; Kivy has no pan, the remake keeps only the volume.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

from eilib import mapdata

#------------------------------------------------------------------------------

# 7969c0: the types of the tiles of tileDesc.reg: (folder, weight, chance of a sound per tick in percents);
# the pan limit (30 or 100) of the table is not used without pan
MATERIALS = (
    ('Grass', 1.0, 6),
    ('Ground', 1.0, 6),
    ('Stone', 1.0, 6),
    ('sand', 1.0, 6),
    ('rock', 1.5, 4),
    ('field', 1.0, 6),
    ('water', 3.0, 6),
    ('road', 3.0, 6),
    ('', 0.0, 0),
    ('snow', 1.0, 6),
    ('Ice', 3.0, 6),
    ('DryGrass', 1.0, 6),
    ('SnowBall', 1.0, 6),
    ('lava', 3.0, 0),
    ('swamp', 3.0, 7),
    ('rock', 1.5, 4),
)

# 6c1bc0 (partly): a type out of the table is counted as the last one
LAST_TYPE = len(MATERIALS) - 1

# 7441d0 * 74420c: the radius of the hearing of a sector
RADIUS = 48.0
# 73ccf4: a material with a smaller sum is silent
MIN_SUM = 0.001
MAX_VOLUME = 100
# the three loudest materials sound
SOUNDING = 3

# CSoundSystem::PlayAt of a single sound: the distances, and the shift of its place from the middle of the sector
SHOT_NEAR = 32.0
SHOT_FAR = 48.0
SHOT_SPREAD = 16.0

# 6c6870 (partly): the folders of the allods
SET_DUNGEON = 'Dungeon'
SET_DEFAULT = 'Gipat'
SETS = ('Ingos', 'Suslanger')

CIRCLE_NAME = 'ambient\\%s\\%s\\circle.wav'
SHOT_NAME = 'ambient\\%s\\%s\\%s\\%d.wav'
DAY_NAME = 'ambient\\%s\\%s\\day\\%d.wav'

#------------------------------------------------------------------------------


def sound_set(allod, cave):
    """
    6c6870 (partly): the folder of the ambient sounds of a zone.
    """
    if cave:
        return SET_DUNGEON
    for name in SETS:
        if (allod or '').lower() == name.lower():
            return name
    return SET_DEFAULT


def day_time(hour):
    """
    6cbbc0: the folder of the hour.
    """
    hour = math.fmod(hour, 24.0)
    if 6.0 <= hour <= 18.0:
        return 'day'
    if 2.0 < hour < 22.0:
        return 'morning' if hour < 6.0 else 'evening'
    return 'night'


def tile_counts(sector, tile_types):
    """
    6c1bc0 (partly): how many tiles of the land and of the water (where the water has a material) the sector has
    by the type of tileDesc.reg.
    """
    counts = [0] * len(MATERIALS)
    layers = [(sector.land_tiles, None)]
    if sector.water_tiles is not None and sector.water_materials is not None:
        layers.append((sector.water_tiles, sector.water_materials))
    for tiles, materials in layers:
        for ty in range(mapdata.SECTOR_TILES):
            for tx in range(mapdata.SECTOR_TILES):
                if materials is not None and int(materials[ty, tx]) == mapdata.NO_LIQUID:
                    continue
                index = int(tiles[ty, tx, 0]) * 64 + int(tiles[ty, tx, 1])
                kind = tile_types[index] if index < len(tile_types) else LAST_TYPE
                counts[kind if kind < len(MATERIALS) else LAST_TYPE] += 1
    return counts


def material_of(counts):
    """
    6cb350: the type whose tiles weigh most, the first of the equal ones.
    """
    best, best_weight = 0, -1.0
    for index, count in enumerate(counts):
        weight = count * MATERIALS[index][1]
        if best_weight < weight:
            best, best_weight = index, weight
    return best


def sector_volume(dx, dy):
    """
    The part of a sector at (dx, dy) from the camera, 0..100.
    """
    r2 = RADIUS * RADIUS
    return (r2 - min(dx * dx + dy * dy, r2)) / r2 * 100.0


class Ambient(object):
    """
    The ambient sounds of one map. sound_exists(name) tells whether sfx.res has the file.
    """

    def __init__(self, terrain, set_name, sound_exists):
        self.terrain = terrain
        self.set_name = set_name
        self.counts = {}
        # 6c6870 (partly): the number of the files day\1.wav, day\2.wav... of every material
        self.variants = []
        for folder, _weight, _chance in MATERIALS:
            count = 0
            while folder and sound_exists(DAY_NAME % (set_name, folder, count + 1)):
                count += 1
            self.variants.append(count)

    def __repr__(self):
        return 'Ambient(%s)' % self.set_name

    def sector_material(self, sx, sy):
        if (sx, sy) not in self.counts:
            sector = self.terrain.sectors.get((sx, sy))
            if sector is None:
                self.counts[(sx, sy)] = None
            else:
                self.counts[(sx, sy)] = material_of(tile_counts(sector, self.terrain.header.tile_types))
        return self.counts[(sx, sy)]

    def cells(self, x, y):
        """
        The 3 x 3 sectors around the point: (material, volume, middle x, middle y) of the ones on the map.
        """
        n = mapdata.SECTOR_CELLS
        sx0, sy0 = int(round(x)) >> 5, int(round(y)) >> 5
        result = []
        for sy in range(sy0 - 1, sy0 + 2):
            for sx in range(sx0 - 1, sx0 + 2):
                if not (0 <= sx < self.terrain.header.sectors_x and 0 <= sy < self.terrain.header.sectors_y):
                    continue
                material = self.sector_material(sx, sy)
                if material is None:
                    continue
                cx, cy = (sx + 0.5) * n, (sy + 0.5) * n
                result.append((material, sector_volume(cx - x, cy - y), cx, cy))
        return result

    def sounding(self, cells):
        """
        [(material, volume 0..100)] of the loudest materials.
        """
        sums = {}
        for material, volume, _cx, _cy in cells:
            sums[material] = sums.get(material, 0.0) + volume * volume
        loud = sorted(sums.items(), key=lambda item: -item[1])[:SOUNDING]
        return [(material, min(int(round(math.sqrt(total))), MAX_VOLUME)) for material, total in loud
                if total > MIN_SUM]

    def tick(self, x, y, hour, rnd):
        """
        One tick at the camera (x, y): ({loop name: volume 0..1}, [(sound name, (x, y, z))] of the single
        sounds which start now). rnd is a random.Random.
        """
        cells = self.cells(x, y)
        loops = {}
        shots = []
        for material, volume in self.sounding(cells):
            folder, _weight, chance = MATERIALS[material]
            loops[CIRCLE_NAME % (self.set_name, folder)] = volume / float(MAX_VOLUME)
            if rnd.randint(1, 100) > chance or not self.variants[material]:
                continue
            _material, _volume, cx, cy = rnd.choice([c for c in cells if c[0] == material])
            name = SHOT_NAME % (self.set_name, folder, day_time(hour), rnd.randrange(self.variants[material]) + 1)
            px = cx + rnd.uniform(-SHOT_SPREAD, SHOT_SPREAD)
            py = cy + rnd.uniform(-SHOT_SPREAD, SHOT_SPREAD)
            ground = self.terrain.height_at(cx, cy)
            water = self.terrain.water_height_at(cx, cy)
            shots.append((name, (px, py, ground if water is None else max(ground, water))))
        return loops, shots
