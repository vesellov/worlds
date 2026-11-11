"""
Tests of engine.ambient: the sounds of the land around the camera (6cb350).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random
import unittest

import numpy as np

#------------------------------------------------------------------------------

from eilib import mapdata
from engine import ambient

#------------------------------------------------------------------------------

GRASS = 0
WATER = 6
SWAMP = 14

#------------------------------------------------------------------------------


class _Header(object):

    def __init__(self, sectors_x, sectors_y, tile_types):
        self.sectors_x = sectors_x
        self.sectors_y = sectors_y
        self.tile_types = tile_types


class _Sector(object):

    def __init__(self, tile_index, water_index=None):
        n = mapdata.SECTOR_TILES
        self.land_tiles = np.zeros((n, n, 3), dtype=np.int64)
        self.land_tiles[:, :, 1] = tile_index
        self.water_tiles = None
        self.water_materials = None
        if water_index is not None:
            self.water_tiles = np.zeros((n, n, 3), dtype=np.int64)
            self.water_tiles[:, :, 1] = water_index
            self.water_materials = np.full((n, n), mapdata.NO_LIQUID, dtype=np.int64)
            # a quarter of the sector is under the water
            self.water_materials[:8, :8] = 0


class _Terrain(object):
    """
    3 x 3 sectors of grass (tile 0), the middle one is a swamp (tile 2).
    """

    def __init__(self):
        self.header = _Header(3, 3, [GRASS, WATER, SWAMP])
        self.sectors = {(x, y): _Sector(0) for x in range(3) for y in range(3)}
        self.sectors[(1, 1)] = _Sector(2)

    def height_at(self, x, y):
        return 1.0

    def water_height_at(self, x, y):
        return None


class AmbientTest(unittest.TestCase):

    def test_sound_set_by_the_allod(self):
        self.assertEqual(ambient.sound_set('ingos', False), 'Ingos')
        self.assertEqual(ambient.sound_set('Suslanger', False), 'Suslanger')
        self.assertEqual(ambient.sound_set('gipat', False), 'Gipat')
        self.assertEqual(ambient.sound_set(None, False), 'Gipat')
        self.assertEqual(ambient.sound_set('ingos', True), 'Dungeon')

    def test_day_time(self):
        # 6cbbc0: 6..18 day, 2..6 morning, 18..22 evening, the rest night
        self.assertEqual(ambient.day_time(6.0), 'day')
        self.assertEqual(ambient.day_time(18.0), 'day')
        self.assertEqual(ambient.day_time(3.0), 'morning')
        self.assertEqual(ambient.day_time(20.0), 'evening')
        self.assertEqual(ambient.day_time(23.0), 'night')
        self.assertEqual(ambient.day_time(1.0), 'night')
        self.assertEqual(ambient.day_time(34.0), 'day')

    def test_tiles_of_land_and_water(self):
        counts = ambient.tile_counts(_Sector(0, water_index=1), [GRASS, WATER])
        self.assertEqual(counts[GRASS], 256)
        self.assertEqual(counts[WATER], 64)
        # water weighs 3: 64 * 3 < 256
        self.assertEqual(ambient.material_of(counts), GRASS)
        counts[WATER] = 100
        self.assertEqual(ambient.material_of(counts), WATER)

    def test_volume_of_a_sector(self):
        self.assertEqual(ambient.sector_volume(0.0, 0.0), 100.0)
        self.assertEqual(ambient.sector_volume(48.0, 0.0), 0.0)
        self.assertEqual(ambient.sector_volume(100.0, 100.0), 0.0)
        self.assertAlmostEqual(ambient.sector_volume(24.0, 0.0), 75.0)

    def test_loops_and_single_sounds(self):
        sound = ambient.Ambient(_Terrain(), 'Gipat', lambda name: name.endswith(('\\1.wav', '\\2.wav')))
        self.assertEqual(sound.variants[GRASS], 2)
        self.assertEqual(sound.variants[8], 0)
        # the camera in the middle of the swamp sector: the swamp is loud, the grass around it too
        cells = sound.cells(48.0, 48.0)
        self.assertEqual(len(cells), 9)
        sounding = dict(sound.sounding(cells))
        self.assertEqual(sounding[SWAMP], 100)
        self.assertEqual(set(sounding), {SWAMP, GRASS})
        loops, shots = sound.tick(48.0, 48.0, 12.0, random.Random(5))
        self.assertEqual(loops['ambient\\Gipat\\swamp\\circle.wav'], 1.0)
        self.assertIn('ambient\\Gipat\\Grass\\circle.wav', loops)
        rnd = random.Random(7)
        names = set()
        for _ in range(400):
            names.update(name for name, position in sound.tick(48.0, 48.0, 23.0, rnd)[1])
        self.assertTrue(names)
        self.assertTrue(all('\\night\\' in name for name in names))
        self.assertTrue(any(name.startswith('ambient\\Gipat\\swamp\\night\\') for name in names))

    def test_out_of_the_map(self):
        sound = ambient.Ambient(_Terrain(), 'Gipat', lambda name: False)
        self.assertEqual(sound.tick(-500.0, -500.0, 12.0, random.Random(1)), ({}, []))


if __name__ == '__main__':
    unittest.main()
