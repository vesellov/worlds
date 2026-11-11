"""
H4: the global map of an allod: the places of the zones, their markers and the roads out of a road zone.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import globalmap

#------------------------------------------------------------------------------


class GlobalMapTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def zone(self, name):
        return next(z for z in self.game_data.zones() if z.name.lower() == name.lower())

    def make_map(self, name):
        return globalmap.GlobalMap.from_game(self.game_data, self.zone(name))

    def test_the_places_of_the_zones(self):
        model = self.make_map('gz1g_bz1g')
        self.assertEqual(model.positions['bz1g'], (-2.749, -0.186))
        self.assertEqual(model.positions['gz3g'], (-0.617, -0.214))
        self.assertNotEqual(model.positions['gz2g'], (0.0, 0.0))

    def test_the_roads_out_of_a_road(self):
        model = self.make_map('gz1g_gz2g')
        names = [option.zone.name for option in model.options]
        self.assertEqual(names, ['bz1g', 'gz2g'])
        self.assertEqual([option.pass_time for option in model.options], [1.0, 2.0])
        self.assertTrue(model.options[0].has_map)
        model = self.make_map('gz1g_gz3g')
        self.assertEqual([option.zone.name for option in model.options], ['bz1g', 'gz3g'])

    def test_a_road_to_another_road(self):
        model = self.make_map('gz1g_bz1g')
        names = [option.zone.name for option in model.options]
        self.assertIn('gz1g_gz2g', names)
        option = model.options[names.index('gz1g_gz2g')]
        self.assertFalse(option.has_map)

    def test_the_markers(self):
        model = self.make_map('gz1g_bz1g')
        marker = model.marker(self.zone('bz1g'))
        self.assertGreaterEqual(marker[0], globalmap.MARGIN)
        self.assertLessEqual(marker[0], 1.0 - globalmap.MARGIN)
        self.assertTrue(0.0 <= marker[0] <= 1.0)
        self.assertTrue(0.0 <= marker[1] <= 1.0)

    def test_the_projection_turns_y(self):
        u, v = globalmap.project(0.0, 1.0, (-1.0, -1.0, 1.0, 1.0))
        self.assertAlmostEqual(u, 0.5)
        self.assertAlmostEqual(v, globalmap.MARGIN)
        u, v = globalmap.project(0.0, -1.0, (-1.0, -1.0, 1.0, 1.0))
        self.assertAlmostEqual(v, 1.0 - globalmap.MARGIN)

    def test_the_zone_by_its_name(self):
        model = self.make_map('gz1g_gz2g')
        self.assertEqual(model.zone_of('GZ2G').name, 'gz2g')
        self.assertIsNone(model.zone_of('nowhere'))


if __name__ == '__main__':
    unittest.main()
