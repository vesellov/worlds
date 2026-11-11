"""
H6: the four gaits of the movement dial, the crawl cycle and the speeds of engine.stats.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import figure
from engine import stats
from engine import unit as unit_module

#------------------------------------------------------------------------------


class GaitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()
        cls.model = figure.FigureModel(cls.game_data, 'unhuma')

    def make_unit(self):
        unit = unit_module.Unit(self.model, position=(0.0, 0.0))
        unit.speeds = (3.0, 1.5, 0.9, 0.4)
        unit.speed_scale = 1.0
        return unit

    def test_crawl_cycle(self):
        unit = self.make_unit()
        self.assertIsNotNone(unit.find_crawl_action())
        self.assertTrue(unit.find_crawl_action().name.lower().startswith('ccrawl'))
        unit.move_to(5.0, 0.0, gait=stats.SPEED_CRAWL)
        self.assertEqual(unit.mode, unit_module.MODE_CRAWL)
        self.assertEqual(unit.speed, 0.4)
        clips = [unit.player.clip] + list(unit.player.queue)
        self.assertTrue(any(clip.action.name.lower().startswith('ccrawl') for clip in clips))

    def test_sneak_and_walk(self):
        unit = self.make_unit()
        unit.move_to(5.0, 0.0, gait=stats.SPEED_SNEAK)
        self.assertEqual(unit.mode, unit_module.MODE_SNEAK)
        self.assertEqual(unit.speed, 0.9)
        unit.move_to(6.0, 0.0, gait=stats.SPEED_WALK)
        self.assertEqual(unit.mode, unit_module.MODE_WALK)
        self.assertEqual(unit.speed, 1.5)
        unit.move_to(7.0, 0.0, run=True)
        self.assertEqual(unit.mode, unit_module.MODE_RUN)
        self.assertEqual(unit.speed, 3.0)


if __name__ == '__main__':
    unittest.main()
