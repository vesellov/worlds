"""
Tests of engine.woundskin: the levels of the wounds of the parts and their layers (536ea0, 5370f0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

import numpy as np

#------------------------------------------------------------------------------

from eilib import idb
from engine import woundskin

#------------------------------------------------------------------------------


class _Part(object):

    def __init__(self, health, max_health):
        self.health = health
        self.max_health = max_health


class _Shape(object):

    def __init__(self, shows_wounds):
        self.shows_wounds = shows_wounds


class _Item(object):

    def __init__(self, shows_wounds):
        self.shape = _Shape(shows_wounds)


class WoundSkinTest(unittest.TestCase):

    def test_part_level(self):
        self.assertEqual(woundskin.part_level(20.0, 20.0), 0)
        self.assertEqual(woundskin.part_level(15.0, 20.0), 0)
        self.assertEqual(woundskin.part_level(14.0, 20.0), 1)
        self.assertEqual(woundskin.part_level(6.6, 20.0), 2)
        self.assertEqual(woundskin.part_level(1.0, 20.0), 3)
        self.assertEqual(woundskin.part_level(-5.0, 20.0), 3)
        # a part of the type with no health
        self.assertEqual(woundskin.part_level(0.0, 0.0), 0)

    def test_other_units_share_the_limbs(self):
        parts = [_Part(10.0, 10.0), _Part(5.0, 10.0), _Part(10.0, 10.0), _Part(1.0, 10.0),
                 _Part(2.0, 10.0), _Part(10.0, 10.0)]
        levels = woundskin.levels(parts, False)
        self.assertEqual(levels, (0, 1, 3, 3, 2, 2))
        self.assertEqual(woundskin.layer_names('unmogo', levels, False),
                         ['unmogobdw1', 'unmogohw3', 'unmogohw3', 'unmogolw2', 'unmogolw2'])

    def test_armor_hides_the_wounds(self):
        parts = [_Part(1.0, 10.0)] * 6
        armors = {idb.ARMOR_PLATE: _Item(False), idb.ARMOR_HELM: _Item(True)}
        levels = woundskin.levels(parts, True, armors)
        self.assertEqual(levels, (3, 0, 0, 0, 3, 3))
        self.assertEqual(woundskin.layer_names('unhuma', levels, True), ['unhumahdw3', 'unhumallw3', 'unhumarlw3'])
        armors[idb.ARMOR_LEGGINS] = _Item(False)
        self.assertEqual(woundskin.levels(parts, True, armors), (3, 0, 0, 0, 0, 0))

    def test_wounds_over_the_body(self):
        base = np.full((256, 256, 4), 200, dtype=np.uint8)
        layer = np.zeros((64, 64, 4), dtype=np.uint8)
        layer[:32, :32] = (255, 0, 0, 255)
        result = woundskin.wounded(base, [layer], False)
        self.assertEqual(result.shape, (256, 256, 4))
        self.assertEqual(tuple(result[10, 10]), (255, 0, 0, 200))
        self.assertEqual(tuple(result[200, 200]), (200, 200, 200, 200))
        # a layer of a wrong size is skipped like 692fd0 does
        self.assertTrue((woundskin.wounded(base, [layer], True) == base).all())


if __name__ == '__main__':
    unittest.main()
