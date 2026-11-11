"""
G9: the prints of the steps of the units (engine.footprints).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random
import unittest

#------------------------------------------------------------------------------

from engine import footprints

#------------------------------------------------------------------------------


class Unit(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.yaw = 0.0
        self.fighter = object()
        self.is_dead = False


class FootprintsTest(unittest.TestCase):

    def test_prints_of_the_steps(self):
        field = footprints.FootprintField(lambda x, y: 2.0, random.Random(1))
        unit = Unit(0.0, 0.0)
        field.update([unit])
        self.assertFalse(field.prints)
        unit.y = -1.0
        field.update([unit])
        self.assertTrue(field.prints)
        print_ = field.prints[0]
        self.assertEqual(print_.z, 2.0)
        self.assertAlmostEqual(print_.x, -footprints.SIDE_OFFSET)
        self.assertEqual(print_.alpha, 1.0)
        unit.y = -1.1
        field.update([unit])
        self.assertEqual(len(field.prints), 2)
        self.assertAlmostEqual(field.prints[1].x, footprints.SIDE_OFFSET)

    def test_prints_fade(self):
        field = footprints.FootprintField(lambda x, y: 0.0, random.Random(2), life=10)
        unit = Unit(0.0, 0.0)
        field.update([unit])
        unit.y = -1.0
        for _ in range(8):
            field.update([unit])
            unit.y -= 0.0
        print_ = field.prints[0]
        print_.age = print_.life - footprints.FOOTPRINT_FADE // 2
        self.assertAlmostEqual(print_.alpha, 0.5)
        print_.age = print_.life
        self.assertEqual(print_.alpha, 0.0)

    def test_atlas_cell(self):
        self.assertEqual(footprints.cell_uv(0), (0.0, 0.875, 0.125, 1.0))
        self.assertEqual(footprints.cell_uv(9), (0.125, 0.75, 0.25, 0.875))

    def test_teleport_does_not_print(self):
        field = footprints.FootprintField(lambda x, y: 0.0, random.Random(3))
        unit = Unit(0.0, 0.0)
        field.update([unit])
        unit.x = 100.0
        field.update([unit])
        self.assertFalse(field.prints)


if __name__ == '__main__':
    unittest.main()
