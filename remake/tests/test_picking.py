"""
Tests of client.picking: the ray of the screen against the boxes of the parts of the body and the
cursors and the called shots of the hit locations.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

import numpy

#------------------------------------------------------------------------------

from client import picking

#------------------------------------------------------------------------------


class PickingTest(unittest.TestCase):

    def test_ray_box(self):
        origin = numpy.array([0.0, 0.0, -5.0])
        direction = numpy.array([0.0, 0.0, 1.0])
        low, high = numpy.array([-1.0, -1.0, -1.0]), numpy.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(picking.ray_box(origin, direction, low, high), 4.0)
        self.assertIsNone(picking.ray_box(origin, numpy.array([1.0, 0.0, 0.0]), low, high))
        self.assertIsNone(picking.ray_box(numpy.array([0.0, 0.0, 5.0]), direction, low, high))
        # a ray which starts inside the box
        self.assertEqual(picking.ray_box(numpy.array([0.0, 0.0, 0.0]), direction, low, high), 0.0)

    def test_hit_locations_and_cursors(self):
        self.assertEqual(sorted(picking.PART_CURSORS), [0, 1, 2, 3, 4, 5])
        self.assertEqual(picking.PART_CURSORS[0], 'cursor_attack_hd')
        self.assertEqual(picking.PART_CURSORS[1], 'cursor_attack_bd')
        self.assertEqual(picking.PART_CURSORS[2], 'cursor_attack_lh')
        self.assertEqual(picking.PART_CURSORS[3], 'cursor_attack_rh')
        self.assertEqual(picking.PART_CURSORS[4], 'cursor_attack_ll')
        self.assertEqual(picking.PART_CURSORS[5], 'cursor_attack_rl')

    def test_called_shots(self):
        self.assertEqual(picking.CALLED_SHOT_NAMES['cs_head'], 0)
        self.assertEqual(picking.CALLED_SHOT_NAMES['cs_body'], 1)
        self.assertEqual(picking.CALLED_SHOT_NAMES['cs_lhand'], 2)
        self.assertEqual(picking.CALLED_SHOT_NAMES['cs_rhand'], 3)
        self.assertEqual(picking.CALLED_SHOT_NAMES['cs_lleg'], 4)
        self.assertEqual(picking.CALLED_SHOT_NAMES['cs_rleg'], 5)

    def test_cursor_without_a_called_shot(self):
        class FakeApp(object):
            aim_part = None

        self.assertEqual(picking.cursor_for(FakeApp(), None, 0), picking.WHOLE_UNIT_CURSOR)
        FakeApp.aim_part = 0
        self.assertEqual(picking.cursor_for(FakeApp(), None, 0), 'cursor_attack_hd')
        self.assertEqual(picking.cursor_for(FakeApp(), None, 9), picking.WHOLE_UNIT_CURSOR)

    def test_hover_falls_back_to_the_ground(self):
        unit = object()

        class Camera(object):
            def view(self):
                return numpy.identity(4)

        class Scene(object):
            x = 0.0
            y = 0.0
            width = 800
            height = 600
            camera = Camera()

            def projection(self):
                return numpy.identity(4)

            def pick_ground(self, x, y):
                return (1.0, 2.0, 3.0)

        class App(object):
            scene = Scene()
            terrain = object()
            aim_part = None
            map_units = []

            def unit_at(self, point):
                return unit

            def corpse_at(self, point):
                return None

        self.assertEqual(picking.hover_at(App(), (10, 20)), (unit, None))

        class Empty(App):
            def unit_at(self, point):
                return None

        self.assertEqual(picking.hover_at(Empty(), (10, 20)), (None, None))

    def test_names_of_the_parts(self):
        for location, names in picking.PART_NAMES.items():
            self.assertTrue(names)
            self.assertIn(location, picking.PART_CURSORS)


if __name__ == '__main__':
    unittest.main()
