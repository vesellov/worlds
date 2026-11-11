"""
Tests of engine.flyinghp: the flying numbers of the lost health and the experience (5e1870, 5e18e0, 52ef90).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import combat
from engine import flyinghp

#------------------------------------------------------------------------------


class FlyingHpTest(unittest.TestCase):

    def test_lost_health(self):
        self.assertEqual(flyinghp.lost_health(1.0, 0.75, 40.0), 10)
        # truncated like crt_ftol
        self.assertEqual(flyinghp.lost_health(1.0, 0.99, 40.0), 0)
        self.assertIsNone(flyinghp.lost_health(0.5, 0.5, 40.0))
        self.assertIsNone(flyinghp.lost_health(0.5, 0.7, 40.0))

    def test_shown_experience(self):
        self.assertEqual(flyinghp.shown_experience(100.0, 25), 100)
        self.assertEqual(flyinghp.shown_experience(100.0, 26), 104)

    def test_hit_flags(self):
        hit = combat.Hit((1.0, ), 5.0, location=0, backstab=1.5)
        self.assertEqual(flyinghp.hit_flags(hit), flyinghp.BACKSTAB | flyinghp.CRITICAL)
        hit = combat.Hit((1.0, ), 5.0, location=1)
        self.assertEqual(flyinghp.hit_flags(hit), 0)

    def test_life_and_delay(self):
        numbers = flyinghp.FlyingNumbers()
        numbers.add('unit', 12, 0, 1000.0)
        numbers.add('unit', 30, flyinghp.EXP, 1000.0)
        numbers.add('unit', -1, 0, 1000.0)
        numbers.add(None, 5, 0, 1000.0)
        self.assertEqual(len(numbers.entries), 2)
        shown = numbers.shown(1200.0)
        # the experience waits half a second
        self.assertEqual([entry.value for entry, t in shown], [12])
        self.assertAlmostEqual(shown[0][1], 200.0 / 3000.0)
        self.assertEqual(len(numbers.shown(1600.0)), 2)
        numbers.update(4100.0)
        self.assertEqual([entry.value for entry in numbers.entries], [30])
        numbers.update(4600.0)
        self.assertEqual(numbers.entries, [])

    def test_quads(self):
        width, height, quads = flyinghp.quads(25, 0)
        self.assertEqual((width, height), (16, 12))
        # "2" is the third glyph of the first row, "5" the second of the second row
        self.assertEqual(quads[0], (0.0, 0.0, 8, 12, 0.25, 1.0, 0.375, 0.84375))
        self.assertEqual(quads[1], (8.0, 0.0, 8, 12, 0.125, 0.84375, 0.25, 0.6875))
        width, height, quads = flyinghp.quads(7, flyinghp.BACKSTAB | flyinghp.CRITICAL)
        self.assertEqual((width, height), (64, 32))
        self.assertEqual(quads[0][0], 28.0)
        self.assertEqual(quads[1], (0.0, 12.0, 64, 10, 0.0, 0.5390625, 1.0, 0.390625))
        self.assertEqual(quads[2], (0.0, 22.0, 64, 10, 0.0, 0.390625, 1.0, 0.2421875))

    def test_place_and_color(self):
        self.assertEqual(flyinghp.place(100.0, 200.0, 0.0, 16, 12), (92.0, 194.0))
        self.assertEqual(flyinghp.place(100.0, 200.0, 0.5, 16, 12), (152.0, 144.0))
        entry = flyinghp.Entry('unit', 3, flyinghp.EXP, 0.0)
        self.assertEqual(flyinghp.color(entry, True), flyinghp.COLOR_EXP)
        entry.flags = 0
        self.assertEqual(flyinghp.color(entry, True), flyinghp.COLOR_PLAYER)
        self.assertEqual(flyinghp.color(entry, False), flyinghp.COLOR_OTHER)


if __name__ == '__main__':
    unittest.main()
