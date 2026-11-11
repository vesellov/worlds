"""
H2: the other members of the party in the zone: the slots behind the hero, the formation of a group
order and the following of the hero.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import unittest

#------------------------------------------------------------------------------

from engine import companions
from engine import stats
from engine import unit as unit_module

#------------------------------------------------------------------------------


class FakeUnit(object):

    def __init__(self, x, y, mode=unit_module.MODE_IDLE):
        self.x, self.y = x, y
        self.mode = mode
        self.foe = None
        self.is_dead = False
        self.is_frozen = False
        self.orders = []

    def go_to(self, x, y, run=False, gait=None):
        self.orders.append(((x, y), run, gait))
        self.mode = unit_module.MODE_WALK


class FakeHero(object):

    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw
        self.is_dead = False


class FollowTest(unittest.TestCase):

    def test_slots_behind_the_hero(self):
        hero = FakeHero(0.0, 0.0, 0.0)
        x, y = companions.follow_slot(hero, 1)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 1.6)
        x, y = companions.follow_slot(hero, 2)
        self.assertAlmostEqual(x, -1.0)
        self.assertAlmostEqual(y, 1.6)
        x, y = companions.follow_slot(hero, 3)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 3.2)

    def test_the_slots_turn_with_the_hero(self):
        hero = FakeHero(0.0, 0.0, math.pi / 2.0)
        x, y = companions.follow_slot(hero, 1)
        self.assertAlmostEqual(x, -1.6)
        self.assertAlmostEqual(y, 1.0)

    def test_formation_points(self):
        points = companions.formation_points(2.0, 3.0, 1)
        self.assertEqual(points, [(2.0, 3.0)])
        points = companions.formation_points(2.0, 3.0, 3)
        self.assertEqual(points[0], (2.0, 3.0))
        self.assertEqual(len(points), 3)
        for x, y in points[1:]:
            self.assertAlmostEqual(math.hypot(x - 2.0, y - 3.0), companions.FORMATION_RADIUS)
        self.assertEqual(companions.formation_points(2.0, 3.0, 0), [])

    def test_a_follower_goes_to_its_slot(self):
        hero = FakeHero(0.0, 0.0, 0.0)
        follower = FakeUnit(10.0, 0.0)
        self.assertTrue(companions.update_follow(hero, [follower]))
        self.assertEqual(len(follower.orders), 1)
        (x, y), run, gait = follower.orders[0]
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 1.6)
        self.assertFalse(run)
        self.assertEqual(gait, stats.SPEED_RUN)

    def test_a_follower_near_its_slot_stands(self):
        hero = FakeHero(0.0, 0.0, 0.0)
        follower = FakeUnit(1.0, 1.6)
        companions.update_follow(hero, [follower])
        self.assertEqual(follower.orders, [])

    def test_apart_and_busy_followers_stand(self):
        hero = FakeHero(0.0, 0.0, 0.0)
        follower = FakeUnit(10.0, 0.0)
        self.assertFalse(companions.update_follow(hero, [follower], together=False))
        self.assertEqual(follower.orders, [])
        follower.foe = object()
        self.assertTrue(companions.update_follow(hero, [follower]))
        self.assertEqual(follower.orders, [])
        follower.foe = None
        follower.mode = unit_module.MODE_ATTACK
        self.assertFalse(companions.can_follow(follower))
        follower.mode = unit_module.MODE_IDLE
        follower.is_dead = True
        self.assertFalse(companions.can_follow(follower))


if __name__ == '__main__':
    unittest.main()
