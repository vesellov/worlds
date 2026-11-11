"""
H7: the hearing of engine.ai, the stealth and the awareness skills of a hero.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import ai
from engine import combat
from engine import experience
from engine import figure
from engine import stats
from engine import unit as unit_module

#------------------------------------------------------------------------------


class Watcher(object):

    def __init__(self, senses=(8.5, 0.0, 0.0, 2.8, 0.0, 0.0)):
        self.senses = senses
        self.detections = (1.0, ) * 6
        self.hearing = 1.0

    def distance_to(self, other):
        return other.distance

    def detection_factor(self):
        return 1.0

    def hearing_noise(self):
        return 0.0

    def hearing_factor(self):
        return self.hearing

    def visibility(self):
        return 1.0


class Walker(Watcher):

    def __init__(self, distance, noise=0.0, factor=1.0, seen=1.0):
        Watcher.__init__(self)
        self.distance = distance
        self.noise = noise
        self.factor = factor
        self.seen = seen

    def visibility(self):
        return self.seen

    def detection_factor(self):
        return self.factor

    def hearing_noise(self):
        return self.noise


class NoticesTest(unittest.TestCase):

    def test_sight(self):
        watcher = Watcher()
        self.assertTrue(ai.notices(watcher, Walker(5.0)))
        self.assertFalse(ai.notices(watcher, Walker(9.0)))

    def test_hearing(self):
        watcher = Watcher(senses=(0.0, 0.0, 0.0, 2.8, 0.0, 0.0))
        self.assertTrue(ai.notices(watcher, Walker(2.0, noise=0.8)))
        self.assertFalse(ai.notices(watcher, Walker(3.0, noise=0.8)))
        self.assertTrue(ai.notices(watcher, Walker(5.0, noise=2.0)))
        # 5536b0: a running watcher hears with 0.5
        watcher.hearing = unit_module.RUN_HEARING
        self.assertFalse(ai.notices(watcher, Walker(5.0, noise=2.0)))
        self.assertTrue(ai.notices(watcher, Walker(2.5, noise=2.0)))

    def test_a_crawler_is_seen_nearer(self):
        watcher = Watcher()
        self.assertTrue(ai.notices(watcher, Walker(5.0, seen=unit_module.VISIBILITY_SNEAK)))
        self.assertFalse(ai.notices(watcher, Walker(5.0, seen=unit_module.VISIBILITY_CRAWL)))
        self.assertTrue(ai.notices(watcher, Walker(4.0, seen=unit_module.VISIBILITY_CRAWL)))

    def test_a_quiet_walker(self):
        watcher = Watcher(senses=(0.0, 0.0, 0.0, 2.8, 0.0, 0.0))
        self.assertFalse(ai.notices(watcher, Walker(2.0, noise=0.35)))
        self.assertFalse(ai.notices(watcher, Walker(8.0, noise=1.0, factor=0.5)))
        self.assertTrue(ai.notices(watcher, Walker(1.2, noise=1.0, factor=0.5)))
        self.assertFalse(ai.notices(watcher, Walker(2.0, noise=1.0, factor=0.5)))


class Hill(object):
    """
    A flat land with a wall 5 high between x = 4 and x = 6.
    """

    def height_at(self, x, y):
        return 5.0 if 4.0 < x < 6.0 else 0.0


class Looker(Watcher):
    """
    A unit at a place which looks along +x.
    """

    def __init__(self, x=0.0, y=0.0, senses=(8.5, 0.0, 0.0, 0.0, 0.0, 0.0), view_cos=0.0, peripheral=0.0):
        Watcher.__init__(self, senses=senses)
        self.x = x
        self.y = y
        self.z = 0.0
        self.yaw = ai.look_yaw(1.0, 0.0)
        self.view_cos = view_cos
        self.peripheral = peripheral
        self.terrain = None
        self.is_dead = False

    def distance_to(self, other):
        return math.hypot(other.x - self.x, other.y - self.y)

    def height(self):
        return 1.5


class SeesTest(unittest.TestCase):

    def test_angle_of_the_sight(self):
        """
        5ca5a0 (partly): the other is seen inside the angle of the sight (UNIT_STATS 10 is the cosine of its half).
        """
        looker = Looker(view_cos=0.0)
        self.assertTrue(ai.sees(looker, Looker(5.0, 0.0)))
        self.assertTrue(ai.sees(looker, Looker(0.1, 5.0)))
        self.assertFalse(ai.sees(looker, Looker(-5.0, 0.0)))
        self.assertFalse(ai.sees(looker, Looker(-0.5, 0.0)))
        self.assertFalse(ai.sees(looker, Looker(9.0, 0.0)))
        narrow = Looker(view_cos=0.707)
        self.assertTrue(ai.sees(narrow, Looker(5.0, 4.0)))
        self.assertFalse(ai.sees(narrow, Looker(4.0, 5.0)))
        # a unit without the record sees all around
        around = Looker(view_cos=-1.0)
        self.assertTrue(ai.sees(around, Looker(-5.0, 0.0)))

    def test_peripheral_vision(self):
        """
        5ca5a0 (partly): ahead of the unit and nearer than UNIT_STATS 11, outside the angle and the reach of
        the sight too.
        """
        looker = Looker(view_cos=0.707, peripheral=5.0)
        self.assertTrue(ai.sees(looker, Looker(1.0, 4.0)))
        self.assertFalse(ai.sees(looker, Looker(1.0, 5.0)))
        self.assertFalse(ai.sees(looker, Looker(-1.0, 1.0)))
        blind = Looker(senses=(0.0, ) * 6, peripheral=1.7)
        self.assertTrue(ai.sees(blind, Looker(1.5, 0.0)))

    def test_life_sense(self):
        """
        5ca5a0 (partly): a living unit nearer than the life sense times its detection is seen anywhere around.
        """
        looker = Looker(senses=(8.5, 0.0, 3.0, 0.0, 0.0, 0.0))
        behind = Looker(-2.0, 0.0)
        self.assertTrue(ai.sees(looker, behind))
        behind.detections = (1.0, 1.0, 0.5, 1.0, 1.0, 1.0)
        self.assertFalse(ai.sees(looker, behind))
        behind.detections = (1.0, ) * 6
        behind.is_dead = True
        self.assertFalse(ai.sees(looker, behind))

    def test_line_of_sight(self):
        """
        5bc3e0 (partly): the ground between the eyes hides the other.
        """
        hill = Hill()
        self.assertEqual(ai.line_of_sight(hill, (0.0, 0.0, 1.5), (3.0, 0.0, 1.5)), 1.0)
        self.assertEqual(ai.line_of_sight(hill, (0.0, 0.0, 1.5), (8.0, 0.0, 1.5)), 0.0)
        self.assertEqual(ai.line_of_sight(hill, (0.0, 0.0, 9.0), (8.0, 0.0, 9.0)), 1.0)
        self.assertEqual(ai.line_of_sight(None, (0.0, 0.0, 1.5), (8.0, 0.0, 1.5)), 1.0)
        looker = Looker()
        looker.terrain = hill
        self.assertTrue(ai.sees(looker, Looker(3.0, 0.0)))
        self.assertFalse(ai.sees(looker, Looker(8.0, 0.0)))

    def test_notices_by_the_hearing(self):
        looker = Looker(senses=(8.5, 0.0, 0.0, 6.0, 0.0, 0.0))
        behind = Walker(5.0, noise=1.0)
        behind.x, behind.y = -5.0, 0.0
        self.assertFalse(ai.sees(looker, behind))
        self.assertTrue(ai.hears(looker, behind))
        self.assertTrue(ai.notices(looker, behind))

#------------------------------------------------------------------------------


class HeroSensesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def make_unit(self):
        unit = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(0.0, 0.0))
        human = self.game_data.units().unit_type('Human Male')
        unit_stats = stats.UnitStats(20, 20, 0, 0, 5, 0, 1, 2, 0,
                                     senses=(8.5, 0.0, 0.0, 2.8, 0.0, 0.0))
        unit.set_stats(unit_stats)
        unit.fighter = combat.Fighter(unit_stats, human)
        unit.hero_stats = experience.from_prototype(self.game_data.units(), 'Human Hero',
                                                    self.game_data.perks())
        return unit

    def test_stealth(self):
        unit = self.make_unit()
        self.assertAlmostEqual(unit.detection_factor(), 1.0)
        unit.hero_stats.skills[experience.SKILL_STEALTH] = 50
        self.assertAlmostEqual(unit.detection_factor(), 0.5)
        unit.hero_stats.skills[experience.SKILL_STEALTH] = 1000
        self.assertAlmostEqual(unit.detection_factor(), 0.1)

    def test_awareness(self):
        unit = self.make_unit()
        unit.hero_stats.skills[experience.SKILL_AWARENESS] = 50
        unit.refresh_senses()
        self.assertAlmostEqual(unit.senses[0], 8.5 * 1.5)
        self.assertAlmostEqual(unit.senses[ai.HEARING], 2.8 * 1.5)

    def test_the_noise_of_the_gaits(self):
        unit = self.make_unit()
        self.assertEqual(unit.hearing_noise(), 0.0)
        unit.move_to(5.0, 0.0, gait=stats.SPEED_CRAWL)
        self.assertAlmostEqual(unit.hearing_noise(), unit_module.NOISE_CRAWL)
        unit.move_to(5.0, 0.0, gait=stats.SPEED_SNEAK)
        self.assertAlmostEqual(unit.hearing_noise(), unit_module.NOISE_SNEAK)
        unit.move_to(5.0, 0.0, gait=stats.SPEED_WALK)
        self.assertAlmostEqual(unit.hearing_noise(), unit_module.NOISE_WALK)
        self.assertAlmostEqual(unit.hearing_factor(), 1.0)
        unit.move_to(5.0, 0.0, gait=stats.SPEED_RUN)
        self.assertAlmostEqual(unit.hearing_noise(), unit_module.NOISE_RUN)
        self.assertAlmostEqual(unit.hearing_factor(), unit_module.RUN_HEARING)

    def test_visibility_of_the_gaits(self):
        unit = self.make_unit()
        self.assertAlmostEqual(unit.visibility(), 1.0)
        unit.move_to(5.0, 0.0, gait=stats.SPEED_CRAWL)
        self.assertAlmostEqual(unit.visibility(), unit_module.VISIBILITY_CRAWL)
        unit.move_to(5.0, 0.0, gait=stats.SPEED_SNEAK)
        unit.stop()
        # the way of moving stays while the unit stands
        self.assertAlmostEqual(unit.visibility(), unit_module.VISIBILITY_SNEAK)


if __name__ == '__main__':
    unittest.main()
