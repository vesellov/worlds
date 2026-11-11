"""
H8: the fields 12..14 of the hero template and the encumbrance of a hero.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import combat
from engine import experience
from engine import figure
from engine import stats
from engine import unit as unit_module

#------------------------------------------------------------------------------


class FakeItem(object):

    def __init__(self, weight):
        self.weight = weight


class TemplateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def test_the_fields_of_the_template(self):
        template = self.game_data.units().template('Human Hero')
        self.assertAlmostEqual(template.max_weight, 200.0)
        self.assertEqual(template.weight_percent, 100)
        self.assertEqual(template.template_flag, 1)

    def test_the_stats_take_them(self):
        hero_stats = experience.from_prototype(self.game_data.units(), 'Human Hero',
                                               self.game_data.perks())
        self.assertAlmostEqual(hero_stats.max_weight, 200.0)
        self.assertEqual(hero_stats.weight_percent, 100)
        copy = hero_stats.copy()
        copy.max_weight = 10.0
        self.assertAlmostEqual(hero_stats.max_weight, 200.0)
        self.assertAlmostEqual(copy.max_weight, 10.0)


class EncumbranceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def make_unit(self):
        unit = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(0.0, 0.0))
        human = self.game_data.units().unit_type('Human Male')
        unit_stats = stats.UnitStats(20, 20, 0, 0, 5, 0, 1, 2, 0,
                                     senses=(8.5, 0.0, 0.0, 2.8, 0.0, 0.0),
                                     speeds=(3.0, 1.5, 0.9, 0.4))
        unit.set_stats(unit_stats)
        unit.fighter = combat.Fighter(unit_stats, human)
        unit.hero_stats = experience.from_prototype(self.game_data.units(), 'Human Hero',
                                                    self.game_data.perks())
        return unit

    def test_the_weight_of_the_things(self):
        unit = self.make_unit()
        unit.fighter.weapon_item = FakeItem(5.0)
        unit.fighter.armor_items = {0: FakeItem(7.0)}
        self.assertAlmostEqual(unit.carried_weight(), 12.0)

    def test_overload_forbids_the_run(self):
        """
        5241c0: the things weigh more than strength * 12 * (1 + lift * 0.01): the hero walks at its speed.
        """
        unit = self.make_unit()
        reasons = []
        unit.on_cant_run = lambda u, reason: reasons.append(reason)
        limit = unit.carry_limit()
        self.assertAlmostEqual(limit, unit.hero_stats.strength * 12.0)
        unit.fighter.weapon_item = FakeItem(limit + 1.0)
        unit.fighter.armor_items = {}
        self.assertEqual(unit.run_limit(), unit_module.RUN_OVERLOAD)
        unit.move_to(5.0, 0.0, gait=stats.SPEED_RUN)
        self.assertEqual(unit.mode, unit_module.MODE_WALK)
        self.assertAlmostEqual(unit.speed, 1.5)
        # told once
        unit.move_to(6.0, 0.0, gait=stats.SPEED_RUN)
        self.assertEqual(reasons, [unit_module.RUN_OVERLOAD])
        unit.fighter.weapon_item = FakeItem(4.0)
        unit.move_to(7.0, 0.0, gait=stats.SPEED_RUN)
        self.assertEqual(unit.mode, unit_module.MODE_RUN)
        self.assertAlmostEqual(unit.speed, 3.0)
        self.assertEqual(unit.run_refusal, unit_module.RUN_FREE)

if __name__ == '__main__':
    unittest.main()
