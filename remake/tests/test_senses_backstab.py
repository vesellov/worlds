"""
H8: the night sight (566360) and the skill of the backstab (525d60) of a hero.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random
import unittest
from unittest import mock

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import combat
from engine import experience
from engine import figure
from engine import stats
from engine import unit as unit_module

#------------------------------------------------------------------------------

SIGHT = 8.5

#------------------------------------------------------------------------------


class NightSightTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def make_unit(self):
        unit = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(0.0, 0.0),
                                rng=random.Random(2))
        unit.speeds = (3.0, 1.5, 0.9, 0.4)
        unit.speed_scale = 1.0
        human = self.game_data.units().unit_type('Human Male')
        unit_stats = stats.UnitStats(20, 20, 0, 0, 5, 0, 1, 2, 0, senses=(SIGHT, 0, 0, 0, 0, 0))
        unit.set_stats(unit_stats)
        unit.fighter = combat.Fighter(unit_stats, human)
        return unit

    def test_darkness_of_the_hours(self):
        self.assertEqual(unit_module.darkness(12.0), 0.0)
        self.assertEqual(unit_module.darkness(20.0), 0.0)
        self.assertAlmostEqual(unit_module.darkness(22.0), 0.5)
        self.assertEqual(unit_module.darkness(3.0), 1.0)
        self.assertEqual(unit_module.darkness(8.0), 0.0)

    def test_night_takes_the_sight_without_infra(self):
        unit = self.make_unit()
        self.assertAlmostEqual(unit.senses[0], SIGHT)
        unit.set_light(0.5)
        self.assertAlmostEqual(unit.senses[0], SIGHT * 0.5)
        unit.set_light(0.5, wet=True)
        self.assertAlmostEqual(unit.senses[0], SIGHT * 0.5 * 0.9)
        unit.set_light(0.0)
        self.assertAlmostEqual(unit.senses[0], SIGHT)

    def test_infra_and_the_night_vision_perk(self):
        unit = self.make_unit()
        unit.base_senses = (SIGHT, 100.0, 0.0, 0.0, 0.0, 0.0)
        unit.set_light(1.0)
        # the infra 100 sees at night as by the day
        self.assertAlmostEqual(unit.senses[0], SIGHT)
        unit.hero_stats = experience.from_prototype(self.game_data.units(), 'Human Hero',
                                                    self.game_data.perks())
        unit.hero_stats.add_perk('night1')
        power = unit.hero_stats.perk_power(experience.GROUP_NIGHT_VISION)
        unit.hero_stats.skills[experience.SKILL_AWARENESS] = 0
        unit.refresh_senses()
        self.assertAlmostEqual(unit.senses[1], 100.0 * (1.0 + power / 100.0))
        self.assertAlmostEqual(unit.senses[0], SIGHT * (1.0 + power / 100.0))


class BackstabTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def make_unit(self, x, y):
        unit = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(x, y),
                                rng=random.Random(2))
        unit.speeds = (3.0, 1.5, 0.9, 0.4)
        unit.speed_scale = 1.0
        human = self.game_data.units().unit_type('Human Male')
        unit_stats = stats.UnitStats(20, 20, 0, 0, 5, 0, 1, 2, 0, senses=(SIGHT, 0, 0, 0, 0, 0))
        unit.set_stats(unit_stats)
        unit.fighter = combat.Fighter(unit_stats, human, in_party=True)
        return unit

    def test_skill_and_perk_go_into_the_blow(self):
        hero = self.make_unit(0.0, 2.0)
        foe = self.make_unit(0.0, 0.0)
        foe.fighter.in_party = False
        hero.hero_stats = experience.from_prototype(self.game_data.units(), 'Human Hero',
                                                    self.game_data.perks())
        hero.hero_stats.skills[experience.SKILL_BACKSTAB] = 10
        hero.hero_stats.add_perk('bs1')
        hero.foe = foe
        with mock.patch.object(unit_module.combat, 'make_hit') as make_hit:
            with mock.patch.object(unit_module.Unit, 'deliver_blow'):
                hero.resolve_blow()
        make_hit.assert_called_once()
        self.assertTrue(make_hit.call_args.kwargs['backstab'])
        self.assertAlmostEqual(make_hit.call_args.kwargs['skill'], 210.0)


if __name__ == '__main__':
    unittest.main()
