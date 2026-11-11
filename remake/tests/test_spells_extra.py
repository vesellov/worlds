"""
H11: the spells of the fog of visions, the charm and the comprehension.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import ai
from engine import combat
from engine import figure
from engine import magic
from engine import stats
from engine import unit as unit_module

#------------------------------------------------------------------------------


class ExtraSpellsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def make_unit(self):
        unit = unit_module.Unit(figure.FigureModel(self.game_data, 'unhuma'), position=(0.0, 0.0))
        human = self.game_data.units().unit_type('Human Male')
        unit_stats = stats.UnitStats(20, 20, 0, 0, 5, 0, 1, 2, 0,
                                     senses=(8.5, 0.0, 0.0, 2.8, 0.0, 0.0))
        unit.set_stats(unit_stats)
        unit.fighter = combat.Fighter(unit_stats, human, in_party=True)
        return unit

    def test_the_new_enchantments(self):
        spells = self.game_data.spells()
        for name in ('VisionFog', 'Comprehend', 'Charm'):
            spell = magic.make_spell(spells, name)
            self.assertIsNotNone(spell)
            self.assertTrue(spell.enchants())

    def test_the_fog_of_visions(self):
        unit = self.make_unit()
        fog = magic.make_spell(self.game_data.spells(), 'VisionFog')
        magic.apply(fog, unit)
        unit.refresh_senses()
        self.assertAlmostEqual(unit.senses[0], 8.5 * magic.VISION_FOG_FACTOR)

    def test_the_charm(self):
        unit = self.make_unit()
        charm = magic.make_spell(self.game_data.spells(), 'Charm')
        unit.foe = object()
        unit.target = (5.0, 0.0)
        unit.take_spell(charm, None)
        self.assertTrue(magic.is_charmed(unit.fighter))
        self.assertIsNone(unit.foe)
        self.assertIsNone(unit.target)
        hero = self.make_unit()
        controller = ai.Controller()
        controller.add_player_unit(hero)
        controller.add(unit, 1)
        self.assertFalse(controller.hostile(hero, unit))
        self.assertEqual(controller.enemies(controller.brains[0]), [])


if __name__ == '__main__':
    unittest.main()
