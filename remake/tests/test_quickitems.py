"""
G6: the quick items of items.idb (potions, wands) and their use from the belt.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import quickitems

#------------------------------------------------------------------------------


class QuickItemsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()
        cls.database = cls.game_data.quick_items()

    def test_quick_items_of_the_database(self):
        potion = self.database.item('tiny potion 1')
        self.assertIsNotNone(potion)
        self.assertEqual(potion.spell_text, 'healing {e1}')
        self.assertTrue(potion.is_potion)
        self.assertIsNotNone(self.database.item('wand 1'))
        self.assertIsNone(self.database.item('long sword.iron'))

    def test_spell_from_text(self):
        spells = self.game_data.spells()
        spell = quickitems.spell_from_text(spells, 'healing {e2;e2;e2}')
        self.assertEqual(spell.name.lower(), 'healing')
        self.assertEqual(len(spell.modifiers), 3)
        spell = quickitems.spell_from_text(spells, 'strength {d2,d1}')
        self.assertEqual(spell.prototype.code, 'strength')
        self.assertEqual(len(spell.modifiers), 2)
        self.assertIsNone(quickitems.spell_from_text(spells, ''))

    def test_belt_and_take(self):
        names = ['tiny potion 1', 'long sword.iron', 'tiny potion 1', 'potion 1', 'wand 1']
        belt = quickitems.belt(names, self.database)
        self.assertEqual([(item.name, count) for item, count in belt],
                         [('tiny potion 1', 2), ('potion 1', 1), ('wand 1', 1)])
        item = belt[0][0]
        self.assertTrue(quickitems.take(names, item))
        self.assertEqual(names.count('tiny potion 1'), 1)
        self.assertFalse(quickitems.take(names, self.database.item('wand 5')))

    def test_the_order_of_the_belt(self):
        names = ['tiny potion 1', 'potion 1', 'wand 1']
        belt = quickitems.belt(names, self.database, order=['wand 1', 'tiny potion 1', 'wand 5'])
        self.assertEqual([item.name for item, count in belt], ['wand 1', 'tiny potion 1'])

    def test_use_of_a_potion(self):
        class Fighter(object):
            parts = []

            def __init__(self):
                self.enchantments = []

            def enchant(self, index, duration, effect):
                self.enchantments.append((index, duration, effect))

        class Unit(object):
            def __init__(self):
                self.fighter = Fighter()

        unit = Unit()
        spells = self.game_data.spells()
        used, healed = quickitems.use(unit, self.database.item('tiny potion 1'), spells)
        self.assertTrue(used)
        self.assertEqual(healed, 0.0)
        used, _ = quickitems.use(unit, self.database.item('tiny potion 2'), spells)
        self.assertTrue(used)
        self.assertTrue(unit.fighter.enchantments)
        used, _ = quickitems.use(unit, self.database.item('wand 1'), spells)
        self.assertFalse(used)


if __name__ == '__main__':
    unittest.main()
