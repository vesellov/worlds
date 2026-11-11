"""
G14: the size of the figure under Enlarge (37) and Shrink (38).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import spellvisual

#------------------------------------------------------------------------------


class SpellVisualTest(unittest.TestCase):

    class Fighter(object):
        def __init__(self):
            self.spells = {}

        def enchantment(self, index):
            return self.spells.get(index, 0.0)

    class Figure(object):
        def __init__(self):
            self.scale = 1.5

    class Unit(object):
        def __init__(self):
            self.fighter = SpellVisualTest.Fighter()
            self.figure = SpellVisualTest.Figure()

    def test_no_spells(self):
        unit = self.Unit()
        self.assertFalse(spellvisual.update(unit))
        self.assertEqual(unit.figure.scale, 1.5)

    def test_enlarge_and_shrink(self):
        unit = self.Unit()
        unit.fighter.spells[spellvisual.SPELL_ENLARGE] = 2.0
        self.assertTrue(spellvisual.update(unit))
        self.assertEqual(unit.figure.scale, 3.0)
        unit.fighter.spells[spellvisual.SPELL_SHRINK] = 2.0
        self.assertTrue(spellvisual.update(unit))
        self.assertEqual(unit.figure.scale, 1.5)
        del unit.fighter.spells[spellvisual.SPELL_ENLARGE]
        self.assertTrue(spellvisual.update(unit))
        self.assertEqual(unit.figure.scale, 0.75)
        del unit.fighter.spells[spellvisual.SPELL_SHRINK]
        self.assertTrue(spellvisual.update(unit))
        self.assertEqual(unit.figure.scale, 1.5)


if __name__ == '__main__':
    unittest.main()
