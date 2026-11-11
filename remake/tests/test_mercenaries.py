"""
H3: the topics of the mercenaries (RecalcMercBriefings 591800) and the end of their briefings (591de0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import mercenaries

#------------------------------------------------------------------------------


def value(game_vars, index, topic):
    return game_vars[mercenaries.variable(index, topic)]


class RecalcTest(unittest.TestCase):

    def vars(self):
        return {'amerc_5': 1.0}

    def test_the_first_hire(self):
        game_vars = self.vars()
        mercenaries.recalc(game_vars, ['Hero'], 0, 'bz8k')
        self.assertEqual(value(game_vars, 5, 1), 1.0)
        self.assertEqual(value(game_vars, 5, 3), 0.0)
        self.assertEqual(value(game_vars, 5, 9), 0.0)

    def test_the_meeting_comes_first(self):
        game_vars = self.vars()
        game_vars[mercenaries.variable(5, 8)] = 1.0
        game_vars[mercenaries.variable(5, 1)] = 1.0
        game_vars[mercenaries.variable(5, 3)] = 1.0
        mercenaries.recalc(game_vars, ['Hero'], 0, 'bz8k')
        self.assertEqual(value(game_vars, 5, 8), 1.0)
        self.assertEqual(value(game_vars, 5, 3), 0.0)
        self.assertEqual(value(game_vars, 5, 1), 0.0)

    def test_not_in_the_story_yet(self):
        game_vars = {}
        game_vars[mercenaries.variable(5, 8)] = 1.0
        mercenaries.recalc(game_vars, ['Hero'], 0, 'bz8k')
        self.assertEqual(value(game_vars, 5, 8), 1.0)
        self.assertEqual(value(game_vars, 5, 11), 0.0)
        game_vars[mercenaries.variable(5, 8)] = 0.0
        mercenaries.recalc(game_vars, ['Hero'], 0, 'bz8k')
        self.assertEqual(value(game_vars, 5, 11), 1.0)

    def test_the_meeting_hides_the_hire(self):
        game_vars = self.vars()
        game_vars[mercenaries.variable(5, 8)] = 1.0
        game_vars[mercenaries.variable(5, 1)] = 1.0
        mercenaries.recalc(game_vars, ['Hero'], 0, 'bz8k')
        self.assertEqual(value(game_vars, 5, 8), 1.0)
        self.assertEqual(value(game_vars, 5, 1), 0.0)

    def test_the_mercenary_in_the_party(self):
        game_vars = self.vars()
        mercenaries.recalc(game_vars, ['Hero', 'merc5'], 1, 'bz8k')
        self.assertEqual(value(game_vars, 5, 2), 1.0)
        self.assertEqual(value(game_vars, 5, 1), 0.0)
        mercenaries.recalc(game_vars, ['Hero', 'merc5'], 1, 'bz2g')
        self.assertEqual(value(game_vars, 5, 2), 0.0)

    def test_the_third_mercenary_is_refused(self):
        game_vars = self.vars()
        mercenaries.recalc(game_vars, ['Hero', 'merc1', 'merc2'], 2, 'bz8k')
        self.assertEqual(value(game_vars, 5, 9), 1.0)
        self.assertEqual(value(game_vars, 5, 1), 0.0)

    def test_a_mercenary_taken_back(self):
        game_vars = self.vars()
        game_vars['amerc_5'] = 2.0
        mercenaries.recalc(game_vars, ['Hero'], 0, 'bz8k')
        self.assertEqual(value(game_vars, 5, 3), 1.0)
        self.assertEqual(value(game_vars, 5, 1), 0.0)

    def test_the_goodbye_clears_the_mercenary(self):
        game_vars = self.vars()
        game_vars[mercenaries.variable(5, 10)] = 1.0
        mercenaries.recalc(game_vars, ['Hero'], 0, 'bz8k')
        self.assertEqual(value(game_vars, 5, 10), 0.0)
        self.assertEqual(game_vars['amerc_5'], 0.0)
        self.assertEqual(value(game_vars, 5, 11), 0.0)


class CompleteTest(unittest.TestCase):

    def test_hire(self):
        game_vars = {}
        self.assertEqual(mercenaries.complete(game_vars, 'b.merc5.n1_5'), (5, 'hire'))
        self.assertEqual(game_vars['amerc_5'], 2.0)
        game_vars = {}
        self.assertEqual(mercenaries.complete(game_vars, 'b.merc5.n3_5'), (5, 'hire'))

    def test_leave(self):
        self.assertEqual(mercenaries.complete({}, 'b.merc5.n2_5'), (5, 'leave'))
        self.assertEqual(mercenaries.complete({}, 'b.merc5.n10_5'), (5, 'leave_zone'))

    def test_meet(self):
        game_vars = {mercenaries.variable(5, 8): 1.0}
        self.assertEqual(mercenaries.complete(game_vars, 'b.merc5.n8_5'), (5, 'met'))
        self.assertEqual(value(game_vars, 5, 8), 0.0)

    def test_another_variable(self):
        self.assertIsNone(mercenaries.complete({}, 'b.elder.s71_1'))
        self.assertIsNone(mercenaries.complete({}, 'b.merc5.n1_4'))

    def test_the_definition(self):
        self.assertEqual(mercenaries.mercenary(3), ('bz2g', 'Human Mercenary Thief'))
        self.assertEqual(mercenaries.name_of(7), 'merc7')
        self.assertIsNone(mercenaries.mercenary(11))


if __name__ == '__main__':
    unittest.main()
