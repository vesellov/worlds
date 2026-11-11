"""
G11: the answers of a unit from acks.db (the groups, the phrases and their chance).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import acks

#------------------------------------------------------------------------------


class FixedRandom(object):
    def __init__(self, values):
        self.values = list(values)

    def randrange(self, count):
        value = self.values.pop(0) if self.values else 0
        return value % count


class AcksTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def player(self, prototype='Human Hero', values=(0, 0)):
        unit = type('Unit', (), {'prototype_name': prototype})()
        return acks.player_for(self.game_data, unit, FixedRandom(values))

    def test_groups_of_the_phrases(self):
        player = self.player()
        self.assertIsNotNone(player)
        phrase = player.choose(acks.GROUP_ATTACK)
        self.assertIsNotNone(phrase)
        self.assertIn('Attack', phrase.sound_name)
        self.assertIsNotNone(player.choose(acks.GROUP_MOVE))
        self.assertIsNone(player.choose(999))
        self.assertIsNone(self.player(prototype='No Such Prototype'))

    def test_chance_of_a_phrase(self):
        # the chance of HH_Selected001 is 70 percent
        self.assertIsNotNone(self.player(values=(0, 69)).choose(acks.GROUP_SELECTED))
        self.assertIsNone(self.player(values=(0, 70)).choose(acks.GROUP_SELECTED))
        # every phrase of NoWayAtt has 100 percent
        self.assertIsNotNone(self.player(values=(0, 99)).choose(acks.GROUP_NO_WAY_ATTACK))

    def test_sound_of_a_phrase(self):
        player = self.player()
        phrase = player.choose(acks.GROUP_MOVE)
        sound = acks.PhraseSound(phrase, (1.0, 2.0, 3.0))
        self.assertEqual(sound.name, phrase.sound_name)
        self.assertEqual(sound.position, (1.0, 2.0, 3.0))
        self.assertFalse(sound.block)


if __name__ == '__main__':
    unittest.main()
