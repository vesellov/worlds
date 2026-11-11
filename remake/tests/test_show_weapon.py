"""
The weapon in the hands is hidden during an action without the weapon bits (ShowWeapon 537cb0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import idb
from engine import equipment

#------------------------------------------------------------------------------


class _Action(object):

    def __init__(self, packed):
        self.packed = packed


class _Clip(object):

    def __init__(self, action):
        self.action = action


class _Player(object):

    def __init__(self, action):
        self.clip = _Clip(action)


class _Shape(object):
    type = idb.WEAPON_SWORD
    number = 3


class _Weapon(object):
    shape = _Shape()


class _Equipment(object):

    def __init__(self, weapon):
        self.weapon = weapon


class _Figure(object):

    def __init__(self, weapon):
        self.equipment = _Equipment(weapon)


class _Unit(object):

    def __init__(self, packed, weapon=_Weapon()):
        self.figure = _Figure(weapon)
        self.player = _Player(_Action(packed) if packed is not None else None)


class ShowWeaponTest(unittest.TestCase):

    def test_action_with_the_weapon(self):
        self.assertEqual(equipment.weapon_hidden_parts(_Unit(0x1)), frozenset())

    def test_action_without_the_weapon(self):
        # the state REST and no weapon bits, like "crest"
        self.assertEqual(equipment.weapon_hidden_parts(_Unit(1 << 15)), frozenset(['rh3.sword03']))

    def test_no_weapon_or_no_action(self):
        self.assertEqual(equipment.weapon_hidden_parts(_Unit(0, weapon=None)), frozenset())
        self.assertEqual(equipment.weapon_hidden_parts(_Unit(None)), frozenset())


if __name__ == '__main__':
    unittest.main()
