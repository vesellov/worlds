"""
G7: the arrow in the hand of a bowman between the frames of the shot (538c20).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import idb
from engine import equipment

#------------------------------------------------------------------------------


class FakeAction(object):
    def __init__(self, probability, frames):
        self.probability = probability
        self.show_hide_frames = frames


class FakeClip(object):
    def __init__(self, action):
        self.action = action


class FakePlayer(object):
    def __init__(self, action, time):
        self.clip = FakeClip(action)
        self.time = time


class FakeShape(object):
    type = idb.WEAPON_BOW


class FakeWeapon(object):
    shape = FakeShape()


class FakeEquipment(object):
    weapon = FakeWeapon()


class FakeFigure(object):
    def __init__(self):
        self.hidden_parts = {equipment.ARROW_IN_HAND}
        self.equipment = FakeEquipment()


class FakeUnit(object):
    def __init__(self, action, time):
        self.figure = FakeFigure()
        self.player = FakePlayer(action, time)


class ArrowInHandTest(unittest.TestCase):

    def test_frames_of_the_shot(self):
        action = FakeAction(1, (2, 5))
        self.assertFalse(equipment.arrow_in_hand_visible(FakeUnit(action, 1.0)))
        self.assertTrue(equipment.arrow_in_hand_visible(FakeUnit(action, 2.0)))
        self.assertTrue(equipment.arrow_in_hand_visible(FakeUnit(action, 4.9)))
        self.assertFalse(equipment.arrow_in_hand_visible(FakeUnit(action, 5.0)))

    def test_action_without_the_bit(self):
        action = FakeAction(0, (2, 5))
        self.assertFalse(equipment.arrow_in_hand_visible(FakeUnit(action, 3.0)))

    def test_other_weapon(self):
        class SwordShape(object):
            type = idb.WEAPON_SWORD

        class SwordWeapon(object):
            shape = SwordShape()

        unit = FakeUnit(FakeAction(1, (2, 5)), 3.0)
        unit.figure.equipment.weapon = SwordWeapon()
        self.assertFalse(equipment.arrow_in_hand_visible(unit))


if __name__ == '__main__':
    unittest.main()
