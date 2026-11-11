"""
H9: the mood signals of 5aea80 and the groups of the answers about the items.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import acks
from engine import signals

#------------------------------------------------------------------------------


class SignalTest(unittest.TestCase):

    def test_the_strongest_level(self):
        signal = signals.Signal()
        signal.raise_level(signals.TO_SUSPICION)
        signal.raise_level(signals.TO_AGGRESSION)
        signal.raise_level(signals.BACK_FROM_AGGRESSION)
        self.assertEqual(signal.update(1), signals.TO_AGGRESSION)
        self.assertEqual(signal.active, signals.TO_AGGRESSION)

    def test_the_twenty_ticks(self):
        signal = signals.Signal()
        signal.raise_level(signals.FIGHT)
        self.assertEqual(signal.update(1), signals.FIGHT)
        for _ in range(signals.SIGNAL_TICKS - 1):
            self.assertEqual(signal.update(1), 0)
        self.assertEqual(signal.active, signals.FIGHT)
        signal.update(1)
        self.assertEqual(signal.active, 0)

    def test_a_new_level_waiting(self):
        signal = signals.Signal()
        signal.raise_level(signals.TO_ALARM)
        self.assertEqual(signal.update(1), signals.TO_ALARM)
        signal.raise_level(signals.ALARM)
        self.assertEqual(signal.update(1), signals.ALARM)


class AckGroupsTest(unittest.TestCase):

    def test_the_item_messages(self):
        self.assertEqual(acks.item_message_group(0x23), acks.GROUP_ARMOR_CRIPPLED)
        self.assertEqual(acks.item_message_group(0x24), acks.GROUP_WEAPON_CRIPPLED)
        self.assertEqual(acks.item_message_group(0x25), acks.GROUP_ARMOR_CRIPPLED)
        self.assertEqual(acks.item_message_group(0x26), acks.GROUP_WEAPON_CRIPPLED)
        self.assertIsNone(acks.item_message_group(0x30))

    def test_the_signal_groups(self):
        self.assertEqual(acks.signal_group(signals.TO_AGGRESSION), acks.GROUP_BIG_ATTACK)
        self.assertEqual(acks.signal_group(signals.ALARM), acks.GROUP_ATTACK)
        self.assertIsNone(acks.signal_group(signals.BACK_FROM_AGGRESSION))


if __name__ == '__main__':
    unittest.main()
