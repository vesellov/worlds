"""
Tests of engine.camp: the quarters of the camp, the grid of the trade, the choice of the member and the
lines of the information window.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import idb
from engine import camp

#------------------------------------------------------------------------------


class FakeItem(object):

    def __init__(self, name, slot=0, weapon=False):
        self.name = name
        self.slot = slot
        self.is_weapon = weapon


class FakeWeapon(FakeItem):

    def __init__(self, name):
        super(FakeWeapon, self).__init__(name, weapon=True)


class FakeEquipment(object):

    def __init__(self, weapon_name=None, armors=()):
        self.armors = {item.slot: item for item in armors}
        self.weapon = FakeWeapon(weapon_name) if weapon_name is not None else None


class FakeFigure(object):

    def __init__(self, weapon_name=None, armors=()):
        self.equipment = FakeEquipment(weapon_name, armors)
        self.model_name = 'unorma'


class FakeFighter(object):

    def __init__(self):
        self.attack = 12
        self.parry = 7
        self.damage_min = 3
        self.damage_spread = 5


class FakeUnit(object):

    def __init__(self, weapon_name=None, armors=()):
        self.figure = FakeFigure(weapon_name, armors)
        self.fighter = FakeFighter()


class FakeParty(object):

    def __init__(self, money=0):
        self.money = money


class CampTest(unittest.TestCase):

    def test_quarters_and_cells(self):
        self.assertEqual(len(camp.MEMBER_RECTS), 4)
        self.assertEqual(camp.MEMBER_RECTS[0], (200, 100, 400, 300))
        self.assertEqual(camp.MEMBER_RECTS[3], (400, 300, 600, 500))
        self.assertEqual(camp.MEMBER_TEXELS, (28, 28, 228, 228))
        self.assertEqual(camp.member_at(300, 200), 0)
        self.assertEqual(camp.member_at(500, 400), 3)
        self.assertIsNone(camp.member_at(100, 100))
        self.assertEqual(len(camp.CELLS), 16)
        self.assertEqual(camp.CELLS[0], (200, 100, 300, 200))
        self.assertEqual(camp.CELLS[15], (500, 400, 600, 500))
        self.assertEqual(camp.cell_at(250, 150), 0)
        self.assertEqual(camp.cell_at(590, 490), 15)
        self.assertIsNone(camp.cell_at(650, 150))

    def test_paper_doll_slots(self):
        self.assertEqual(len(camp.SLOT_ORDER), 8)
        self.assertEqual(camp.SLOT_ORDER[0], 'helm')
        self.assertEqual(camp.SLOT_ORDER[-1], camp.SLOT_WEAPON)
        rects = camp.slot_rects(0)
        self.assertEqual(len(rects), 8)
        self.assertEqual(rects[0], ('helm', (206, 204, 250, 248)))
        self.assertEqual(rects[4][0], 'pants')
        self.assertEqual(camp.slot_at(228, 226), (0, 'helm'))
        self.assertEqual(camp.slot_at(228, 274), (0, 'pants'))
        self.assertIsNone(camp.slot_at(10, 10))
        # the slots stay inside their quarter
        for quarter in range(4):
            member = camp.MEMBER_RECTS[quarter]
            for slot, rect in camp.slot_rects(quarter):
                self.assertTrue(member[0] <= rect[0] < rect[2] <= member[2])
                self.assertTrue(member[1] <= rect[1] < rect[3] <= member[3])

    def test_slot_of_and_equipped(self):
        helm = FakeItem('cap', idb.ARMOR_HELM)
        boots = FakeItem('boots', idb.ARMOR_BOOTS)
        sword = FakeWeapon('saber')
        self.assertEqual(camp.slot_of(helm), 'helm')
        self.assertEqual(camp.slot_of(boots), 'boots')
        self.assertEqual(camp.slot_of(sword), 'weapon')
        self.assertIsNone(camp.slot_of(None))
        unit = FakeUnit('saber', armors=[helm, boots])
        equipped = camp.equipped_items(unit)
        self.assertEqual(equipped['helm'].name, 'cap')
        self.assertEqual(equipped['boots'].name, 'boots')
        self.assertEqual(equipped['weapon'].name, 'saber')
        self.assertEqual(sorted(camp.equipped_names(unit)), ['boots', 'cap', 'saber'])
        self.assertEqual(camp.equipped_items(None), {})

    def test_choice_of_the_member(self):
        camp_window = camp.Camp([FakeUnit(), FakeUnit()], names=['Zak', 'Kate'])
        self.assertEqual(camp_window.member_count(), 2)
        self.assertEqual(camp_window.member_name(1), 'Kate')
        self.assertEqual(camp_window.click(300, 200), 'select')
        self.assertEqual(camp_window.selected, 0)
        # an empty quarter is not a member and is not chosen
        self.assertIsNone(camp_window.click(300, 400))
        self.assertEqual(camp_window.selected, 0)
        self.assertEqual(camp_window.member_name(0), 'Zak')
        self.assertEqual(camp_window.member_name(2), '')

    def test_windows_and_cells(self):
        party = FakeParty()
        camp_window = camp.Camp([FakeUnit()], items=['sword'], spells=[], party=party,
                                repair_items=['cap'])
        self.assertEqual(camp_window.window, camp.CAMP_WINDOW)
        self.assertEqual(camp_window.money, 0)
        self.assertTrue(camp_window.open(camp.TRADE_WINDOW))
        self.assertFalse(camp_window.open('unknown'))
        self.assertEqual(camp_window.click(*_center(camp.CLOSE_RECT)), 'close')
        self.assertEqual(camp_window.window, camp.CAMP_WINDOW)
        camp_window.open(camp.REPAIR_WINDOW)
        self.assertEqual(camp_window.click(*_center(camp.CELLS[2])), 'cell 2')
        self.assertEqual(camp_window.cell_items(), ['cap'])
        camp_window.open(camp.TRADE_WINDOW)
        self.assertEqual(camp_window.cell_items(), ['sword'])
        party.money = 42
        self.assertEqual(camp_window.money, 42)

    def test_lines_of_the_information(self):
        unit = FakeUnit('saber.steel')
        camp_window = camp.Camp([unit])
        lines = camp_window.stat_lines(labels={'string infounit_3': 'Attack:',
                                               'string infounit_4': 'Defense',
                                               'string infounit_9': 'Damage:',
                                               'string infounit_14': 'Current weapon:'})
        self.assertEqual(lines, [('Attack:', '12'), ('Defense', '7'), ('Damage:', '3-8'),
                                 ('Current weapon:', 'saber.steel')])
        self.assertEqual(camp.Camp([None]).stat_lines(), [])


def _center(rect):
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2
