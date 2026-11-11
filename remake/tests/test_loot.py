"""
Tests of eilib.lootdb and engine.loot: the lists and the money of a prototype, the rolls and the transfer
of the things of a body to the party.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random
import struct
import unittest

#------------------------------------------------------------------------------

from eilib import lootdb
from engine import loot as loot_module

#------------------------------------------------------------------------------


def _field(tag, data):
    return bytes([tag, len(data) << 1]) + data


def _floats(*values):
    return b''.join(_field(index, struct.pack('<f', value)) for index, value in enumerate(values))


class FakeItem(object):

    def __init__(self, name):
        self.name = name


class FakeEquipment(object):

    def __init__(self, armors=(), weapon=None):
        self.armors = {index: FakeItem(name) for index, name in enumerate(armors)}
        self.weapon = FakeItem(weapon) if weapon is not None else None


class FakeFigure(object):

    def __init__(self, equipment=None):
        self.equipment = equipment


class FakeUnit(object):

    def __init__(self, equipment=None, quest_items=()):
        self.figure = FakeFigure(equipment)
        self.quest_items = [FakeItem(name) for name in quest_items]
        self.looted = False
        self.prototype_name = None


class FakeParty(object):

    def __init__(self):
        self.items = []
        self.money = 0

    def give_item(self, name):
        self.items.append(name)

    def give_money(self, amount):
        self.money += amount


class LootDatabaseTest(unittest.TestCase):

    def test_item_names(self):
        data = _field(1, b'graywolfskin\x00') + _field(1, b'material.thin [1]\x00')
        self.assertEqual(lootdb.item_names(data), ['graywolfskin', 'material.thin [1]'])
        self.assertEqual(lootdb.item_names(b''), [])

    def test_money_of(self):
        data = _floats(100.0, 0.0, 77.0, 307.0)
        self.assertEqual(lootdb.money_of(data), (100.0, 77.0, 307.0))
        self.assertEqual(lootdb.money_of(b''), (0.0, 0.0, 0.0))

    def test_loot_prototype(self):
        fields = {lootdb.LOOT_TAG: _field(1, b'graywolfskin\x00'),
                  lootdb.ARMOR_TAG: _field(1, b'gipat low helm.thin\x00'),
                  lootdb.WEAPON_TAG: b'long sword.iron\x00',
                  lootdb.MONEY_TAG: _floats(100.0, 0.0, 5.0, 10.0),
                  lootdb.MONEY_SECOND_TAG: _floats(10.0, 0.0, 0.0, 0.0)}
        prototype = lootdb.prototype_of('Wolf', fields)
        self.assertEqual(prototype.items, ['graywolfskin'])
        self.assertEqual(prototype.armors, ['gipat low helm.thin'])
        self.assertEqual(prototype.weapon, 'long sword.iron')
        self.assertEqual(prototype.money, (100.0, 5.0, 10.0))
        self.assertEqual(prototype.money_second, (10.0, 0.0, 0.0))


class LootTest(unittest.TestCase):

    def test_roll_money(self):
        rng = random.Random(1)
        self.assertEqual(loot_module.roll_money((0.0, 5.0, 10.0), rng), 0)
        self.assertEqual(loot_module.roll_money((100.0, 7.0, 7.0), rng), 7)
        self.assertTrue(5 <= loot_module.roll_money((100.0, 5.0, 10.0), rng) <= 10)
        self.assertTrue(5 <= loot_module.roll_money((100.0, 10.0, 5.0), rng) <= 10)

    def test_take_all(self):
        unit = FakeUnit(FakeEquipment(['helm', 'boots'], 'sword'), quest_items=['key'])
        equipment = unit.figure.equipment
        entries = [loot_module.LootEntry('helm', loot_module.KIND_ARMOR),
                   loot_module.LootEntry('sword', loot_module.KIND_WEAPON),
                   loot_module.LootEntry('graywolfskin', loot_module.KIND_ITEM),
                   loot_module.LootEntry('key', loot_module.KIND_QUEST),
                   loot_module.LootEntry('money', loot_module.KIND_MONEY, 12)]
        loot = loot_module.Loot(unit, entries)
        party = FakeParty()
        taken = loot.take_all(party)
        self.assertEqual(len(taken), 5)
        self.assertTrue(loot.is_empty)
        self.assertEqual(party.items, ['helm', 'sword', 'graywolfskin', 'key'])
        self.assertEqual(party.money, 12)
        self.assertEqual(unit.quest_items, [])
        self.assertEqual(equipment.weapon, None)
        self.assertEqual(list(equipment.armors.values())[0].name, 'boots')

    def test_take_one(self):
        party = FakeParty()
        loot = loot_module.Loot(None, [loot_module.LootEntry('a'), loot_module.LootEntry('b')])
        self.assertTrue(loot.take(1, party))
        self.assertEqual(party.items, ['b'])
        self.assertEqual(len(loot.entries), 1)
        self.assertFalse(loot.take(5, party))

    def test_make_loot_marks_the_body(self):
        class Tables(object):

            def get(self, name):
                return lootdb.LootPrototype('Wolf', ['skin'], ['helm'], 'claw', (100.0, 3.0, 3.0))

        class GameData(object):

            def cached(self, key, loader):
                return loader()

            def archive(self, name):
                raise AssertionError('the real database must not be read')

        game_data = GameData()
        game_data.A = None
        # the prototype lookup is replaced for the test: the table of the game data would parse units.udb
        original = loot_module.lootdb.of_game
        loot_module.lootdb.of_game = lambda data: {'wolf': Tables().get('wolf')}
        try:
            unit = FakeUnit(FakeEquipment([], None))
            unit.prototype_name = 'Wolf'
            loot = loot_module.make_loot(game_data, unit, rng=random.Random(1))
        finally:
            loot_module.lootdb.of_game = original
        self.assertTrue(unit.looted)
        self.assertIn('skin', [entry.name for entry in loot.entries])
        self.assertIn('helm', [entry.name for entry in loot.entries])
        self.assertIn('claw', [entry.name for entry in loot.entries])
        self.assertIn('money', [entry.name for entry in loot.entries])
