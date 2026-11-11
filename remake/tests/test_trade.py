"""
Tests of engine.trade: the price of an item from items.idb, the sale and the repair, the money of the party.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from engine import combat
from engine import trade

#------------------------------------------------------------------------------


class FakeShape(object):

    def __init__(self, price=0.0, components=0.0, weapon=False):
        self.price = price
        self.components = components
        self.type = 0
        self.number = 1
        self.variant = 0
        self.weight = 1.0
        self.is_weapon = weapon


class FakeMaterial(object):

    def __init__(self, price=0.0):
        self.price = price
        self.code = 'lt'


class FakeItem(object):

    def __init__(self, shape, material=None):
        self.name = 'thing'
        self.shape = shape
        self.material = material


class FakeDatabase(object):

    def __init__(self, items):
        self.items = items

    def item(self, name):
        return self.items.get(name, (None, None))


class FakeParty(object):

    def __init__(self, items=(), money=0):
        self.items = list(items)
        self.money = money
        self.changes = []

    def changed(self, kind, value):
        self.changes.append((kind, value))


class TradeTest(unittest.TestCase):

    def test_item_price(self):
        # 6a5d20: the price of the shape plus the components times the price of the material
        self.assertEqual(trade.item_price(FakeItem(FakeShape(400.0))), 400)
        self.assertEqual(trade.item_price(FakeItem(FakeShape(400000.0, 3.0), FakeMaterial(5000.0))), 415000)
        self.assertEqual(trade.item_price(FakeItem(FakeShape(16.0, 8.0), FakeMaterial(8.0))), 80)
        self.assertEqual(trade.item_price(None), 0)

    def test_sale_and_repair_prices(self):
        item = FakeItem(FakeShape(400000.0, 3.0), FakeMaterial(5000.0))
        self.assertEqual(trade.sale_price(item), 207500)
        self.assertEqual(trade.repair_price(item, 1.0), 0)
        self.assertEqual(trade.repair_price(item, 0.5), 207500)
        self.assertEqual(trade.repair_price(item, 0.0), 415000)
        self.assertEqual(trade.sale_price(None), 0)
        self.assertEqual(trade.repair_price(FakeItem(FakeShape(0.0)), 0.0), 0)

    def test_sell(self):
        database = FakeDatabase({'saber.steel': (FakeShape(400.0), None)})
        party = FakeParty(['saber.steel', 'axe.thick'], money=10)
        deals = trade.Trade(party, database)
        self.assertEqual(deals.sale_price('saber.steel'), 200)
        self.assertEqual(deals.sell('saber.steel'), 200)
        self.assertEqual(party.items, ['axe.thick'])
        self.assertEqual(party.money, 210)
        self.assertEqual(party.changes, [('money', 200)])
        self.assertIsNone(deals.sell('saber.steel'))
        self.assertIsNone(deals.sell('unknown'))

    def test_repair(self):
        database = FakeDatabase({'cap.leather': (FakeShape(100.0), None)})
        party = FakeParty(['cap.leather'], money=60)
        deals = trade.Trade(party, database)
        self.assertEqual(deals.repair_price('cap.leather'), 0)
        self.assertIsNone(deals.repair('cap.leather'))
        deals.set_condition('cap.leather', 0.5)
        self.assertEqual(deals.repair_price('cap.leather'), 50)
        self.assertEqual(deals.repair('cap.leather'), 50)
        self.assertEqual(party.money, 10)
        self.assertEqual(deals.condition('cap.leather'), 1.0)
        deals.set_condition('cap.leather', 0.0)
        self.assertFalse(deals.repair('cap.leather'))
        self.assertEqual(party.money, 10)
        self.assertIsNone(deals.repair('unknown'))

    def test_changes_go_to_the_party(self):
        database = FakeDatabase({'cap.leather': (FakeShape(40.0), None)})
        party = FakeParty(['cap.leather'], money=0)
        deals = trade.Trade(party, database)
        deals.sell('cap.leather')
        self.assertEqual(party.money, 20)
        self.assertEqual(trade.item_price(FakeItem(FakeShape(40.0), FakeMaterial(0.0))), 40)



class FakeWornItem(object):

    def __init__(self, name, shape, durability, max_durability):
        self.name = name
        self.shape = shape
        self.material = None
        self.durability = durability
        self.max_durability = max_durability

    @property
    def condition(self):
        return self.durability / self.max_durability


class FakeFighter(object):

    def __init__(self, weapon=None, armors=None):
        self.weapon_item = weapon
        self.armor_items = dict(armors or {})
        self.rebuilt = 0

    def rebuild_armor(self):
        self.rebuilt += 1


class FakeUnit(object):

    def __init__(self, fighter):
        self.fighter = fighter


class WornRepairTest(unittest.TestCase):

    def test_repair_of_a_worn_item(self):
        """
        The condition of a worn item is the share of its durability, the repair makes it whole (6111e0)
        and builds the armor again.
        """
        shape = FakeShape(price=100.0)
        plate = FakeWornItem('plate', shape, 25.0, 100.0)
        fighter = FakeFighter(armors={1: plate})
        unit = FakeUnit(fighter)
        party = FakeParty(money=1000)
        deal = trade.Trade(party, FakeDatabase({'plate': (shape, None)}), units=lambda: [unit])
        self.assertEqual(trade.worn_items(unit), [plate])
        self.assertAlmostEqual(deal.condition('plate'), 0.25)
        self.assertEqual(deal.repair_price('plate'), 75)
        self.assertEqual(deal.repair('plate'), 75)
        self.assertEqual(plate.durability, 100.0)
        self.assertEqual(fighter.rebuilt, 1)
        self.assertEqual(party.money, 925)
        self.assertIsNone(deal.repair('plate'))
        self.assertEqual(combat.MESSAGE_ARMOR_BROKEN, 0x23)


if __name__ == '__main__':
    unittest.main()
