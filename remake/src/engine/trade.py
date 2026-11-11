"""
The prices of the things and the deals of the camp with the money of the party (Party.money).

The price of an item is built by CItemManager__BuildItem (6a5d20) from the records of items.idb:
price = the price of the shape (tag 7) + the components of the shape (tag 13) times the price of a unit
of the material (tag 4). Without a material the price is the price of the shape. The fields are read by
eilib.idb (Shape.price, Shape.components, Material.price).

game.exe computes the deal value, the repair price and the construction prices by a virtual method
(+0xd8) of the camp window; its sale and repair shares are not read yet. The remake uses the shares
SALE_RATE and REPAIR_RATE for now. The condition of a thing is the share of its durability (engine.equipment.Item,
worn out by engine.combat): Trade.units gives the units of the party whose weapon and armor items are looked
up by the name; a thing of the bag is whole (Trade.conditions can set its condition).

The trade window of the camp sells the things of the party: a click on a cell takes the thing out of the
party and gives its price to Party.money. The repair window charges a thing of the party for the missing
part of its condition; with whole things it has nothing to repair. A repair sets the durability to the maximum
(6111e0, +0x20 = +0x24) and builds the armor of the unit again, so a broken item protects again.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import equipment

#------------------------------------------------------------------------------

# the share of the price which a merchant pays for a thing and which a repair of a worn out thing costs
SALE_RATE = 0.5
REPAIR_RATE = 1.0

#------------------------------------------------------------------------------


def worn_items(unit):
    """
    The weapon and the armor items of engine.equipment which the unit has in the fight.
    """
    fighter = getattr(unit, 'fighter', None)
    if fighter is None:
        return []
    items = [fighter.weapon_item] if fighter.weapon_item is not None else []
    return items + [item for _, item in sorted(fighter.armor_items.items())]


def repair_item(unit, item):
    """
    6111e0: the durability of the item is whole again (+0x20 = +0x24, CItem::FixDurability 4ba030), the armor
    of the unit is built again.
    """
    item.durability = item.max_durability
    fighter = getattr(unit, 'fighter', None)
    if fighter is not None:
        fighter.rebuild_armor()


def item_price(item):
    """
    6a5d20: the price of the shape plus the material it needs; 0 for an unknown item.
    """
    if item is None or item.shape is None:
        return 0
    price = float(getattr(item.shape, 'price', 0.0))
    material = getattr(item, 'material', None)
    if material is not None:
        price += float(getattr(item.shape, 'components', 0.0)) * float(getattr(material, 'price', 0.0))
    return int(round(price))


def sale_price(item):
    """
    What the party gets for the thing: a share of its price (the share of game.exe is not read yet).
    """
    price = item_price(item)
    if price <= 0:
        return 0
    return max(1, int(round(price * SALE_RATE)))


def repair_price(item, condition=1.0):
    """
    The price of the repair of a thing: the missing part of its condition times its price.
    condition is the share 0..1 of the condition which the thing still has.
    """
    condition = min(max(float(condition), 0.0), 1.0)
    if condition >= 1.0:
        return 0
    price = item_price(item)
    if price <= 0:
        return 0
    return max(1, int(round(price * (1.0 - condition) * REPAIR_RATE)))

#------------------------------------------------------------------------------


class Trade(object):
    """
    The deals of the party: selling the things of its common inventory and repairing its things.
    conditions: the name of a thing -> the share of its condition (1.0 when it is not known), for the things
    which no unit wears. units: a function which gives the units of the party (engine.unit.Unit).
    """

    def __init__(self, party, item_database=None, units=None):
        self.party = party
        self.item_database = item_database if item_database is not None else getattr(party, 'item_database', None)
        self.conditions = {}
        self.units = units

    def __repr__(self):
        return 'Trade(%d money, %d things)' % (self.money(), len(getattr(self.party, 'items', [])))

    def money(self):
        return getattr(self.party, 'money', 0)

    def item(self, name):
        """
        The Item of a name from items.idb, None when there is no such thing.
        """
        if self.item_database is None:
            return None
        return equipment.make_item(self.item_database, name)

    def worn(self, name):
        """
        (unit, item) of the most worn out item of the name which a unit of the party wears, (None, None)
        when no unit wears it.
        """
        best = (None, None)
        for unit in (self.units() if self.units is not None else []):
            for item in worn_items(unit):
                if item.name == name and (best[1] is None or item.condition < best[1].condition):
                    best = (unit, item)
        return best

    def condition(self, name):
        _, item = self.worn(name)
        if item is not None:
            return item.condition
        return self.conditions.get(name, 1.0)

    def set_condition(self, name, condition):
        self.conditions[name] = min(max(float(condition), 0.0), 1.0)

    def sale_price(self, name):
        return sale_price(self.item(name))

    def repair_price(self, name):
        return repair_price(self.item(name), self.condition(name))

    def sell(self, name):
        """
        The thing leaves the common inventory of the party, its price goes to Party.money. None when the
        party has no such thing.
        """
        items = getattr(self.party, 'items', None)
        if items is None or name not in items:
            return None
        price = self.sale_price(name)
        items.remove(name)
        self.conditions.pop(name, None)
        self.party.money += price
        self.party.changed('money', price)
        return price

    def repair(self, name, condition=None):
        """
        A thing with a missing part of its condition becomes whole for the price of the repair. None when
        the party has no such thing or nothing to repair, False when the party cannot pay.
        """
        unit, worn = self.worn(name)
        item = worn if worn is not None else self.item(name)
        if item is None:
            return None
        condition = self.condition(name) if condition is None else float(condition)
        price = repair_price(item, condition)
        if price <= 0:
            return None
        if self.money() < price:
            return False
        self.party.money -= price
        self.set_condition(name, 1.0)
        if worn is not None:
            repair_item(unit, worn)
        return price
