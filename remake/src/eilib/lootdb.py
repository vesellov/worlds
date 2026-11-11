"""
The things which a unit of units.udb leaves when it dies (CUnitPrototype::Read, game.exe 52d740).

The prototype of a unit keeps its loot in the fields (the tags are the order of the format of EIrepack):

- tag 27, a node of four floats: the chance in percents, an unused value, the minimum and the maximum of
  the money (FUN_00461cb0 @ 461cb0 reads +0xc8, +0xd0 and +0xd4 of the prototype);
- tag 28, the same node for the second money roll (chance 10 percents for most units, usually 0 money);
- tag 29, the list of the things of the body (CUnit::CreatePrototypeItems @ 5141f0 puts them into +0xe8);
- tag 31, the spells of the unit (+0x100), not looted;
- tag 32, the armor of the unit by default (+0x10c) and tag 33, its weapon (+0x118).

A list is a sequence of nodes of the tag 1 with the name of an item and its ending zero; the length of
a node is written with its lowest bit 0, so the byte is twice the number of the bytes (game.exe 409070).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import struct

#------------------------------------------------------------------------------

from eilib import idb

#------------------------------------------------------------------------------

MONEY_TAG = 27
MONEY_SECOND_TAG = 28
LOOT_TAG = 29
SPELLS_TAG = 31
ARMOR_TAG = 32
WEAPON_TAG = 33

# the fields of the nodes of the money: the chance, the minimum and the maximum
CHANCE_FIELD = 0
MIN_FIELD = 2
MAX_FIELD = 3

DATABASE_NAME = 'res/database.res'
UNITS_NAME = 'units.udb'

#------------------------------------------------------------------------------


class LootPrototype(object):
    """
    The loot of a prototype of units.udb: the things of the body, the armor and the weapon by default and
    two rolls of money (chance in percents, minimum and maximum).
    """

    def __init__(self, name, items=(), armors=(), weapon=None, money=(0.0, 0.0, 0.0),
                 money_second=(0.0, 0.0, 0.0)):
        self.name = name
        self.items = list(items)
        self.armors = list(armors)
        self.weapon = weapon
        self.money = tuple(money)
        self.money_second = tuple(money_second)

    def __repr__(self):
        return 'LootPrototype(%s, %d items, %s, money %s)' % (self.name, len(self.items), self.weapon,
                                                             self.money)

#------------------------------------------------------------------------------


def item_names(data):
    """
    The names of a list of loot fields: the data of a node, every child is the tag 1 and the name.
    """
    result = []
    for child in idb.Node(0, data).children():
        name = bytes(child.data).split(b'\0', 1)[0].decode(idb.ENCODING, errors='replace')
        if name:
            result.append(name)
    return result


def money_of(data):
    """
    (chance, minimum, maximum) of the node of the money, (0, 0, 0) when the node is empty.
    """
    values = {}
    for child in idb.Node(0, data).children():
        if len(child.data) >= 4:
            values[child.tag] = struct.unpack_from('<f', child.data)[0]
    return (values.get(CHANCE_FIELD, 0.0), values.get(MIN_FIELD, 0.0), values.get(MAX_FIELD, 0.0))


def prototype_of(name, fields):
    return LootPrototype(name,
                         item_names(fields.get(LOOT_TAG) or b''),
                         item_names(fields.get(ARMOR_TAG) or b''),
                         idb._text(fields, WEAPON_TAG) or None,
                         money_of(fields.get(MONEY_TAG) or b''),
                         money_of(fields.get(MONEY_SECOND_TAG) or b''))


def parse(data):
    """
    name -> LootPrototype of units.udb.
    """
    sections = {node.tag: node for node in idb.root(data).children()}
    section = sections.get(idb.SECTION_UNIT_PROTOTYPES)
    result = {}
    for record in (section.children() if section is not None else []):
        fields = record.fields()
        name = idb._text(fields, 0)
        if name:
            result[name.lower()] = prototype_of(name, fields)
    return result


def of_game(game_data):
    """
    The cached name -> LootPrototype of the game data.
    """
    def _load():
        archive = game_data.archive(DATABASE_NAME)
        return parse(archive.read(UNITS_NAME))
    return game_data.cached(('lootdb', ), _load)
