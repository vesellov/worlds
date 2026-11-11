"""
The loot of a dead unit (game.exe 461cb0, the transfer 5942d0) and the window which takes it.

FUN_00461cb0 @ 461cb0 builds two lists of the things of the body: the things of the prototype
(eilib.lootdb), the quick items of the unit, its quest items and the armor and the weapon which it wears;
the money is rolled from the two ranges of the prototype. The body gets the flag 0x100000 at once, before
the player takes anything, so WasLooted of the scripts answers True as soon as the window is opened
(first_scan.md, section 27). FUN_005942d0 @ 5942d0 moves the things and the money into the party.

The remake keeps the entries of the loot in the order of the lists of game.exe and gives them to
engine.party.Party by the names: give_item puts a quest item into the quest items when items.idb knows it,
give_money adds the money. The armor and the weapon of the body are taken from engine.equipment of the
figure of the unit; taking them removes them from that equipment (the model of the corpse is not built
again, the main flow will do it when the equipment of the units is its part).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import random

#------------------------------------------------------------------------------

from eilib import lootdb

#------------------------------------------------------------------------------

KIND_ITEM = 'item'
KIND_ARMOR = 'armor'
KIND_WEAPON = 'weapon'
KIND_QUEST = 'quest'
KIND_MONEY = 'money'

MONEY_NAME = 'money'

#------------------------------------------------------------------------------


class LootEntry(object):
    """
    One line of the loot: the name of the thing (of the money for KIND_MONEY), its kind and the amount.
    """

    def __init__(self, name, kind=KIND_ITEM, amount=1):
        self.name = name
        self.kind = kind
        self.amount = amount

    def __repr__(self):
        return 'LootEntry(%s %s %s)' % (self.kind, self.name, self.amount)

    @property
    def is_money(self):
        return self.kind == KIND_MONEY

#------------------------------------------------------------------------------


def roll_money(money, rng):
    """
    461cb0: the chance in percents, then a value between the minimum and the maximum rounded to an integer.
    """
    chance, minimum, maximum = money
    if chance <= 0.0:
        return 0
    if rng.randint(1, 100) > chance:
        return 0
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    if maximum <= minimum:
        return max(int(round(minimum)), 0)
    return max(int(round(rng.uniform(minimum, maximum))), 0)

#------------------------------------------------------------------------------


class Loot(object):
    """
    The things of one body: the entries and the unit they belong to.
    """

    def __init__(self, unit=None, entries=()):
        self.unit = unit
        self.entries = list(entries)

    def __repr__(self):
        return 'Loot(%d entries)' % len(self.entries)

    @property
    def is_empty(self):
        return not self.entries

    def take(self, index, party):
        """
        Give one entry to the party, True when it was taken.
        """
        if not 0 <= index < len(self.entries):
            return False
        entry = self.entries[index]
        if entry.is_money:
            party.give_money(entry.amount)
        else:
            party.give_item(entry.name)
            self.forget_item(entry)
        self.entries.pop(index)
        return True

    def take_all(self, party):
        """
        Give every entry to the party, the list of the taken entries.
        """
        taken = list(self.entries)
        for entry in taken:
            if entry.is_money:
                party.give_money(entry.amount)
            else:
                party.give_item(entry.name)
                self.forget_item(entry)
        self.entries = []
        return taken

    def forget_item(self, entry):
        """
        Remove a quest item or a worn thing from the unit itself.
        """
        unit = self.unit
        if unit is None:
            return
        if entry.kind == KIND_QUEST:
            for item in list(getattr(unit, 'quest_items', [])):
                if item.name == entry.name:
                    unit.quest_items.remove(item)
                    break
        equipment = getattr(getattr(unit, 'figure', None), 'equipment', None)
        if equipment is None:
            return
        if entry.kind == KIND_WEAPON and equipment.weapon is not None and equipment.weapon.name == entry.name:
            equipment.weapon = None
        elif entry.kind == KIND_ARMOR:
            for slot, item in list(equipment.armors.items()):
                if item.name == entry.name:
                    del equipment.armors[slot]
                    break

#------------------------------------------------------------------------------


def make_loot(game_data, unit=None, prototype_name=None, equipment=None, quest_items=None, rng=None):
    """
    The loot of a unit: its prototype things and money, its quest items and the armor and the weapon which
    it wears. The body is marked as looted at once, like 461cb0 does it when the lists are built.
    """
    prototype_name = prototype_name or getattr(unit, 'prototype_name', None)
    prototype = find_prototype(game_data, prototype_name)
    entries = []
    if equipment is None:
        equipment = getattr(getattr(unit, 'figure', None), 'equipment', None)
    if equipment is not None:
        for slot, item in sorted(equipment.armors.items()):
            entries.append(LootEntry(item.name, KIND_ARMOR))
        if equipment.weapon is not None:
            entries.append(LootEntry(equipment.weapon.name, KIND_WEAPON))
    if prototype is not None:
        if equipment is None or not equipment.armors:
            entries.extend(LootEntry(name, KIND_ARMOR) for name in prototype.armors)
        if (equipment is None or equipment.weapon is None) and prototype.weapon:
            entries.append(LootEntry(prototype.weapon, KIND_WEAPON))
        entries.extend(LootEntry(name, KIND_ITEM) for name in prototype.items)
        if rng is None:
            rng = random.Random()
        amount = roll_money(prototype.money, rng)
        if amount > 0:
            entries.append(LootEntry(MONEY_NAME, KIND_MONEY, amount))
        amount = roll_money(prototype.money_second, rng)
        if amount > 0:
            entries.append(LootEntry(MONEY_NAME, KIND_MONEY, amount))
    quest_items = quest_items
    if quest_items is None:
        quest_items = getattr(unit, 'quest_items', [])
    entries.extend(LootEntry(item.name, KIND_QUEST) for item in quest_items)
    if unit is not None:
        unit.looted = True
    return Loot(unit, entries)


def find_prototype(game_data, prototype_name):
    """
    The LootPrototype of units.udb by the name, None for an unknown one.
    """
    if not prototype_name:
        return None
    return lootdb.of_game(game_data).get(prototype_name.lower())
