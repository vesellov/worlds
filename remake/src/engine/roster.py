"""
The parties of a player between the zones (CParty 661010, the player object of the server 594180).

A player has the party by default (+0x70) and the parties made by CreateParty( player, name ) (0x9a, 4a5e90),
a list of CParty found by their names (4a5e10: an unknown name gives the one named ""). The name of the party
which goes into the next zones is kept in +0x24: SetCurrentParty( player, name ) (0x9c), "" for the one by default.
A party has the records of its units (+0x30, 0x784 bytes each): the name +4 (and its hash +0x1c), the prototype
of units.udb +0x14 and from +0x20 the stats (0x704 bytes: the numbers, the skills, the experience).

The units are named "party::unit" in the scripts (4baec0: the text before "::" is the party, the rest is the unit;
without "::" the party by default):
    - AddUnitToParty( player, "party::unit", prototype ) (0x86, 662890) adds a record made of the prototype (6621b0);
    - RemoveUnitFromParty( player, "party::unit" ) (0x87, 6620b0) removes it;
    - CopyStats( player, from, to ) (0x9d) copies the stats of a record to another (and the text +0);
    - CopyItems( player, from, to ) (0xd4, 6604c0) gives the items of a unit to another;
    - CopyLoot( player, from, to ) (0xcb) makes the things of a party those of another,
      AddLoot( player, from, to ) (0xe0) adds them;
    - GetMercsNumber( player ) (0xd1, 591740): the living units of the party in the zone without the hero.
RedeployParty( player ) (0xd3) brings the current party into the zone again (590700).

The remake: a record keeps the name of the prototype whose numbers it has (stats_name), the experience, the attributes
and the skills of a hero or a mercenary (engine.experience.HeroStats, made of the template of the prototype when
the unit first comes into a zone), the items by their names and the model; the things of the parties are lists
of the names of the items.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import sav
from engine import experience

#------------------------------------------------------------------------------

# 4baec0: "party::unit"
SEPARATOR = '::'

# the party by default (+0x70) and the name which SetCurrentParty gives to it
DEFAULT_PARTY = ''

# the name of the hero in the party by default
HERO_NAME = 'Hero'

#------------------------------------------------------------------------------


class UnitRecord(object):
    """
    A unit of a party (0x784 bytes): its name, the prototype of units.udb (the look, the type, the voice),
    the prototype whose numbers it has (CopyStats copies them), the names of its items and the model when
    the prototype does not tell it.
    """

    def __init__(self, name, prototype_name, stats_name=None, item_names=None, model_name=None):
        self.name = name
        self.prototype_name = prototype_name
        self.stats_name = stats_name or prototype_name
        # None: the weapon of the prototype
        self.item_names = None if item_names is None else list(item_names)
        self.model_name = model_name
        # engine.experience.HeroStats, None until the unit comes into a zone (or for a monster)
        self.hero_stats = None
        # the names of the four quick items of the belt (UNIT_QUICK_ITEMS of the unit), None for the
        # quick items of the things of the unit
        self.quick_items = None
        # the name -> engine.equipment.Item which the unit wears, they keep their wear when the unit is built
        # again (the remake's simplification: an item taken off into the bag is whole again)
        self.worn_items = {}

    def __repr__(self):
        return 'UnitRecord(%s %s%s)' % (self.name, self.prototype_name,
                                        '' if self.stats_name == self.prototype_name else ' as ' + self.stats_name)


class UnitParty(object):

    def __init__(self, name):
        self.name = name
        self.units = []
        # the things of the party, the names of the items
        self.loot = []

    def __repr__(self):
        return 'UnitParty(%r %s)' % (self.name, [u.name for u in self.units])

    def unit(self, name):
        """
        662170: the record by its name (the names are compared without the case of the letters).
        """
        name = name.lower()
        for u in self.units:
            if u.name.lower() == name:
                return u
        return None

    def lead(self):
        return self.units[0] if self.units else None

#------------------------------------------------------------------------------


class Roster(object):
    """
    The parties of one player.
    """

    def __init__(self):
        self.default = UnitParty(DEFAULT_PARTY)
        self.parties = []
        # +0x24
        self.current_name = DEFAULT_PARTY

    def __repr__(self):
        return 'Roster(%r %s)' % (self.current_name, [p.name for p in self.parties])

    def party(self, name):
        """
        4a5e10 and the party by default; None when there is none.
        """
        if name == DEFAULT_PARTY:
            return self.default
        fallback = None
        for p in self.parties:
            if p.name.lower() == name.lower():
                return p
            if p.name == DEFAULT_PARTY:
                fallback = p
        return fallback

    def records(self):
        """
        The UnitRecord of all the parties.
        """
        result = []
        for party in [self.default] + [p for p in self.parties if p is not self.default]:
            result.extend(party.units)
        return result

    def split(self, path):
        """
        4baec0: "party::unit" -> (the party or None, the name of the unit).
        """
        party_name, separator, unit_name = path.partition(SEPARATOR)
        if not separator:
            party_name, unit_name = DEFAULT_PARTY, path
        return self.party(party_name), unit_name

    def record(self, path):
        party, unit_name = self.split(path)
        return party.unit(unit_name) if party is not None else None

    def current(self):
        return self.party(self.current_name) or self.default

    def create_party(self, name):
        """
        4a5e90: a new empty party at the end of the list; an old one of the same name is found first after it.
        """
        party = UnitParty(name)
        self.parties = [p for p in self.parties if p.name.lower() != name.lower()]
        self.parties.append(party)
        return party

    def add_unit(self, path, prototype_name, model_name=None):
        party, unit_name = self.split(path)
        if party is None:
            return None
        record = UnitRecord(unit_name, prototype_name, model_name=model_name)
        party.units.append(record)
        return record

    def remove_unit(self, path):
        party, unit_name = self.split(path)
        record = party.unit(unit_name) if party is not None else None
        if record is None:
            return False
        party.units.remove(record)
        return True

    def copy_stats(self, source_path, dest_path):
        source = self.record(source_path)
        dest = self.record(dest_path)
        if source is None or dest is None:
            return False
        dest.stats_name = source.stats_name
        dest.hero_stats = source.hero_stats.copy() if source.hero_stats is not None else None
        return True

    def copy_items(self, source_path, dest_path):
        source = self.record(source_path)
        dest = self.record(dest_path)
        if source is None or dest is None:
            return False
        dest.item_names = None if source.item_names is None else list(source.item_names)
        return True

    def copy_loot(self, source_name, dest_name, add=False):
        """
        CopyLoot (0xcb) and AddLoot (0xe0, add=True).
        """
        source = self.party(source_name)
        dest = self.party(dest_name)
        if source is None or dest is None or source is dest:
            return False
        if not add:
            dest.loot = []
        dest.loot.extend(source.loot)
        return True

    def set_current(self, name):
        """
        SetCurrentParty: the party which comes into the next zones; the old one when the name is not known.
        """
        party = self.party(name)
        if party is None:
            return None
        old = self.current()
        self.current_name = party.name
        return old

#------------------------------------------------------------------------------


def new_roster(prototype_name, item_names=None, model_name=None):
    """
    The roster of a new game: the hero in the party by default.
    """
    roster = Roster()
    roster.add_unit(HERO_NAME, prototype_name, model_name=model_name)
    roster.default.lead().item_names = None if item_names is None else list(item_names)
    return roster


def saved_item_name(item, items, quick_items=None):
    """
    The name of an item of a save of the game (eilib.sav.SavedObject) like the maps name them: "shape.material"
    for a weapon or an armor (the indices of items.idb), the name of the quick item of quick.idb; None for
    another class.
    """
    shape_index, material_index = item.high, item.low
    if item.class_id in (sav.CLASS_WEAPON, sav.CLASS_ARMOR):
        shapes = items.weapons if item.class_id == sav.CLASS_WEAPON else items.armors
        if shape_index >= len(shapes) or material_index >= len(items.materials):
            return None
        return '%s.%s' % (shapes[shape_index].name, items.materials[material_index].name)
    if item.class_id == sav.CLASS_QUICK_ITEM and quick_items is not None:
        if shape_index < len(quick_items.items):
            return quick_items.items[shape_index].name
    return None


def record_from_saved(unit, party, items, quick_items=None, perks=None, settings=None):
    """
    UnitRecord of a unit of a party of a save of the game (eilib.sav.SavedUnit of eilib.sav.SavedParty) and
    the names of its other items. Of its four lists the first keeps the weapon in the hands and the third the armor
    on the unit (both are worn), the fourth the quick items; the second was empty in the saves seen.
    """
    lists = list(unit.item_lists) + [[]] * 4
    worn_ids = lists[0] + lists[2]
    worn = []
    other = []
    for object_id in unit.item_ids:
        item = party.item(object_id)
        name = saved_item_name(item, items, quick_items) if item is not None else None
        if name is None:
            continue
        (worn if object_id in worn_ids else other).append(name)
    record = UnitRecord(unit.name, unit.prototype_name, stats_name=unit.stats_name or None, item_names=worn,
                        model_name=unit.model_name or None)
    if unit.stats is not None:
        record.hero_stats = experience.HeroStats.from_saved(unit.stats, perks, settings)
    return record, other
