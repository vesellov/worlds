"""
The quick items of the section 4 of items.idb (the potions, the wands and the containers of the spells).

A record (tags as in idb):
    0   the name ("tiny potion 1")
    1   the code ("scroll", "wand", "dangersensor")
    5   the kind (10..21 the potions and their sizes, 26..39 the wands)
    6   the stack (0: one, -1: a pile)
    7   the weight
    10  the strength of the effect
    11  the level of the spell
    12  the price
    21  the flags of the quick slot
    22  the number of the icon
    23  the time of the use, seconds
    25  the spell "healing {e1}" (the device of magic.spell_from_text)

The same potion heals or enchants through its spell; the records with "spell container" take their spell
from the save (not read yet).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import idb

#------------------------------------------------------------------------------

SECTION_QUICK_ITEMS = 4

# the codes of the quick items
CODE_SCROLL = 'scroll'
CODE_WAND = 'wand'
CODE_SENSOR = 'dangersensor'

#------------------------------------------------------------------------------


class QuickItem(object):
    """
    A record of the section 4: a potion or a wand.
    """

    def __init__(self, index, fields):
        self.index = index
        self.name = idb._text(fields, 0)
        self.code = idb._text(fields, 1)
        self.kind = idb._dword(fields, 5)
        self.stack = idb._dword(fields, 6)
        self.weight = idb._float(fields, 7)
        self.effect = idb._float(fields, 10)
        self.spell_level = idb._dword(fields, 11)
        self.price = idb._float(fields, 12)
        self.flags = idb._dword(fields, 21)
        self.icon = idb._dword(fields, 22)
        self.use_time = idb._float(fields, 23)
        self.spell_text = idb._text(fields, 25)

    def __repr__(self):
        return 'QuickItem(%r, %s, spell %r)' % (self.name, self.code, self.spell_text)

    @property
    def is_potion(self):
        return bool(self.spell_text)

    @property
    def is_wand(self):
        return self.code == CODE_WAND

    @property
    def is_sensor(self):
        return self.code == CODE_SENSOR


class QuickItemDatabase(object):

    def __init__(self, items):
        self.items = items
        self._names = {item.name.lower(): item for item in items}

    def __repr__(self):
        return 'QuickItemDatabase(%d items)' % len(self.items)

    def item(self, name):
        """
        The quick item by its name without the case, None for a thing which is not quick.
        """
        return self._names.get((name or '').lower())

#------------------------------------------------------------------------------


def parse_quick_items(data):
    """
    The QuickItemDatabase of items.idb.
    """
    sections = {node.tag: node for node in idb.root(data).children()}
    records = sections.get(SECTION_QUICK_ITEMS)
    items = [QuickItem(i, r.fields()) for i, r in enumerate(records.children() if records is not None else [])]
    if _Debug:
        print('items.idb: %d quick items' % len(items))
    return QuickItemDatabase(items)
