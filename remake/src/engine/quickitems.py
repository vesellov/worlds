"""
The quick items of the hero: the potions and the wands of the section 4 of items.idb.

The use of a potion (552c60 for the scrolls, the item takes its spell from the tag 25 and the mana of the
scroll when the spell needs it): the spell of the item is applied to its owner at once, without the words
and the gestures of a cast: CUnitServer gives the effect to the unit (67f830 through 58e850). The healing
goes to combat.heal, the strength, speed, enlarge and shrink become the enchantments of the fighter.
The item is spent after the use; game.exe plays a sound of the belt and the time of the use (tag 23).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import magic

#------------------------------------------------------------------------------

# the items of the belt of the hero: game.exe keeps the names in UNIT_QUICK_ITEMS, the remake takes the
# quick items of the things of the party which have a record in the section 4 of items.idb
BELT_SIZE = 4

#------------------------------------------------------------------------------


def spell_from_text(spells, text):
    """
    "healing {e2;e2;e2}" -> magic.Spell("healing", ["e2", "e2", "e2"]), None for an empty text or an
    unknown spell. The braces hold the modifiers separated by ";" or ",".
    """
    text = (text or '').strip()
    if not text:
        return None
    name, separator, rest = text.partition('{')
    codes = []
    if separator:
        rest = rest.rsplit('}', 1)[0]
        codes = [code.strip() for code in rest.replace(',', ';').split(';') if code.strip()]
    return magic.make_spell(spells, name.strip(), codes)


def use(unit, quick_item, spells):
    """
    The effect of a quick item on its owner: (True, the health which came back) when the spell was applied,
    (False, 0.0) for a wand, an empty container or a unit without a fighter.
    """
    if unit is None or unit.fighter is None or quick_item is None:
        return False, 0.0
    spell = spell_from_text(spells, quick_item.spell_text)
    if spell is None:
        return False, 0.0
    healed = magic.apply(spell, unit)
    if _Debug:
        print('quick item %s: %s' % (quick_item.name, spell.name))
    return True, healed


def belt(item_names, database, size=BELT_SIZE, order=None):
    """
    The quick items of a list of the names of the things of the party: [(QuickItem, count)] in the order of
    their first names, the first four of them are item1..item4 of keyboard.ini. order gives the names of the
    belt of the hero (UNIT_QUICK_ITEMS), the counts come from item_names.
    """
    counts = {}
    sequence = []
    for name in item_names:
        quick_item = database.item(name)
        if quick_item is None:
            continue
        key = quick_item.name
        if key not in counts:
            counts[key] = 0
            sequence.append(quick_item)
        counts[key] += 1
    if order is not None:
        by_name = {quick_item.name.lower(): quick_item for quick_item in sequence}
        sequence = [by_name[name.lower()] for name in order if name.lower() in by_name][:size]
    return [(quick_item, counts[quick_item.name]) for quick_item in sequence[:size]]


def take(item_names, quick_item):
    """
    The name of a used quick item leaves the list of the things (one of the copies).
    """
    if quick_item is None:
        return False
    wanted = quick_item.name.lower()
    for index, name in enumerate(item_names):
        if (name or '').lower() == wanted:
            del item_names[index]
            return True
    return False
