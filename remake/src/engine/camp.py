"""
The camp screen of the party: the four quarters of CCampWindow__Build (621fc0), the information window of
622d70, the skills window of 624ed0, the trade and repair lists of 627060 and 6279a0 and the constructors of
628c80 and 628080. The strip of items at the top or at the bottom is engine.inventory.InventoryStrip (621120).

The window of a camp covers the whole screen: the four quarters 200 x 200 at (200, 100), (400, 100),
(200, 300) and (400, 300) with the textures camp1..4 over campm1..4, cut from the texels (28, 28)-(228, 228);
the information at (0, 100)-(200, 400); the skills, the trade, the repair and the constructors at
(200, 100)-(600, 500). The trade and repair windows are a grid of 16 cells 100 x 100 with the textures
trade1..4 and repair1..4.

The remake does not read the shop lists, the recipes and the paper dolls of the camp of game.exe yet: the
windows show the items of the party and its spells, and the buttons which need the missing data answer
"not made yet" through the client. The tests cover the geometry, the choice of the member and the lists.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import idb

#------------------------------------------------------------------------------

# the windows of the camp
CAMP_WINDOW = 'camp'
INFO_WINDOW = 'info'
SKILLS_WINDOW = 'skills'
TRADE_WINDOW = 'trade'
REPAIR_WINDOW = 'repair'
ITEM_CONSTRUCTOR_WINDOW = 'item constructor'
SPELL_CONSTRUCTOR_WINDOW = 'spell constructor'
WINDOWS = (CAMP_WINDOW, INFO_WINDOW, SKILLS_WINDOW, TRADE_WINDOW, REPAIR_WINDOW,
           ITEM_CONSTRUCTOR_WINDOW, SPELL_CONSTRUCTOR_WINDOW)

# 621fc0: four quarters of 200 x 200 from the texels 28..228 of camp1..4 over campm1..4
MEMBER_RECTS = ((200, 100, 400, 300), (400, 100, 600, 300), (200, 300, 400, 500), (400, 300, 600, 500))
MEMBER_TEXELS = (28, 28, 228, 228)
MEMBER_TEXTURES = ('camp1', 'camp2', 'camp3', 'camp4')
MEMBER_BACKGROUNDS = ('campm1', 'campm2', 'campm3', 'campm4')

# 621fc0: the strip of the inventory over the bottom or the top of the camp
STRIP_BOTTOM_RECT = (0, 500, 800, 600)
STRIP_TOP_RECT = (0, 0, 800, 100)

# 622d70: the information about the member at the left
INFO_RECT = (0, 100, 200, 400)
INFO_BUTTONS = ((0, 125, 30, 145), (170, 125, 200, 145))

# 624ed0, 627060, 6279a0, 628080: the big windows
MAIN_RECT = (200, 100, 600, 500)

# 627060 and 6279a0: the grid of 4 x 4 cells of 100 x 100 from (200, 100)
CELL_SIZE = 100
CELL_COLUMNS = 4
CELL_ROWS = 4
CELLS = tuple((200 + column * CELL_SIZE, 100 + row * CELL_SIZE,
               200 + (column + 1) * CELL_SIZE, 100 + (row + 1) * CELL_SIZE)
              for row in range(CELL_ROWS) for column in range(CELL_COLUMNS))
TRADE_TEXTURES = ('trade1', 'trade2', 'trade3', 'trade4')
REPAIR_TEXTURES = ('repair1', 'repair2', 'repair3', 'repair4')
CELL_TEXELS = (28, 28, 128, 128)
CELL_FRAME_BORDER = 15

# the remake's button which closes a big window (game.exe closes them by the script of the camp)
CLOSE_RECT = (600, 72, 625, 97)

# 62d0c0: putting the things of the camp on and off; the slots of the paper doll of the remake
SLOT_WEAPON = 'weapon'
SLOT_ORDER = ('helm', 'plate', 'leggins', 'shirt', 'pants', 'boots', 'gloves', SLOT_WEAPON)
SLOT_LABELS = {
    'helm': 'Helm', 'plate': 'Plate', 'leggins': 'Leggins', 'shirt': 'Shirt', 'pants': 'Pants',
    'boots': 'Boots', 'gloves': 'Gloves', SLOT_WEAPON: 'Weapon',
}
ARMOR_SLOTS = {
    idb.ARMOR_HELM: 'helm', idb.ARMOR_PLATE: 'plate', idb.ARMOR_LEGGINS: 'leggins',
    idb.ARMOR_SHIRT: 'shirt', idb.ARMOR_PANTS: 'pants', idb.ARMOR_BOOTS: 'boots',
    idb.ARMOR_GLOVES: 'gloves',
}
# the eight squares of the paper doll: four in a row, two rows under the name of a member
SLOT_SIZE = 44
SLOT_STEP = 48
SLOT_LEFT = 6
SLOT_TOP = 104

# 622d70 and 624ed0: the captions of the texts
INFO_TEXT_KEYS = ('string infounit_3', 'string infounit_4', 'string infounit_9', 'string infounit_14',
                  'string infounit_17')
SKILLS_TEXT_KEYS = ('string skills', 'string perks', 'string perks_av', 'string magic_skills')
TRADE_TEXT_KEYS = ('string camp_item_trade', 'string camp_item_repair', 'string camp_item_constr',
                   'string camp_spell_constr', 'string camp_item_deconstr', 'string camp_spell_deconstr')
TITLE_KEYS = {
    INFO_WINDOW: 'string infounit_3',
    SKILLS_WINDOW: 'string skills',
    TRADE_WINDOW: 'string camp_item_trade',
    REPAIR_WINDOW: 'string camp_item_repair',
    ITEM_CONSTRUCTOR_WINDOW: 'string camp_item_constr',
    SPELL_CONSTRUCTOR_WINDOW: 'string camp_spell_constr',
}

# the remake's lines of the windows whose data game.exe keeps in the party records, not read yet
NOT_MADE_LINE = 'not made yet'

#------------------------------------------------------------------------------


def cell_rect(index):
    return CELLS[index]


def slot_rects(quarter_index):
    """
    The (slot, rect) of the paper doll of a member: four squares in a row and two rows under the name.
    """
    left, top, right, bottom = MEMBER_RECTS[quarter_index]
    result = []
    for index, slot in enumerate(SLOT_ORDER):
        x = left + SLOT_LEFT + (index % 4) * SLOT_STEP
        y = top + SLOT_TOP + (index // 4) * SLOT_STEP
        result.append((slot, (x, y, x + SLOT_SIZE, y + SLOT_SIZE)))
    return result


def slot_at(x, y, quarters=None):
    """
    (quarter index, slot) of the slot under the point, None when there is none.
    """
    for quarter in range(quarters if quarters is not None else len(MEMBER_RECTS)):
        for slot, rect in slot_rects(quarter):
            if _inside(rect, x, y):
                return quarter, slot
    return None


def slot_of(item):
    """
    The slot of an item of engine.equipment: "weapon" or the armor slot, None for an unknown item.
    """
    if item is None:
        return None
    if item.is_weapon:
        return SLOT_WEAPON
    return ARMOR_SLOTS.get(item.slot)


def equipped_items(unit):
    """
    slot -> Item which the unit wears (game.exe CUnitClientSpecific of the unit).
    """
    equipment = getattr(getattr(unit, 'figure', None), 'equipment', None)
    result = {}
    if equipment is None:
        return result
    for slot, item in equipment.armors.items():
        name = ARMOR_SLOTS.get(slot)
        if name is not None:
            result[name] = item
    if equipment.weapon is not None:
        result[SLOT_WEAPON] = equipment.weapon
    return result


def equipped_names(unit):
    return [item.name for item in equipped_items(unit).values()]


def member_at(x, y):
    """
    The index of the quarter under the point, None when there is none.
    """
    for index, rect in enumerate(MEMBER_RECTS):
        if _inside(rect, x, y):
            return index
    return None


def cell_at(x, y):
    """
    The index of the cell of the grid under the point, None when there is none.
    """
    for index, rect in enumerate(CELLS):
        if _inside(rect, x, y):
            return index
    return None


def _inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


def _name_of(item):
    """
    The name of an item of engine.equipment or the name itself.
    """
    return item if isinstance(item, str) else item.name

#------------------------------------------------------------------------------


class Camp(object):
    """
    The state of the camp: the members, the chosen one, the open window, the items of the trade and of the
    repair and the spells of the constructor.
    """

    def __init__(self, units=(), items=(), spells=(), names=(), party=None, repair_items=()):
        self.units = list(units)
        self.names = list(names)
        self.items = list(items)
        self.spells = list(spells)
        # engine.party.Party: the money of the deals
        self.party = party
        self.selected = 0
        self.window = CAMP_WINDOW
        self.trade_items = list(items)
        # the worn things of the members, engine.trade charges their repair
        self.repair_items = list(repair_items)
        # the name of the item of the strip which waits for a slot of the paper doll
        self.picked = None

    def __repr__(self):
        return 'Camp(%d members, %s)' % (len(self.units), self.window)

    @property
    def unit(self):
        if 0 <= self.selected < len(self.units):
            return self.units[self.selected]
        return None

    @property
    def money(self):
        """
        The money of the party (Party.money), 0 without a party.
        """
        return getattr(self.party, 'money', 0)

    def member_count(self):
        return max(len(self.units), len(self.names))

    def member_name(self, index):
        if 0 <= index < len(self.names):
            return self.names[index]
        if 0 <= index < len(self.units) and self.units[index] is not None:
            return self.units[index].figure.model_name
        return ''

    def select(self, index):
        if 0 <= index < max(self.member_count(), 1):
            self.selected = index
            return True
        return False

    def open(self, window):
        if window not in WINDOWS:
            return False
        self.window = window
        return True

    def close_window(self):
        self.window = CAMP_WINDOW

    def click(self, x, y):
        """
        "select", "close", "cell <i>" or None: the member of the quarter, the close button or a cell of the grid.
        """
        if self.window == CAMP_WINDOW:
            # the quarters are under the big windows, they are chosen only on the camp screen
            index = member_at(x, y)
            if index is not None and index < self.member_count():
                self.select(index)
                return 'select'
        else:
            if _inside(CLOSE_RECT, x, y):
                self.close_window()
                return 'close'
            index = cell_at(x, y)
            if index is not None:
                return 'cell %d' % index
        return None

    def window_items(self):
        """
        The items of the open window: the trade, the repair, the constructor of items or the spells.
        """
        if self.window == TRADE_WINDOW:
            return list(self.trade_items)
        if self.window == REPAIR_WINDOW:
            return list(self.repair_items)
        if self.window == ITEM_CONSTRUCTOR_WINDOW:
            return list(self.items)
        return []

    def cell_items(self):
        """
        The texts of the cells of the grid: the names of the items or of the spells, 16 of them at most.
        """
        if self.window == SPELL_CONSTRUCTOR_WINDOW:
            names = [spell.name for spell in self.spells]
        else:
            names = [_name_of(item) for item in self.window_items()]
        return names[:len(CELLS)]

    def stat_lines(self, unit=None, labels=None):
        """
        The lines of the information window: (caption, value) of the attack, the defense, the damage, the
        current weapon and the overload (622d70). labels: text name -> text of texts.res.
        """
        unit = self.unit if unit is None else unit
        labels = labels or {}
        fighter = unit.fighter if unit is not None else None
        if fighter is None:
            return []
        lines = [(labels.get('string infounit_3', 'Attack:'), '%d' % fighter.attack),
                 (labels.get('string infounit_4', 'Defense'), '%d' % fighter.parry),
                 (labels.get('string infounit_9', 'Damage:'),
                  '%d-%d' % (fighter.damage_min, fighter.damage_min + fighter.damage_spread))]
        equipment = getattr(unit.figure, 'equipment', None)
        weapon = equipment.weapon if equipment is not None else None
        if weapon is not None:
            lines.append((labels.get('string infounit_14', 'Current weapon:'), weapon.name))
        return lines
