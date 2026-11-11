"""
The window of the things of a dead unit: the list of engine.loot.Loot, a click on a line takes the thing,
the button takes everything and closes the window, Escape closes it too.

The window is the remake's: game.exe builds the two inventories of the body in FUN_00461cb0 @ 461cb0 and
opens them in the camp-like window of the trade; the corpse has no own window function in the names of the
game (CItemLoot of 6a4960 is a thing lying in the world, not the body). The game stands while the window
of the remake is open.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'loot'

TEXT_COLOR = (228, 215, 167)
TITLE_COLOR = (255, 179, 49)
SELECTED_COLOR = (255, 215, 120)
DIM_COLOR = (150, 145, 115)
MONEY_COLOR = (255, 215, 120)
PANEL_COLOR = (90, 90, 66)
PANEL_HIGH = (132, 130, 90)
SHADE = (0.0, 0.0, 0.0, 0.62)

PANEL_RECT = (250, 100, 550, 500)
TITLE_BOX = (262, 108, 538, 132)
ROWS_TOP = 140
ROW_HEIGHT = 20
VISIBLE_ROWS = 14
ROWS_X = (262, 500)
TAKE_ALL_RECT = (262, 448, 392, 478)
CLOSE_RECT = (410, 448, 538, 478)

TITLE = 'Things of the body'
TAKE_ALL_LABEL = 'Take all'
CLOSE_LABEL = 'Close'
MONEY_LABEL = 'Money: %d'

#------------------------------------------------------------------------------


class LootView(object):

    def __init__(self, app):
        self.app = app
        self.loot = None
        self.shown_key = None

    @property
    def is_open(self):
        return self.loot is not None

    def __repr__(self):
        return 'LootView(%d)' % (len(self.loot.entries) if self.loot is not None else 0)

    def open(self, loot):
        self.loot = loot
        self.shown_key = None
        self.update()

    def close(self):
        self.loot = None
        self.shown_key = None
        self.app.interface.set_layer(LAYER_NAME, [])

    def update(self):
        if self.loot is None:
            return
        key = tuple((entry.kind, entry.name, entry.amount) for entry in self.loot.entries)
        if key == self.shown_key:
            return
        self.shown_key = key
        self.app.interface.set_layer(LAYER_NAME, self.elements())

    def row_rect(self, row):
        top = ROWS_TOP + row * ROW_HEIGHT
        return (ROWS_X[0], top, ROWS_X[1], top + ROW_HEIGHT)

    def elements(self):
        items = [ui.box_item((0, 0, 800, 600), SHADE)]
        items.extend(ui.panel_items(PANEL_RECT, 2, PANEL_COLOR, PANEL_HIGH, background=(0.08, 0.08, 0.07, 0.96)))
        items.append(ui.text_item(TITLE, TITLE_BOX, ui.FONT_MEDIUM, TITLE_COLOR, align='center'))
        entries = self.loot.entries
        for row, entry in enumerate(entries[:VISIBLE_ROWS]):
            rect = self.row_rect(row)
            text = MONEY_LABEL % entry.amount if entry.is_money else entry.name
            color = MONEY_COLOR if entry.is_money else TEXT_COLOR
            items.append(ui.text_item(text, rect, ui.FONT_SMALL, color))
        if len(entries) > VISIBLE_ROWS:
            items.append(ui.text_item('...', self.row_rect(VISIBLE_ROWS - 1), ui.FONT_SMALL, DIM_COLOR))
        if not entries:
            items.append(ui.text_item('nothing', (ROWS_X[0], ROWS_TOP, ROWS_X[1], ROWS_TOP + ROW_HEIGHT),
                                      ui.FONT_SMALL, DIM_COLOR))
        items.append(ui.box_item(TAKE_ALL_RECT, (0.30, 0.26, 0.18, 1.0)))
        items.append(ui.text_item(TAKE_ALL_LABEL, TAKE_ALL_RECT, ui.FONT_SMALL, TEXT_COLOR, align='center'))
        items.append(ui.box_item(CLOSE_RECT, (0.30, 0.26, 0.18, 1.0)))
        items.append(ui.text_item(CLOSE_LABEL, CLOSE_RECT, ui.FONT_SMALL, TEXT_COLOR, align='center'))
        return items

    def on_mouse(self, x, y):
        return self.loot is not None

    def on_click(self, x, y):
        if self.loot is None:
            return False
        party = self.app.party
        if _inside(TAKE_ALL_RECT, x, y):
            self.message_taken(self.loot.take_all(party))
            self.app.rebuild_corpse(self.loot.unit)
            self.close()
            return True
        if _inside(CLOSE_RECT, x, y):
            self.close()
            return True
        for row in range(min(len(self.loot.entries), VISIBLE_ROWS)):
            if _inside(self.row_rect(row), x, y):
                entry = self.loot.entries[row]
                if self.loot.take(row, party):
                    self.message_taken([entry])
                    self.app.rebuild_corpse(self.loot.unit)
                if self.loot.is_empty:
                    self.close()
                else:
                    self.shown_key = None
                    self.update()
                return True
        return True

    def message_taken(self, entries):
        """
        10202 of the text window: the things which came to the party.
        """
        battle = getattr(self.app, 'battle', None)
        if battle is None:
            return
        for entry in entries:
            text = MONEY_LABEL % entry.amount if entry.is_money else entry.name
            battle.add_item_message(text)

    def on_key(self, name):
        if self.loot is None:
            return False
        if name == 'escape':
            self.close()
        elif name in ('enter', 'spacebar'):
            self.message_taken(self.loot.take_all(self.app.party))
            self.app.rebuild_corpse(self.loot.unit)
            self.close()
        return True

#------------------------------------------------------------------------------


def _inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom
