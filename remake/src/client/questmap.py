"""
The window of the map of an allod and of the journal of its quests (game.exe 64cbd0 and 64dc20): the map
texture "<allod>map" with the markers of the quests, the list of the quests and the text of the chosen one.

The markers of the remake are squares: green for a completed quest, red for the others. A click on a marker
chooses its quest; the click on a row of the list does too. The window is opened by the button of the quests
of the disk of the pause (game.exe opens it by the script of the game, 64cbd0 is called from 6488a0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import zones as zones_module
from engine import questmap as questmap_module
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'quest map'

TEXT_COLOR = (228, 215, 167)
TITLE_COLOR = (255, 179, 49)
SELECTED_COLOR = (255, 215, 120)
DONE_COLOR = (110, 200, 110)
OPEN_COLOR = (220, 90, 70)
SHADE = (0.0, 0.0, 0.0, 0.62)
PANEL_BACKGROUND = (0.08, 0.08, 0.07, 0.96)
PANEL_COLOR = (90, 90, 66)
PANEL_HIGH = (132, 130, 90)

MARKER_SIZE = 5

# the sounds of the map of the allod (buttons/Globalmap/... of sfx.res)
SOUND_SELECT = 'buttons/Globalmap/Zone.wav'
SOUND_CLOSE = 'buttons/Globalmap/Ok.wav'

#------------------------------------------------------------------------------


class QuestMapView(object):

    def __init__(self, app):
        self.app = app
        self.model = None
        self.page = 0
        self.shown_key = None

    @property
    def is_open(self):
        return self.model is not None

    def __repr__(self):
        return 'QuestMapView(%s)' % (self.model.allod if self.model is not None else 'closed')

    def open(self, allod=None):
        app = self.app
        if allod is None:
            zone = app.game_data.zone_of_map(app.map_name) if app.map_name else None
            allod = zone.allod if zone is not None else zones_module.DEFAULT_ALLOD
        completed = list(app.party.completed_quests) if getattr(app, 'party', None) is not None else []
        self.model = questmap_module.QuestMap.from_game(app.game_data, allod, completed)
        self.model.select(0)
        self.shown_key = None
        self.update()

    def close(self):
        self.model = None
        self.shown_key = None
        self.app.interface.set_layer(LAYER_NAME, [])

    def update(self):
        if self.model is None:
            return
        key = (self.model.allod, self.model.selected, self.model.scroll, tuple(self.model.completed))
        if key == self.shown_key:
            return
        self.shown_key = key
        self.app.interface.set_layer(LAYER_NAME, self.elements())

    def elements(self):
        model = self.model
        items = [ui.box_item((0, 0, 800, 600), SHADE)]
        items.append(ui.sprite_item(model.allod + 'map', questmap_module.MAP_RECT))
        items.extend(self.marker_items())
        items.extend(self.list_items())
        items.extend(self.details_items())
        rect = questmap_module.TITLE_BOX
        items.append(ui.text_item(questmap_module.TITLE_TEXT, rect, ui.FONT_LARGE, TITLE_COLOR))
        close = questmap_module.CLOSE_RECT
        items.append(ui.box_item(close, (0.35, 0.30, 0.20, 1.0)))
        items.append(ui.text_item('X', (close[0] + 6, close[1] + 2, close[2], close[3]), ui.FONT_MEDIUM,
                                  TEXT_COLOR))
        return items

    def marker_items(self):
        items = []
        model = self.model
        selected = model.selected_quest
        for quest in model.map_quests():
            point = model.map_point(quest)
            if point is None:
                continue
            size = MARKER_SIZE + (2 if quest is selected else 0)
            color = DONE_COLOR if model.is_completed(quest) else OPEN_COLOR
            rgba = tuple(channel / 255.0 for channel in color) + (1.0,)
            rect = (point[0] - size, point[1] - size, point[0] + size, point[1] + size)
            items.append(ui.box_item(rect, rgba))
        if selected is not None:
            point = model.map_point(selected)
            if point is not None:
                items.append(ui.text_item(selected.name, (point[0] + 10, point[1] - 8, point[0] + 90,
                                                          point[1] + 10), ui.FONT_SMALL, TEXT_COLOR))
        return items

    def list_items(self):
        rect = questmap_module.LIST_RECT
        items = ui.panel_items(rect, 2, PANEL_COLOR, PANEL_HIGH, background=PANEL_BACKGROUND)
        model = self.model
        entries = model.entries
        for row in range(questmap_module.LIST_VISIBLE_ROWS):
            index = model.scroll + row
            if index >= len(entries):
                break
            quest = entries[index]
            row_rect = model.row_rect(row)
            selected = index == model.selected
            name = quest.name
            if model.is_completed(quest):
                name = quest.name + ' *'
            items.append(ui.text_item(name, row_rect, ui.FONT_SMALL,
                                      SELECTED_COLOR if selected else TEXT_COLOR))
        return items

    def details_items(self):
        rect = questmap_module.DETAILS_RECT
        items = ui.panel_items(rect, 2, PANEL_COLOR, PANEL_HIGH, background=PANEL_BACKGROUND)
        model = self.model
        quest = model.selected_quest
        if quest is None:
            return items
        title, lines = model.description()
        title = title or quest.name
        items.append(ui.text_item(title, (rect[0] + 8, rect[1] + 8, rect[2] - 8, rect[1] + 30), ui.FONT_MEDIUM,
                                  TITLE_COLOR))
        text = '\n'.join(lines)
        items.append(ui.text_item(text, (rect[0] + 8, rect[1] + 34, rect[2] - 8, rect[3] - 8), ui.FONT_SMALL,
                                  TEXT_COLOR))
        return items

    def on_mouse(self, x, y):
        return self.model is not None

    def on_click(self, x, y):
        if self.model is None:
            return False
        action = self.model.click(x, y)
        if action == 'close':
            self.app.ui_sound(SOUND_CLOSE)
            self.close()
        else:
            self.app.ui_sound(SOUND_SELECT)
            self.update()
        return True

    def on_key(self, name):
        if self.model is None:
            return False
        action = self.model.on_key(name)
        if action == 'close':
            self.app.ui_sound(SOUND_CLOSE)
            self.close()
        else:
            self.app.ui_sound(SOUND_SELECT)
            self.update()
        return True
