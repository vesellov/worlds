"""
The window of a conversation with a peaceful unit: the topics of engine.talk in five rows, the chosen one
opens the briefing window, "goodbye" and Escape close the conversation.

The window of game.exe (FUN_006071a0 @ 6071a0) sits at (195, 406): the topics at (210, 440) in rows of
24 pixels, five at once, and the row "goodbye" under them (FUN_00608b20 @ 608b20). The remake draws the
same rows over a panel of the same place; the title of a row is the title of its briefing text.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import talk as talk_module
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'talk'

TEXT_COLOR = (228, 215, 167)
TITLE_COLOR = (255, 179, 49)
SELECTED_COLOR = (255, 215, 120)
DIM_COLOR = (150, 145, 115)
PANEL_COLOR = (90, 90, 66)
PANEL_HIGH = (132, 130, 90)
SHADE = (0.0, 0.0, 0.0, 0.45)

# 6071a0: the rows are relative to the corner of the window at (195, 406)
ORIGIN = (195, 406)
PANEL_RECT = (195, 398, 600, 600)
TITLE_BOX = (205, 402, 590, 426)

#------------------------------------------------------------------------------


def owner_name_of(app, unit):
    """
    The OBJNAME of the unit in the scripts (the owner of its topics "b.<owner>.<name>").
    """
    host = getattr(app, 'script_host', None)
    for name, obj in getattr(host, 'objects_by_name', {}).items():
        if obj is unit:
            return name
    return getattr(unit, 'name', '') or ''


def topics_of(app, unit):
    """
    The topics of a unit which the game variables "b.<owner>.<name>" open.
    """
    game_vars = getattr(getattr(app, 'scripts', None), 'game_vars', {})
    return talk_module.topics(game_vars, owner_name_of(app, unit))


def has_topics(app, unit):
    return bool(topics_of(app, unit))

#------------------------------------------------------------------------------


class TalkView(object):

    def __init__(self, app):
        self.app = app
        self.talk = None
        self.shown_key = None
        # called with the chosen topic when the player takes one
        self.on_topic = None

    @property
    def is_open(self):
        return self.talk is not None

    def __repr__(self):
        return 'TalkView(%s)' % (self.talk.owner_name if self.talk is not None else 'closed')

    def open(self, unit):
        """
        The topics of the owner of the unit: the variables "b.<owner>.<name>" with the value 1.
        """
        owner_name = self.owner_name_of(unit)
        topics = topics_of(self.app, unit)
        if not topics:
            return False
        text_of = self.app.game_data.text
        for topic in topics:
            talk_module.title_of(topic, text_of)
        self.talk = talk_module.Talk(owner_name, topics)
        self.shown_key = None
        self.update()
        return True

    def owner_name_of(self, unit):
        return owner_name_of(self.app, unit)

    def close(self):
        self.talk = None
        self.shown_key = None
        self.app.interface.set_layer(LAYER_NAME, [])

    def update(self):
        if self.talk is None:
            return
        key = (self.talk.owner_name, self.talk.selected, self.talk.scroll, tuple(
            (topic.name, topic.title) for topic in self.talk.topics))
        if key == self.shown_key:
            return
        self.shown_key = key
        self.app.interface.set_layer(LAYER_NAME, self.elements())

    def row_rect(self, row):
        left, top, right, bottom = self.talk.row_rect(row)
        return (left + ORIGIN[0], top + ORIGIN[1], right + ORIGIN[0], bottom + ORIGIN[1])

    def goodbye_rect(self):
        return self.row_rect(talk_module.GOODBYE_ROW)

    def elements(self):
        talk = self.talk
        items = [ui.box_item((0, 0, 800, 600), SHADE)]
        items.extend(ui.panel_items(PANEL_RECT, 2, PANEL_COLOR, PANEL_HIGH, background=(0.08, 0.08, 0.07, 0.96)))
        title = talk_module.TITLE_FORMAT % talk.owner_name
        items.append(ui.text_item(title, TITLE_BOX, ui.FONT_MEDIUM, TITLE_COLOR))
        for row in range(talk_module.VISIBLE_ROWS):
            index = talk.scroll + row
            if index >= len(talk.topics):
                break
            topic = talk.topics[index]
            color = SELECTED_COLOR if index == talk.selected else TEXT_COLOR
            items.append(ui.text_item(topic.title, self.row_rect(row), ui.FONT_SMALL, color))
        items.append(ui.text_item(talk_module.GOODBYE_TEXT, self.goodbye_rect(), ui.FONT_SMALL, DIM_COLOR))
        return items

    def on_mouse(self, x, y):
        return self.talk is not None

    def on_click(self, x, y):
        if self.talk is None:
            return False
        action = self.talk.click(x - ORIGIN[0], y - ORIGIN[1])
        if action == 'goodbye':
            self.close()
        elif action == 'topic':
            topic = self.talk.selected_topic
            callback = self.on_topic
            self.close()
            if callback is not None and topic is not None:
                callback(topic)
        else:
            self.update()
        return True

    def on_key(self, name):
        if self.talk is None:
            return False
        action = self.talk.on_key(name)
        if action == 'goodbye':
            self.close()
        elif action == 'topic':
            topic = self.talk.selected_topic
            callback = self.on_topic
            self.close()
            if callback is not None and topic is not None:
                callback(topic)
        else:
            self.update()
        return True
