"""
The options window of the main menu (engine.options, game.exe 650380) as a layer of the interface:
the plaque OPTIONS above the window (the parts options and optionslabel of the model of the escape menu),
the frame and the background, the plaques of the groups (651310), the bar of the selected option, the
rows of 651900 with their toggles and the sliders of 65b870, the buttons of the saveload atlas.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

from engine import difficulty as difficulty_module
from engine import figure
from engine import menus
from engine import options as options_module
from engine import prop as prop_module
from engine import tooltip as tooltip_module
from client import gl
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'options'
TIP_LAYER_NAME = 'options_tip'

ATLAS_NAME = 'saveload'

SLIDER_ATLAS_NAME = 'Scrollbar'

# 651310: the texts of the selected group and of the options 0xa7d7e4, the other groups 0xa0a0a0 on their
# plaques at half the brightness; the label of a hotkey which waits for its key 0xff
BRIGHT_COLOR = (228, 215, 167)
DIM_COLOR = (160, 160, 160)
CAPTURE_COLOR = (255, 0, 0)
DIM_PLAQUE = (0.5, 0.5, 0.5, 1.0)
SCROLL_COLOR = (0.35, 0.35, 0.30, 1.0)
THUMB_COLOR = (0.70, 0.68, 0.55, 1.0)

GROUP_FONT = ui.FONT_MEDIUM
OPTION_FONT = ui.FONT_MEDIUM
TITLE_FONT = ui.FONT_LARGE

# the pieces of the saveload atlas (the texels of game.exe upside down, as OK_TEXELS): the strip of the
# frame of 5e7d60 (2, 0xfe)-(0xb4, 0xf9) and the plaque of a group (0x7a, 0x6c)-(0xfc, 0x7f)
FRAME_TEXELS = (2, 2, 0xb4, 7)
GROUP_TEXELS = (0x7a, 148, 0xfc, 129)
# the pieces of Scrollbar (64 x 64; game.exe gives them in 1/256 of the texture, upside down), all of them
# turned by a quarter: the track (8, 4)-(0x20, 0xfc), the knob (0x28, 8)-(0x50, 0x78), the left arrow
# (0x28, 0x80)-(0x50, 0xb0) and the right one with the top and the bottom swapped
TRACK_TEXELS = (2, 63, 8, 1)
KNOB_TEXELS = (10, 62, 20, 34)
LEFT_ARROW_TEXELS = (10, 32, 20, 20)
RIGHT_ARROW_TEXELS = (10, 20, 20, 32)

# 650380: the plaque is the model of the escape menu with two parts, at (400, 480) of the screen and the
# depth 4.8 (6508e0 slides it from 700 when the window opens in the game)
PLAQUE_PARTS = ('options', 'optionslabel')
PLAQUE_SCREEN_POINT = (400.0, 480.0)
PLAQUE_DEPTH = 4.8
PLAQUE_TURN = math.pi * 0.5

CAPTURE_TEXT = 'Press a key for %s'

# the texts of the rows and the title stand a pixel higher than the middle of their boxes in Kivy
# (measured on the frames of plan 3; the names of the groups and the tooltips match without it)
TEXT_RAISE = 1

SOUND_CHANGE = 'buttons/Menu/Ok.wav'
SOUND_CANCEL = 'buttons/MessBox/Cancel.wav'

#------------------------------------------------------------------------------


class OptionsView(object):
    """
    The modal window of the options; the changes are applied by the V button and dropped by X/Escape.
    """

    def __init__(self, app):
        self.app = app
        self.settings = None
        self.dialog = None
        self.is_open = False
        self.plaque = None
        self.plaque_view = None
        self.tooltip = tooltip_module.Tooltip()

    def text(self, key):
        return (self.app.game_data.text(key) or '').replace('\r', '')

    def open(self):
        self.settings = self.app.settings.copy()
        self.dialog = options_module.Dialog(self.settings)
        self.is_open = True
        self.tooltip = tooltip_module.Tooltip()
        self.show_plaque()
        self.update()
        return True

    def show_plaque(self):
        scene = getattr(self.app, 'scene', None)
        if scene is None or self.plaque_view is not None:
            return
        model = figure.FigureModel(self.app.game_data, menus.ESC_MENU_MODEL, parts=PLAQUE_PARTS)
        position = gl.interface_point(PLAQUE_SCREEN_POINT[0], PLAQUE_SCREEN_POINT[1], PLAQUE_DEPTH)
        rotation = (np.cos(PLAQUE_TURN * 0.5), np.sin(PLAQUE_TURN * 0.5), 0.0, 0.0)
        self.plaque = prop_module.Prop(model, prop_module.placement_matrix(position, rotation))
        self.plaque.idle()
        self.plaque_view = scene.add_prop(self.plaque, menus.ESC_MENU_TEXTURE, menus.ESC_MENU_LABELS, interface=True)

    def close(self):
        self.is_open = False
        self.dialog = None
        self.settings = None
        self.app.interface.set_layer(LAYER_NAME, [])
        self.app.interface.set_layer(TIP_LAYER_NAME, [])
        if self.plaque_view is not None:
            self.app.scene.remove_interface_prop(self.plaque_view)
        self.plaque = None
        self.plaque_view = None

    def confirm(self):
        settings = self.settings
        self.close()
        self.app.settings.values = dict(settings.values)
        self.app.settings.file_path = settings.file_path
        self.app.settings.save()
        self.app.settings.apply(self.app)
        if getattr(self.app, 'keys', None) is not None:
            self.app.keys.reload()

    def cancel(self):
        if self.dialog is not None:
            self.dialog.cancel()
        self.close()

    def update(self):
        if not self.is_open:
            return
        dialog = self.dialog
        settings = self.settings
        # no dark layer over the screen: the pause filter of the scene grays the world (P1-T4), the plaque
        # and the menu columns keep their colors
        items = frame_items(options_module.FRAME_RECT, options_module.FRAME_WIDTH)
        items.append(ui.box_item(options_module.PANEL_RECT, options_module.PANEL_COLOR))
        text = self.app.game_data.text
        left, top, right, bottom = options_module.TITLE_BOX
        title_box = (left, top - TEXT_RAISE, right, bottom - TEXT_RAISE)
        items.append(ui.text_item(options_module.group_label(text, dialog.group), title_box,
                                  TITLE_FONT, BRIGHT_COLOR, align='center', valign='middle'))
        for row in range(options_module.VISIBLE_ROWS):
            group_index = dialog.group_scroll + row
            if group_index < len(options_module.GROUPS):
                items.extend(self.group_items(row, group_index))
            option_index = dialog.option_scroll + row
            if option_index < len(dialog.group_keys) and dialog.group_keys[option_index] is not None:
                items.extend(self.option_items(row, option_index))
        items.append(ui.sprite_item(ATLAS_NAME, options_module.OK_RECT, difficulty_module.OK_TEXELS))
        items.append(ui.sprite_item(ATLAS_NAME, options_module.CANCEL_RECT,
                                    difficulty_module.CANCEL_TEXELS))
        items.extend(self.scrollbar_items())
        self.app.interface.set_layer(LAYER_NAME, items)

    def on_mouse(self, x, y):
        """
        The tooltip of the region under the mouse (650380, 651d30) waits for its delay again.
        """
        if not self.is_open:
            return False
        if self.tooltip.hover(self.dialog.tip_at(x, y), x, y):
            self.app.interface.set_layer(TIP_LAYER_NAME, [])
        return True

    def tick(self, dt_ms):
        """
        65e540 (partly): the tooltip comes up when the mouse stood over its region for the delay of the options.
        """
        if not self.is_open:
            return
        slider = self.app.settings.values.get('tooltip_time', options_module.TOOLTIP_TIME)
        if self.tooltip.update(dt_ms, tooltip_module.delay_ms(slider)):
            text = self.app.game_data.text(self.tooltip.tip) or ''
            self.app.interface.set_layer(TIP_LAYER_NAME, ui.tooltip_items(self.app.interface, text,
                                                                          self.tooltip.mouse))

    def group_items(self, row, group_index):
        """
        651310: the plaque of the group, bright when it is selected, and its name.
        """
        rect = self.dialog.group_row_rect(row)
        selected = group_index == self.dialog.group_index
        plaque = (options_module.GROUP_PLAQUE_X[0], rect[1] + options_module.GROUP_PLAQUE_Y[0],
                  options_module.GROUP_PLAQUE_X[1], rect[1] + options_module.GROUP_PLAQUE_Y[1])
        group = options_module.GROUPS[group_index][0]
        return [
            ui.sprite_item(ATLAS_NAME, plaque, GROUP_TEXELS, None if selected else DIM_PLAQUE),
            ui.text_item(options_module.group_label(self.app.game_data.text, group), plaque, GROUP_FONT,
                         BRIGHT_COLOR if selected else DIM_COLOR, align='center', valign='middle'),
        ]

    def option_items(self, row, option_index):
        """
        651900: the bar of the selected row, the label, and the value as a toggle text or a slider.
        """
        dialog = self.dialog
        key = dialog.group_keys[option_index]
        rect = dialog.option_row_rect(row)
        selected = option_index == dialog.option_index
        items = []
        if selected:
            items.append(ui.box_item(rect, options_module.SELECTION_COLOR))
        text = self.app.game_data.text
        color = CAPTURE_COLOR if selected and dialog.capture else BRIGHT_COLOR
        items.append(ui.text_item(options_module.option_label(text, key),
                                  (options_module.LABEL_X, rect[1] - TEXT_RAISE, options_module.VALUE_X - 5,
                                   rect[3] - TEXT_RAISE),
                                  OPTION_FONT, color, valign='middle'))
        kind, _default, _low, high, _choices = options_module.OPTIONS[key]
        value = self.settings.value(key)
        if kind == options_module.SLIDER:
            items.extend(slider_items(rect[1], value, high))
        else:
            items.append(ui.text_item(options_module.value_text(text, key, value),
                                      (options_module.VALUE_X, rect[1] - TEXT_RAISE, rect[2], rect[3] - TEXT_RAISE),
                                      OPTION_FONT, BRIGHT_COLOR, valign='middle'))
        return items

    def scrollbar_items(self):
        """
        The strip of the options at the right edge: the track and the thumb (not from the atlas yet).
        """
        dialog = self.dialog
        left, top, right, bottom = options_module.SCROLLBAR_RECT
        count = len(dialog.group_keys)
        if count <= options_module.VISIBLE_ROWS:
            return []
        items = [ui.box_item((left, top, right, bottom), SCROLL_COLOR)]
        height = bottom - top
        thumb = max(int(height * options_module.VISIBLE_ROWS / count), 12)
        travel = height - thumb
        most = count - options_module.VISIBLE_ROWS
        y = top + int(travel * dialog.option_scroll / most) if most else top
        items.append(ui.box_item((left, y, right, y + thumb), THUMB_COLOR))
        return items

    def on_click(self, x, y):
        """
        True while the window is open: it takes every click.
        """
        if not self.is_open:
            return False
        action = self.dialog.click(x, y)
        if action == 'confirm':
            self.app.ui_sound(SOUND_CHANGE)
            self.confirm()
        elif action == 'cancel':
            self.app.ui_sound(SOUND_CANCEL)
            self.cancel()
        else:
            if action in ('change', 'capture'):
                self.app.ui_sound(SOUND_CHANGE)
            self.update()
        return True

    def on_key(self, name):
        """
        True while the window is open: it takes every key.
        """
        if not self.is_open:
            return False
        action = self.dialog.on_key(name)
        if action == 'confirm':
            self.app.ui_sound(SOUND_CHANGE)
            self.confirm()
        elif action == 'cancel':
            self.app.ui_sound(SOUND_CANCEL)
            self.cancel()
        else:
            if action in ('change', 'capture'):
                self.app.ui_sound(SOUND_CHANGE)
            self.update()
        return True

#------------------------------------------------------------------------------


def frame_items(rect, width):
    """
    5e7d60 (partly): the frame of the strip of the saveload atlas, its first row outside, as four pieces.
    """
    x0, y0, x1, y1 = rect
    u0, v0, u1, v1 = FRAME_TEXELS
    return [
        ui.sprite_item(ATLAS_NAME, (x0, y0, x1, y0 + width), (u0, v0, u1, v1)),
        ui.sprite_item(ATLAS_NAME, (x0, y1 - width, x1, y1), (u0, v1, u1, v0)),
        ui.sprite_item(ATLAS_NAME, (x0, y0 + width, x0 + width, y1 - width), (u0, v1, u1, v0), turn=True),
        ui.sprite_item(ATLAS_NAME, (x1 - width, y0 + width, x1, y1 - width), (u0, v0, u1, v1), turn=True),
    ]


def slider_items(top, value, high):
    """
    65b870: the arrows, the track and the knob of the slider of the row at top, all turned by a quarter.
    """
    left, track, knob, right = options_module.slider_rects(top, value, high)
    return [
        ui.sprite_item(SLIDER_ATLAS_NAME, track, TRACK_TEXELS, turn=True),
        ui.sprite_item(SLIDER_ATLAS_NAME, left, LEFT_ARROW_TEXELS, turn=True),
        ui.sprite_item(SLIDER_ATLAS_NAME, right, RIGHT_ARROW_TEXELS, turn=True),
        ui.sprite_item(SLIDER_ATLAS_NAME, knob, KNOB_TEXELS, turn=True),
    ]
