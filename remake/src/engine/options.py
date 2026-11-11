"""
The options of the game: the records of the table at 787434 (26 options) and the groups of 79479c
(video, sound, sens, game, select, actions, camera, items, spells, cshots, windows, debug1, debug2).

This module makes the settings which the remake can use: the volumes, the difficulty, the brightness of
the video, the toggles of the picture and the hotkeys. The settings live in a file of the remake
(settings.json of ~/.cache/ei-remake by default); game.exe keeps them in the registry.

The records of the options at 787430 are (id, key, kind, maximum, group, row): the kind 0 is a slider
of 0..maximum, the kind 1 a toggle of two values (the difficulty is one too, Novice/Normal). The records
of the hotkeys at 787080 are (id, action, group, row). 651900 (partly) and 6515c0 put each record on its row of
its group, so the rows which no record takes stay empty: GROUPS keeps them as None.

The video settings of game.exe (P2): Brightness, Contrast and Gamma are floats of the registry in
[-1, 1] (0 is the middle of the slider), loaded into the fields +0x8c/+0x90/+0x94 of the settings
object 79bc60 (459100). FUN_00504a60 builds the table of the gamma control from them: a ramp of
3 x 256 words, the same value on the three channels. The brightness is halved, the contrast gives
the scale 1 + 4 * |c| (1 / scale when c is negative) and the offset (1 - scale) / 2, the gamma gives
the exponent 1 / (1 + 2.5 g) for g > 0 and 1 / (1 + 0.25 g) for g < 0; the ramp value is
((i / 255) ** exponent * scale + offset + brightness) clamped to 0..1. The table goes to the gamma
control of the DirectDraw surface (a virtual call +0x10 of IDirectDrawGammaControl), which works in
the fullscreen mode only: under Wine on Xvfb the X gamma stays 1.0 even at the maximum, so the
picture of neither game changes there (P2-T2). The remake keeps the values and waits for P28.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import json
import os

#------------------------------------------------------------------------------

from eilib import keys as keys_module
from engine import difficulty

#------------------------------------------------------------------------------

TOGGLE = 'toggle'
SLIDER = 'slider'
CHOICE = 'choice'
KEY = 'key'

DEFAULT_DIR_PATH = os.path.join(os.path.expanduser('~'), '.cache', 'ei-remake')
SETTINGS_FILE_NAME = 'settings.json'

ON_KEY = 'string option_on'
OFF_KEY = 'string option_off'
GROUP_KEY = 'string option_group_%s'
OPTION_KEY = 'string option_%s'
ACTION_KEY = 'string action_%s'
TIP_KEY = 'tip option_%s'
GROUP_TIP_KEY = 'tip option_group_%s'
ACTION_TIP_KEY = 'tip action_%s'

# the default of the slider of the tooltip time: 0.5 s
TOOLTIP_TIME = 2

# 652460: CameraDefaultScrollPowerKeyboard and ...Mouse are the sliders times 0.02
POWER_STEP = 0.02

# key -> (the kind of the value, the default, the smallest, the largest, the names of the choices);
# the defaults are the values of game.exe on clean settings (the registry it writes, read by 6521c0)
OPTIONS = {
    # 652460: Brightness, Contrast and Gamma of the registry are slider * 0.02 - 1
    'gamma': (SLIDER, 50, 0, 100, None),
    'contrast': (SLIDER, 50, 0, 100, None),
    'brightness': (SLIDER, 50, 0, 100, None),
    # P3-D9: character shadows on, the shadows of objects and trees off, the marker of the selected
    # character off, the hit result on
    'shadow_flora': (TOGGLE, 0, 0, 1, None),
    'shadow_buildings': (TOGGLE, 0, 0, 1, None),
    'shadow_units': (TOGGLE, 1, 0, 1, None),
    'marks': (TOGGLE, 1, 0, 1, None),
    'footprints': (TOGGLE, 1, 0, 1, None),
    'show_path': (TOGGLE, 1, 0, 1, None),
    'switch_filters': (TOGGLE, 1, 0, 1, None),
    'volume_sfx': (SLIDER, 100, 0, 100, None),
    'volume_stream': (SLIDER, 100, 0, 100, None),
    'volume_voice': (SLIDER, 100, 0, 100, None),
    'reverse_stereo': (TOGGLE, 1, 0, 1, None),
    # the remake's option: the movies of game.exe play in a window at their size (mode 1 of 5db3f0)
    'movie_window': (TOGGLE, 0, 0, 1, None),
    # 652460: CameraDefaultScrollPowerKeyboard / Mouse are slider * 0.02 (0.5 and 1.0 by default)
    'power_kbd': (SLIDER, 25, 0, 100, None),
    'power_mouse': (SLIDER, 50, 0, 100, None),
    # CameraBorderScrollArea (5 by default) and CameraFrameSelectionSensetiveArea / 10 (20 by default)
    'scroll_border': (SLIDER, 5, 0, 10, None),
    'rubber_select': (SLIDER, 2, 0, 10, None),
    'difficulty': (CHOICE, difficulty.NORMAL, difficulty.NOVICE, difficulty.NORMAL,
                   (difficulty.NOVICE, difficulty.NORMAL)),
    'autosave': (TOGGLE, 1, 0, 1, None),
    'show_tutorial': (TOGGLE, 1, 0, 1, None),
    'show_flying_hp': (TOGGLE, 1, 0, 1, None),
    'select_type': (TOGGLE, 0, 0, 1, None),
    # CameraFlipXAxis / CameraFlipYAxis: 0 on clean settings, the option is Off
    'camera_reverse_x': (TOGGLE, 0, 0, 1, None),
    'camera_reverse_y': (TOGGLE, 0, 0, 1, None),
    # 6522c7: the slider is ToolTip Time of the registry (0.5 s on clean settings, 74079) times 4 (73baa0)
    'tooltip_time': (SLIDER, TOOLTIP_TIME, 0, 20, None),
}

# the hotkeys of config/keyboard.ini: the value is the name of the key of Windows
for _action in keys_module.ACTIONS:
    OPTIONS['key_%s' % _action] = (KEY, keys_module.DEFAULTS[_action], 0, 0, None)

# the groups of 79479c and their rows: the options of 787430 and the hotkeys of 787080 with their
# group and row numbers; None is a row which game.exe leaves empty (a gap in the numbers)
GROUPS = (
    ('video', ('shadow_units', 'shadow_buildings', 'shadow_flora', None, 'marks', 'footprints',
               'select_type', 'show_path', 'show_flying_hp', None, 'brightness', 'contrast', 'gamma')),
    # movie_window is the remake's own row after the rows of game.exe
    ('sound', ('volume_sfx', 'volume_stream', 'volume_voice', None, 'reverse_stereo', 'movie_window')),
    ('sens', ('power_kbd', 'power_mouse', 'scroll_border', 'rubber_select', 'tooltip_time', None,
              'camera_reverse_x', 'camera_reverse_y')),
    ('game', ('difficulty', None, 'key_accel', 'key_decel', 'key_pause', None, 'key_obj', None,
              'key_quicksave', 'key_quickload', None, 'autosave', 'switch_filters')),
    ('select', ('key_select1', 'key_select2', 'key_select3', None, 'key_select_all')),
    ('actions', ('key_follow', 'key_use_science', 'key_swarm', None, 'key_run', 'key_walk', 'key_sneak',
                 'key_crawl')),
    ('camera', ('key_camera_track', 'key_camera_up', 'key_camera_down', 'key_camera_left',
                'key_camera_right', 'key_camera_zoom_in', 'key_camera_zoom_out', None, 'key_camera_norm',
                'key_camera1', 'key_camera2', 'key_camera3', 'key_camera4')),
    ('items', ('key_weapon1', 'key_weapon2', 'key_weapon3', 'key_weapon4', None, 'key_item1', 'key_item2',
               'key_item3', 'key_item4')),
    ('spells', tuple('key_spell%d' % number for number in range(1, 9))),
    ('cshots', ('key_cs_head', 'key_cs_body', 'key_cs_rleg', 'key_cs_lleg', 'key_cs_rhand', 'key_cs_lhand')),
    ('windows', ('key_w_text1', 'key_w_text2', None, 'key_w_info1', 'key_w_info2', 'key_w_info3',
                 'key_w_info4', None, 'key_w_minimap', None, 'key_tutorial_script', 'show_tutorial')),
)

# the options which changed their meaning: the values of an older file of the remake are dropped
SETTINGS_VERSION = 2
CHANGED_KEYS = ('camera_reverse_x', 'camera_reverse_y', 'scroll_border', 'rubber_select', 'power_kbd',
                'power_mouse')

# the window of 650380 on the screen of 800 x 600: 651900 (partly) puts the value of the row n at
# (530, 160 + 24 n) - (680, 176 + 24 n), a slider at (530, 163 + 24 n) - (680, 173 + 24 n)
# 650380: the frame of 5e7d60 of 5 pixels around the window, its background of black with the alpha
# 0xa0; 651310: the title (the name of the group) in the box of 10..34 of the window
FRAME_RECT = (95, 95, 705, 505)
FRAME_WIDTH = 5
PANEL_RECT = (100, 100, 700, 500)
PANEL_COLOR = (0.0, 0.0, 0.0, 0xa0 / 255.0)
TITLE_BOX = (100, 110, 700, 134)
ROWS_TOP = 156
ROW_HEIGHT = 24
VISIBLE_ROWS = 13
# the regions of the group buttons and their plaques of the saveload atlas at (110, top + 2)-(240, top + 21)
GROUP_ROWS_X = (100, 250)
GROUP_PLAQUE_X = (110, 240)
GROUP_PLAQUE_Y = (2, 21)
OPTION_ROWS_X = (250, 692)
LABEL_X = 260
VALUE_X = 530
# 651310: the bar of the selected option, 0xc8a06800
SELECTION_COLOR = (0xa0 / 255.0, 0x68 / 255.0, 0.0, 0xc8 / 255.0)
# 65b870 and 65b6f0: a slider of (530, top + 7)-(680, top + 17); the arrows of 12 pixels at its ends,
# the track of 6 pixels between them and the knob of 28 x 10 which goes from x 556 to 654
SLIDER_X = (530, 680)
SLIDER_Y = (7, 17)
SLIDER_ARROW = 12
SLIDER_TRACK_INSET = 4
SLIDER_TRACK_HEIGHT = 6
SLIDER_KNOB = (28, 10)
SLIDER_KNOB_INSET = 26
OK_RECT = (213, 526, 287, 574)
CANCEL_RECT = (513, 526, 587, 574)
SCROLLBAR_RECT = (694, ROWS_TOP, 702, ROWS_TOP + VISIBLE_ROWS * ROW_HEIGHT)

#------------------------------------------------------------------------------


def option_definition(key):
    return OPTIONS[key]


# the labels of the options which texts.res does not have (the remake's ones)
REMAKE_LABELS = {
    'movie_window': 'Movies in a window',
}

#------------------------------------------------------------------------------


def hotkey_label(action):
    words = action.replace('_', ' ')
    return words[:1].upper() + words[1:]


def option_label(text, key):
    """
    651900 (partly) and 6515c0: "string option_<key>" of texts.res, "string action_<name>" for a hotkey.
    """
    if key.startswith('key_'):
        label = text(ACTION_KEY % key[4:]) or hotkey_label(key[4:])
    else:
        label = text(OPTION_KEY % key) or REMAKE_LABELS.get(key)
    return (label or key).replace('\r', '').strip()


def group_label(text, group):
    return (text(GROUP_KEY % group) or group.capitalize()).replace('\r', '').strip()


def option_tip(text, key):
    return (text(TIP_KEY % key) or '').replace('\r', '').strip()


def value_text(text, key, value):
    """
    The text of the value: On/Off, a number or the name of the chosen level.
    """
    kind, _default, low, high, choices = OPTIONS[key]
    if kind == TOGGLE:
        return (text(ON_KEY if value else OFF_KEY) or ('On' if value else 'Off')).replace('\r', '').strip()
    if kind == CHOICE:
        return difficulty.level_label(text, int(value))
    if kind == KEY:
        return keys_module.display_name(str(value))
    return '%d' % int(value)


def keyboard_power(values):
    """
    652460: CameraDefaultScrollPowerKeyboard of the slider "Keyboard sensitivity" (0.5 by default).
    """
    return float(values.get('power_kbd', OPTIONS['power_kbd'][1])) * POWER_STEP


def slider_rects(top, value, high):
    """
    65b870 and 65b6f0: the rects (left arrow, track, knob, right arrow) of the slider of the row at top.
    """
    x0, x1 = SLIDER_X
    y0, y1 = top + SLIDER_Y[0], top + SLIDER_Y[1]
    middle = y1 - 5
    track = (x0 + SLIDER_TRACK_INSET, middle - SLIDER_TRACK_HEIGHT // 2, x1 - SLIDER_TRACK_INSET,
             middle + SLIDER_TRACK_HEIGHT // 2)
    low_x, high_x = x0 + SLIDER_KNOB_INSET, x1 - SLIDER_KNOB_INSET
    center = low_x + int((high_x - low_x) * value / high) if high else low_x
    knob = (center - SLIDER_KNOB[0] // 2, y0, center + SLIDER_KNOB[0] // 2, y0 + SLIDER_KNOB[1])
    return (x0, y0, x0 + SLIDER_ARROW, y1), track, knob, (x1 - SLIDER_ARROW, y0, x1, y1)


def slider_value(x, high):
    """
    The value of a click on the track: the knob goes to the point (65b6f0 the other way round).
    """
    low_x, high_x = SLIDER_X[0] + SLIDER_KNOB_INSET, SLIDER_X[1] - SLIDER_KNOB_INSET
    return min(max(int(round((x - low_x) * high / float(high_x - low_x))), 0), high)


class Settings(object):
    """
    The values of the options and their file.
    """

    def __init__(self, values=None, file_path=None):
        self.values = dict((key, definition[1]) for key, definition in OPTIONS.items())
        if values:
            for key, value in values.items():
                if key in self.values:
                    self.values[key] = value
        # the options which the file of the settings has (the hotkeys of keyboard.ini fill the others)
        self.explicit = set(key for key in values if key in self.values) if values else set()
        self.file_path = file_path

    def __repr__(self):
        return 'Settings(%d values)' % len(self.values)

    def copy(self):
        settings = Settings(dict(self.values), self.file_path)
        settings.explicit = set(self.explicit)
        return settings

    def value(self, key):
        return self.values[key]

    def set_value(self, key, value):
        kind, _default, low, high, _choices = OPTIONS[key]
        if kind == KEY:
            self.values[key] = str(value)
            return self.values[key]
        value = int(value)
        self.values[key] = min(max(value, low), high)
        return self.values[key]

    def change(self, key, step):
        kind, _default, low, high, choices = OPTIONS[key]
        value = self.values[key]
        if kind == KEY:
            return value
        if kind == TOGGLE:
            value = 0 if value else 1
        elif kind == CHOICE:
            values = choices
            index = values.index(int(value)) if int(value) in values else 0
            value = values[(index + (1 if step > 0 else -1)) % len(values)]
        else:
            value = min(max(value + step, low), high)
        self.values[key] = value
        return value

    def apply(self, app):
        """
        The volumes and the difficulty of the game; the video settings wait for the renderer.
        """
        audio = getattr(app, 'audio', None)
        if audio is not None:
            audio.volume_sfx = self.values['volume_sfx'] / 100.0
            audio.volume_voice = self.values['volume_voice'] / 100.0
        music = getattr(app, 'music', None)
        if music is not None:
            music.set_volume(self.values['volume_stream'] / 100.0)
        app.difficulty = int(self.values['difficulty'])

    @classmethod
    def load(cls, file_path=None):
        if file_path is None:
            file_path = os.path.join(DEFAULT_DIR_PATH, SETTINGS_FILE_NAME)
        values = None
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    values = json.load(f)
            except (OSError, ValueError):
                values = None
        if isinstance(values, dict) and values.get('version', 1) < SETTINGS_VERSION:
            for key in CHANGED_KEYS:
                values.pop(key, None)
        if not isinstance(values, dict):
            values = None
        return cls(values, file_path)

    def save(self):
        if not self.file_path:
            return False
        dir_path = os.path.dirname(self.file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        values = dict(self.values)
        values['version'] = SETTINGS_VERSION
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(values, f, indent=1, sort_keys=True)
        return True

#------------------------------------------------------------------------------


class Dialog(object):
    """
    The groups, the options and the clicks of the window of 650380.
    """

    def __init__(self, settings):
        self.settings = settings
        self.original = dict(settings.values)
        self.group_index = 0
        # the row of the selected option in the group; the empty rows are never selected
        self.option_index = 0
        self.group_scroll = 0
        self.option_scroll = 0
        # the next key becomes the key of the selected hotkey
        self.capture = False

    def __repr__(self):
        return 'OptionsDialog(%s, %s)' % (self.group, self.option_key)

    @property
    def group(self):
        return GROUPS[self.group_index][0]

    @property
    def group_keys(self):
        return GROUPS[self.group_index][1]

    @property
    def option_key(self):
        return self.group_keys[self.option_index]

    def select_group(self, index):
        index = min(max(index, 0), len(GROUPS) - 1)
        self.group_index = index
        self.option_index = 0
        self.option_scroll = 0
        self.select_option(0)

    def select_option(self, index, step=1):
        """
        The row of the index, or the next one with an option in the direction of the step.
        """
        rows = self.group_keys
        index = min(max(index, 0), len(rows) - 1)
        while rows[index] is None:
            index += step
            if index < 0 or index >= len(rows):
                return
        self.option_index = index
        row = index
        if row < self.option_scroll:
            self.option_scroll = row
        elif row >= self.option_scroll + VISIBLE_ROWS:
            self.option_scroll = row - VISIBLE_ROWS + 1

    def change(self, step):
        return self.settings.change(self.option_key, step)

    def confirm(self):
        self.settings.save()

    def cancel(self):
        self.settings.values = dict(self.original)

    def group_row_rect(self, row):
        top = ROWS_TOP + row * ROW_HEIGHT
        return (GROUP_ROWS_X[0], top, GROUP_ROWS_X[1], top + ROW_HEIGHT)

    def option_row_rect(self, row):
        top = ROWS_TOP + row * ROW_HEIGHT
        return (OPTION_ROWS_X[0], top, OPTION_ROWS_X[1], top + ROW_HEIGHT)

    def click(self, x, y):
        """
        "group", "option", "change", "confirm", "cancel" or None.
        """
        for row in range(VISIBLE_ROWS):
            if _inside(self.group_row_rect(row), x, y):
                index = self.group_scroll + row
                if index < len(GROUPS):
                    self.select_group(index)
                    return 'group'
            if _inside(self.option_row_rect(row), x, y):
                index = self.option_scroll + row
                if index < len(self.group_keys) and self.group_keys[index] is not None:
                    return self.click_option(index, row, x)
        if _inside(OK_RECT, x, y):
            return 'confirm'
        if _inside(CANCEL_RECT, x, y):
            return 'cancel'
        return None

    def tip_at(self, x, y):
        """
        650380 and 651d30: the name of the tooltip of the region under the mouse, "tip option_group_<group>"
        over a group, "tip option_<key>" or "tip action_<action>" over an option, None elsewhere.
        """
        for row in range(VISIBLE_ROWS):
            if _inside(self.group_row_rect(row), x, y):
                index = self.group_scroll + row
                if index < len(GROUPS):
                    return GROUP_TIP_KEY % GROUPS[index][0]
                return None
            if _inside(self.option_row_rect(row), x, y):
                index = self.option_scroll + row
                if index >= len(self.group_keys) or self.group_keys[index] is None:
                    return None
                key = self.group_keys[index]
                if key.startswith('key_'):
                    return ACTION_TIP_KEY % key[4:]
                return TIP_KEY % key
        return None

    def click_option(self, index, row, x):
        """
        A click on a row: the slider and the toggle of 651900 take it in the column of the values, a hotkey
        waits for its key after a click on the value of the selected row.
        """
        selected = index == self.option_index
        self.select_option(index)
        if x < VALUE_X:
            return 'option'
        key = self.option_key
        kind, _default, _low, high, _choices = OPTIONS[key]
        if kind == KEY:
            if not selected:
                return 'option'
            self.capture = True
            return 'capture'
        if kind == SLIDER:
            left, _track, _knob, right = slider_rects(self.option_row_rect(row)[1], self.settings.value(key), high)
            if x >= SLIDER_X[1]:
                return 'option'
            if x < left[2]:
                self.change(-1)
            elif x >= right[0]:
                self.change(1)
            else:
                self.settings.set_value(key, slider_value(x, high))
            return 'change'
        self.change(1)
        return 'change'

    def on_key(self, name):
        """
        The keys of the window: up/down the options, left/right a value, page up/down the groups,
        Escape cancels, Enter confirms or takes the next key for a hotkey.
        """
        if self.capture:
            if name == 'escape':
                self.capture = False
                return 'capture'
            key = keys_module.windows_name(name)
            if not key:
                return 'capture'
            self.settings.set_value(self.option_key, key)
            self.capture = False
            return 'change'
        if name == 'up':
            self.select_option(self.option_index - 1, -1)
        elif name == 'down':
            self.select_option(self.option_index + 1, 1)
        elif name == 'left':
            if OPTIONS[self.option_key][0] != KEY:
                self.change(-1)
                return 'change'
        elif name == 'right':
            if OPTIONS[self.option_key][0] != KEY:
                self.change(1)
                return 'change'
        elif name == 'pageup':
            self.select_group(self.group_index - 1)
            return 'group'
        elif name == 'pagedown':
            self.select_group(self.group_index + 1)
            return 'group'
        elif name == 'escape':
            return 'cancel'
        elif name == 'enter':
            if OPTIONS[self.option_key][0] == KEY:
                self.capture = True
                return 'capture'
            return 'confirm'
        return None

#------------------------------------------------------------------------------


def _inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


def load_hotkeys(settings, text):
    """
    The hotkeys of config/keyboard.ini fill the options which the settings file does not have.
    """
    if not text:
        return settings
    for action, key in keys_module.parse(text).items():
        option_key = 'key_%s' % action
        if option_key in settings.values and option_key not in settings.explicit:
            settings.values[option_key] = key
    return settings


def load_settings(file_path=None):
    return Settings.load(file_path)
