"""
The keys of game.exe: config/keyboard.ini (the file is read by the game as a text of lines "key action")
and the names of the keys of Kivy.

A line of keyboard.ini is the name of a key of Windows and the action of the game: "Z run", "KP_UP cs_head",
"F1 select1", "SPACE pause". The actions are the groups of the options actions, items, spells, cshots,
windows and camera (PLAN groups of 650380). The remake keeps the key of an action in the options as a
string of the same names, so the file and the options speak the same language.

Kivy names the keys in its own way (window.keycode_to_string): "Z" is "z", "SPACE" is "spacebar",
"KP_UP" is "numpad8", "KP_PLUS" is "numpadadd". kivy_name() and windows_name() translate the two.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

ENCODING = 'cp1251'

# the groups of the hotkeys of the options window (650380), in the order of the window
GROUPS = (
    ('actions', ('obj', 'run', 'walk', 'sneak', 'crawl', 'pause', 'swarm', 'use_science', 'follow',
                 'tutorial_script', 'select1', 'select2', 'select3', 'select_all', 'accel', 'decel')),
    ('items', ('weapon1', 'weapon2', 'weapon3', 'weapon4', 'item1', 'item2', 'item3', 'item4',
               'quicksave', 'quickload')),
    ('spells', ('spell1', 'spell2', 'spell3', 'spell4', 'spell5', 'spell6', 'spell7', 'spell8')),
    ('cshots', ('cs_head', 'cs_body', 'cs_lhand', 'cs_rhand', 'cs_lleg', 'cs_rleg')),
    ('windows', ('w_text1', 'w_text2', 'w_info1', 'w_info2', 'w_info3', 'w_info4', 'w_minimap')),
    ('camera', ('camera_norm', 'camera1', 'camera2', 'camera3', 'camera4', 'camera_track',
                'camera_zoom_in', 'camera_zoom_out', 'camera_up', 'camera_down', 'camera_left',
                'camera_right')),
)

# the actions of all the groups in the order of the window
ACTIONS = tuple(action for _group, actions in GROUPS for action in actions)

# the keys of config/keyboard.ini of the game
DEFAULTS = {
    'spell1': '1', 'spell2': '2', 'spell3': '3', 'spell4': '4', 'spell5': '5', 'spell6': '6',
    'spell7': '7', 'spell8': '8',
    'obj': 'TAB', 'weapon1': 'Q', 'weapon2': 'W', 'weapon3': 'E', 'weapon4': 'R',
    'item4': 'U', 'item3': 'I', 'item2': 'O', 'item1': 'P',
    'swarm': 'A', 'use_science': 'S', 'follow': 'F', 'tutorial_script': 'H',
    'w_text2': 'K', 'w_text1': 'L', 'w_info4': "'", 'w_info1': ',', 'w_info2': '.', 'w_info3': '/',
    'run': 'Z', 'walk': 'X', 'sneak': 'C', 'crawl': 'V', 'camera_norm': 'N', 'w_minimap': 'M',
    'pause': 'SPACE',
    'select1': 'F1', 'select2': 'F2', 'select3': 'F3', 'select_all': 'F4',
    'quicksave': 'F5', 'quickload': 'F8', 'camera1': 'F9', 'camera2': 'F10', 'camera3': 'F11',
    'camera4': 'F12',
    'cs_head': 'KP_UP', 'decel': 'KP_MINUS', 'cs_rhand': 'KP_LEFT', 'cs_body': 'KP_CENTER',
    'cs_lhand': 'KP_RIGHT', 'accel': 'KP_PLUS', 'cs_rleg': 'KP_END', 'cs_lleg': 'KP_PGDN',
    'camera_track': 'HOME', 'camera_up': 'UP', 'camera_zoom_in': 'PGUP', 'camera_left': 'LEFT',
    'camera_right': 'RIGHT', 'camera_down': 'DOWN', 'camera_zoom_out': 'PGDN',
}

# the names of the keys of Windows and of Kivy
WINDOWS_TO_KIVY = {
    'TAB': 'tab', 'SPACE': 'spacebar', 'ESC': 'escape', 'ESCAPE': 'escape',
    'KP_UP': 'numpad8', 'KP_DOWN': 'numpad2', 'KP_LEFT': 'numpad4', 'KP_RIGHT': 'numpad6',
    'KP_CENTER': 'numpad5', 'KP_HOME': 'numpad7', 'KP_END': 'numpad1', 'KP_PGUP': 'numpad9',
    'KP_PGDN': 'numpad3', 'KP_DEL': 'numpaddecimal', 'KP_ENTER': 'numpadenter',
    'KP_PLUS': 'numpadadd', 'KP_MINUS': 'numpadsubstract', 'KP_MULTIPLY': 'numpadmul',
    'KP_DIVIDE': 'numpaddivide',
    'UP': 'up', 'DOWN': 'down', 'LEFT': 'left', 'RIGHT': 'right',
    'HOME': 'home', 'END': 'end', 'PGUP': 'pageup', 'PGDN': 'pagedown', 'INS': 'insert',
    'DEL': 'delete', 'BACK': 'backspace', 'ENTER': 'enter',
    'F1': 'f1', 'F2': 'f2', 'F3': 'f3', 'F4': 'f4', 'F5': 'f5', 'F6': 'f6', 'F7': 'f7',
    'F8': 'f8', 'F9': 'f9', 'F10': 'f10', 'F11': 'f11', 'F12': 'f12',
}

KIVY_TO_WINDOWS = dict((value, key) for key, value in WINDOWS_TO_KIVY.items())

# the names of the keys as the window of the options of game.exe writes them (650380): a keypad
# key is KP_Up, Page Up is Prior, space is lower case
DISPLAY_NAMES = {
    'KP_UP': 'KP_Up', 'KP_DOWN': 'KP_Down', 'KP_LEFT': 'KP_Left', 'KP_RIGHT': 'KP_Right',
    'KP_CENTER': 'KP_Begin', 'KP_HOME': 'KP_Home', 'KP_END': 'KP_End',
    'KP_PGUP': 'KP_Prior', 'KP_PGDN': 'KP_Next', 'KP_DEL': 'KP_Delete',
    'KP_PLUS': 'KP_Add', 'KP_MINUS': 'KP_Subtract', 'KP_MULTIPLY': 'KP_Multiply',
    'KP_DIVIDE': 'KP_Divide', 'KP_ENTER': 'KP_Enter',
    'PGUP': 'Prior', 'PGDN': 'Next', 'INS': 'Insert', 'DEL': 'Delete', 'BACK': 'Backspace',
    'SPACE': 'space', 'TAB': 'Tab', 'ESC': 'Esc', 'ESCAPE': 'Esc',
    'UP': 'Up', 'DOWN': 'Down', 'LEFT': 'Left', 'RIGHT': 'Right', 'HOME': 'Home', 'END': 'End',
}

#------------------------------------------------------------------------------


def display_name(name):
    """
    The name of the key in the window of the options of game.exe, the name itself when there is none.
    """
    return DISPLAY_NAMES.get((name or '').upper(), name)


def kivy_name(name):
    """
    The name of a key of Windows (as keyboard.ini writes it) in Kivy, '' when the remake has no such key.
    """
    if not name:
        return ''
    upper = name.upper()
    if upper in WINDOWS_TO_KIVY:
        return WINDOWS_TO_KIVY[upper]
    if len(name) == 1:
        return name.lower()
    return ''


def windows_name(kivy):
    """
    The name of a key of Kivy in the names of keyboard.ini, '' when it has none.
    """
    if not kivy:
        return ''
    lower = kivy.lower()
    if lower in KIVY_TO_WINDOWS:
        return KIVY_TO_WINDOWS[lower]
    if len(kivy) == 1:
        return kivy.upper()
    return ''


def parse(text):
    """
    action -> key of keyboard.ini; the comments and the empty lines are skipped.
    """
    result = {}
    for raw_line in (text or '').splitlines():
        line = raw_line.split(';', 1)[0].split('//', 1)[0].strip()
        if not line:
            continue
        words = line.split()
        if len(words) < 2:
            continue
        key, action = words[0], words[1].lower()
        if action in ACTIONS:
            result[action] = key.upper()
    return result


def bindings(settings):
    """
    action -> key for the client: the value of the options when it has one, the default otherwise.
    """
    result = dict(DEFAULTS)
    for action in ACTIONS:
        key = settings.get('key_%s' % action) if settings is not None else None
        if key:
            result[action] = str(key)
    return result
