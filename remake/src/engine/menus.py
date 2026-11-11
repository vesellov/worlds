"""
The main and the escape menus of game.exe: stone columns with buttons, the screen rects of menus.reg.

Main menu (window 632670, mouse 6333c0): the map ZoneMainMenuNew seen by camera\\MainMenu.cam,
the column "unmoco2" is the object with NID 2 of the .mob. The rects of menus.reg [MainMenu] are read
in the order NewGame, LoadGame, ExitGame, Options, Credits, Multiplayer (5ace80) and become the regions 0..5.
When the mouse comes into another region the column plays the action SPECIAL of that button
(1, 3, 5, 7, 11, 9: the stone moves out), out of the buttons it plays the idle 0x80590000,
every change sounds buttons\\menu_stone.wav. The window starts the time of the day at 6 hours.

Escape menu (62aae0): the column "unmoco1" stands in the space of the interface at the screen point (400, 700)
4.8 units deep, turned by 90 degrees around x. 5fc3b0(n) plays SPECIAL n + 1: the column comes up
with SPECIAL 2 (buttons\\menu\\shift.wav) and goes down with SPECIAL 1 (62ac70). The rects of [EscMenu]: SaveGame, LoadGame, Options, ResumeGame, ExitMM,
the stones of the buttons move with SPECIAL 3..7.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import datetime
import math

#------------------------------------------------------------------------------

MENUS_REG = 'menus.reg'

# the models game.exe 667e80 marks as menu columns
COLUMN_MODELS = ('unmoco1', 'unmoco2')

MAIN_MENU = 'MainMenu'
MAIN_MENU_MAP = 'zonemainmenunew'
MAIN_MENU_CAMERA = 'MainMenu'
MAIN_MENU_COLUMN_NID = 2
# game.exe 57c648 reads the local time after the map of the menu is loaded and sets the world hour
# to wHour + wMinute / 60

NEW_GAME = 'NewGame'
LOAD_GAME = 'LoadGame'
EXIT_GAME = 'ExitGame'
OPTIONS = 'Options'
CREDITS = 'Credits'
MULTIPLAYER = 'Multiplayer'

MAIN_MENU_BUTTONS = (NEW_GAME, LOAD_GAME, EXIT_GAME, OPTIONS, CREDITS, MULTIPLAYER)
MAIN_MENU_SPECIALS = (1, 3, 5, 7, 11, 9)

ESC_MENU = 'EscMenu'
ESC_MENU_MODEL = 'unmoco1'
ESC_MENU_TEXTURE = 'escmenu00'
ESC_MENU_LABELS = 'escmenu00labels'
ESC_MENU_SCREEN_POINT = (400.0, 700.0)
ESC_MENU_DEPTH = 4.8
ESC_MENU_TURN = math.pi * 0.5
ESC_MENU_SHOW = 2
ESC_MENU_HIDE = 1

SAVE_GAME = 'SaveGame'
RESUME_GAME = 'ResumeGame'
EXIT_TO_MAIN_MENU = 'ExitMM'

ESC_MENU_BUTTONS = (SAVE_GAME, LOAD_GAME, OPTIONS, RESUME_GAME, EXIT_TO_MAIN_MENU)
ESC_MENU_SPECIALS = (3, 4, 5, 6, 7)

STONE_SOUND = 'buttons\\menu_stone.wav'
SHIFT_SOUND = 'buttons\\menu\\shift.wav'

# while a modal window is up the world and the HUD of game.exe go gray and dark: the color is
# blended to its luminance, then multiplied by 0.41 (measured on the frames of the difficulty
# window and of the menu Esc; the saturation of the scene falls to 0)
PAUSE_FILTER = (1.0, 0.41)
PAUSE_NONE = (0.0, 1.0)

# version text of the main menu (6335a0): "string version_name" in the box 100, 480 - 780, 600
VERSION_TEXT = 'string version_name'
VERSION_BOX = (100, 480, 780, 600)
VERSION_COLOR = (228, 215, 167)

#------------------------------------------------------------------------------


class ButtonColumn(object):
    """
    Which button of a menu is under the mouse and which action of the column it starts.
    """

    def __init__(self, rects, buttons, specials):
        """
        rects: button name -> (left, top, right, bottom) on the 800 x 600 screen.
        """
        self.buttons = [name for name in buttons if name in rects]
        self.rects = [tuple(rects[name]) for name in self.buttons]
        self.specials = dict(zip(buttons, specials))
        self.hovered = None

    def button_at(self, x, y):
        for name, (left, top, right, bottom) in zip(self.buttons, self.rects):
            if left <= x < right and top <= y < bottom:
                return name
        return None

    def hover(self, x, y):
        """
        (changed, special modifier or None for the idle) when the mouse moves to the point.
        """
        name = self.button_at(x, y)
        if name == self.hovered:
            return False, None
        self.hovered = name
        return True, self.specials.get(name)

#------------------------------------------------------------------------------


def main_menu_hour(now=None):
    """
    The hour of the world shown by the main menu: wHour + wMinute / 60 of the local time (57c648).
    """
    now = now or datetime.datetime.now()
    return now.hour + now.minute / 60.0


def main_menu(registry):
    return ButtonColumn(registry.get(MAIN_MENU, {}), MAIN_MENU_BUTTONS, MAIN_MENU_SPECIALS)


def esc_menu(registry):
    return ButtonColumn(registry.get(ESC_MENU, {}), ESC_MENU_BUTTONS, ESC_MENU_SPECIALS)
