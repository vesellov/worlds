"""
The modal message box of game.exe (634750): the dark panel with the title, the text of the question and
the two buttons of the saveload atlas.

The escape menu makes it for "Exit to main menu" (62b00d): the title is "string quit" ("Quit game"),
the text is "string quit_msg" ("Do you really want to quit this game?"), the message id is 0x10000 and
the flag 3 gives both buttons, the check (message id | 1) and the cross (message id | 2). The window of
the escape menu closes before the box comes up. The rectangles here are measured on the screen of
game.exe (800 x 600) for this text.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

# the keys of texts.res
TITLE_KEY = 'string quit'
MESSAGE_KEY = 'string quit_msg'

# the black panel over the scene and the frame of 5 pixels around it (5e7d60): 3 dark pixels outside
# and 2 light ones inside them
PANEL_RECT = (200, 224, 600, 326)
FRAME_RECT = (195, 219, 605, 331)
BORDER_WIDTH = 5
BORDER_HIGH_WIDTH = 2

# the title and the text are centered in the panel, their boxes measured on the screen of game.exe
TITLE_BOX = (200, 230, 600, 262)
MESSAGE_BOX = (200, 271, 600, 303)

# the sprites of the buttons and their regions of the mouse (633cb1, 634ad7)
OK_RECT = (262, 357, 337, 405)
CANCEL_RECT = (463, 357, 537, 405)
OK_REGION = (263, 357, 337, 405)
CANCEL_REGION = (463, 357, 537, 405)

# the pieces of the saveload atlas in its pixels from the top of the picture, as engine.difficulty
ATLAS_NAME = 'saveload'
OK_TEXELS = (0x51, 254, 0x9b, 206)
CANCEL_TEXELS = (0xa0, 254, 0xea, 206)

# the colors of the window of 634750 (COLORREF, written here as RGB)
TITLE_COLOR = (255, 179, 49)
TEXT_COLOR = (228, 215, 167)
# the panel is black with the alpha 0xa0 and the scene shows through it
PANEL_COLOR = (0.0, 0.0, 0.0, 160.0 / 255.0)
BORDER_COLOR = (49, 52, 36)
BORDER_HIGH_COLOR = (120, 117, 82)

# 634e20: Enter reports the check, Escape the cross
ENTER_KEYS = ('enter', 'return', 'kp_enter')

#------------------------------------------------------------------------------


def inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


class Dialog(object):
    """
    The clicks and the keys of the message box: "confirm", "cancel" or None.
    """

    def click(self, x, y):
        if inside(OK_REGION, x, y):
            return 'confirm'
        if inside(CANCEL_REGION, x, y):
            return 'cancel'
        return None

    def on_key(self, name):
        if name in ENTER_KEYS:
            return 'confirm'
        if name == 'escape':
            return 'cancel'
        return None
