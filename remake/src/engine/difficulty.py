"""
The window of the difficulty of a new game (633770) and the levels it chooses.

The main menu asks the difficulty before "New game": the window 633770 draws the title "Difficulty"
(string option_difficulty), the two levels with empty check boxes and the selected check, the text of
tip option_difficulty and the buttons of the saveload atlas. The rectangles and the sprite texels here
are measured on the screen of game.exe (800 x 600).

The level of the game is DAT_0079beb0: 0 is Normal (the default, the full Attack, Defence and Absorption
of the enemies) and 1 is Novice (the ratios [1] of [DifficultyLevels] of config/ai.reg: Attack 0.66,
Defence 0.66, Absorption 0.33). CUnitServer__ApplyDifficulty (52e0d0) gives them to the units which are
not of the party. The option option_difficulty of the options window is the same setting, its button
shows difficulty_hard when it is on and difficulty_easy when it is off.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

NOVICE = 0
NORMAL = 1

DEFAULT = NORMAL

LEVELS = (NOVICE, NORMAL)

# DAT_0079beb0: the index of the arrays of [DifficultyLevels] of config/ai.reg
GAME_INDEX = {
    NOVICE: 1,
    NORMAL: 0,
}

# the keys of texts.res
TITLE_KEY = 'string option_difficulty'
TIP_KEY = 'tip option_difficulty'
LEVEL_KEYS = {
    NOVICE: 'string difficulty_easy',
    NORMAL: 'string difficulty_hard',
}
LEVEL_NAMES = {
    NOVICE: 'Novice',
    NORMAL: 'Normal',
}

# the window of 633770 on the screen of 800 x 600
PANEL_RECT = (195, 115, 605, 434)
PANEL_BORDER = 4
TITLE_BOX = (195, 130, 605, 160)
DESCRIPTION_BOX = (210, 228, 560, 424)
ROW_RECTS = {
    NOVICE: (350, 166, 570, 194),
    NORMAL: (350, 194, 570, 222),
}
CHECKBOX_RECTS = {
    NOVICE: (357, 171, 378, 192),
    NORMAL: (357, 195, 378, 216),
}
# the check is 9 x 15 like its piece of the atlas, 6 right and 5 down from the corner of its box
CHECK_RECTS = {
    NOVICE: (363, 176, 372, 191),
    NORMAL: (363, 200, 372, 215),
}
LABEL_BOXES = {
    NOVICE: (383, 166, 570, 196),
    NORMAL: (383, 196, 570, 226),
}
# 633cb1: the buttons are 0x107 - 0x151 and 0x1cf - 0x219 wide and 48 high, as their pieces of the atlas
OK_RECT = (263, 455, 337, 503)
CANCEL_RECT = (463, 455, 537, 503)

# the pieces of the saveload atlas in its pixels from the top of the picture; game.exe keeps v from the
# bottom of the picture and puts v0 at the top of the button (5e70e0), so the pieces stand upside down
# on the screen: the "top" here is below the "bottom" and client.ui draws them flipped. These are
# 256 minus its v of CHECKBOX (0xd1, 0x37, 0xe6, 0x4c), CHECK (192, 60, 201, 75),
# OK (0x51, 2, 0x9b, 0x32) and CANCEL (0xa0, 2, 0xea, 0x32)
CHECKBOX_TEXELS = (0xd1, 201, 0xe6, 180)
CHECK_TEXELS = (192, 196, 201, 181)
OK_TEXELS = (0x51, 254, 0x9b, 206)
CANCEL_TEXELS = (0xa0, 254, 0xea, 206)

# the colors of the window of 633770 (COLORREF, written here as RGB)
TITLE_COLOR = (255, 179, 49)
TEXT_COLOR = (228, 215, 167)
PANEL_COLOR = (0, 0, 0)
BORDER_COLOR = (90, 90, 66)
BORDER_HIGH_COLOR = (132, 130, 90)

#------------------------------------------------------------------------------


def inside(rect, x, y):
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


def level_label(text, level):
    """
    The name of a level from texts.res, level_names when the text is missing.
    """
    name = (text(LEVEL_KEYS[level]) or '').replace('\r', '').strip()
    return name or LEVEL_NAMES[level]


class Dialog(object):
    """
    The choice of the new game: the level, the clicks and the keys of the window of 633770.
    """

    def __init__(self, selection=DEFAULT):
        self.selection = selection

    def __repr__(self):
        return 'DifficultyDialog(%s)' % LEVEL_NAMES[self.selection]

    def select(self, level):
        if level in LEVELS:
            self.selection = level

    def click(self, x, y):
        """
        The action of a click: "select" (the choice moved), "confirm", "cancel" or None.
        """
        for level in LEVELS:
            if inside(ROW_RECTS[level], x, y):
                self.selection = level
                return 'select'
        if inside(OK_RECT, x, y):
            return 'confirm'
        if inside(CANCEL_RECT, x, y):
            return 'cancel'
        return None

    def on_key(self, name):
        """
        The keys of the window: Escape cancels it (checked on game.exe), the others do nothing.
        """
        return 'cancel' if name == 'escape' else None

    @property
    def game_index(self):
        """
        DAT_0079beb0 for the chosen level.
        """
        return GAME_INDEX[self.selection]
