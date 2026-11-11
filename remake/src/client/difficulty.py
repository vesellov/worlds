"""
The window of the difficulty of a new game (engine.difficulty, game.exe 633770) as a layer of the
interface: the dark panel, the title, the two levels with their check boxes, the text of the tip and
the buttons of the saveload atlas.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import difficulty
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'difficulty'

ATLAS_NAME = 'saveload'

# the panel of the window is black with the alpha 0xa0 over the menu which the pause filter has
# already made gray and dark (game.exe 633bb5); the menu shows through it
PANEL_COLOR = (0.0, 0.0, 0.0, 160.0 / 255.0)

TITLE_FONT = ui.FONT_LARGE
LEVEL_FONT = ui.FONT_LARGE
TEXT_FONT = ui.FONT_MEDIUM

# the title is the text of option_difficulty, "Difficulty"
SOUND_SELECT = 'buttons/Save/Select.wav'
SOUND_OK = 'buttons/Menu/Ok.wav'
SOUND_CANCEL = 'buttons/MessBox/Cancel.wav'

TITLE_FALLBACK = 'Difficulty'

#------------------------------------------------------------------------------


def _color(rgb):
    return tuple(c / 255.0 for c in rgb) + (1.0, )


class DifficultyWindow(object):
    """
    The modal window of the new game: True from open() to close(), the clicks go to the dialog.
    """

    def __init__(self, app, on_confirm=None, on_cancel=None, selection=difficulty.DEFAULT):
        self.app = app
        self.dialog = difficulty.Dialog(selection)
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.is_open = False

    def text(self, key):
        return (self.app.game_data.text(key) or '').replace('\r', '')

    def open(self):
        self.is_open = True
        self.update()

    def close(self):
        self.is_open = False
        self.app.interface.set_layer(LAYER_NAME, [])

    def update(self):
        """
        Draws the layer again: the choice shows its check at once (633770).
        """
        items = ui.panel_items(difficulty.PANEL_RECT, difficulty.PANEL_BORDER,
                               _color(difficulty.BORDER_COLOR), _color(difficulty.BORDER_HIGH_COLOR),
                               PANEL_COLOR)
        title = self.text(difficulty.TITLE_KEY).strip() or TITLE_FALLBACK
        items.append(ui.text_item(title, difficulty.TITLE_BOX, TITLE_FONT, difficulty.TITLE_COLOR,
                                  align='center', valign='middle'))
        items.append(ui.text_item(self.text(difficulty.TIP_KEY).strip(), difficulty.DESCRIPTION_BOX,
                                  TEXT_FONT, difficulty.TEXT_COLOR))
        for level in difficulty.LEVELS:
            items.append(ui.sprite_item(ATLAS_NAME, difficulty.CHECKBOX_RECTS[level],
                                        difficulty.CHECKBOX_TEXELS))
            if level == self.dialog.selection:
                items.append(ui.sprite_item(ATLAS_NAME, difficulty.CHECK_RECTS[level],
                                            difficulty.CHECK_TEXELS))
            items.append(ui.text_item(difficulty.level_label(self.app.game_data.text, level),
                                      difficulty.LABEL_BOXES[level], LEVEL_FONT, difficulty.TEXT_COLOR,
                                      valign='middle'))
        items.append(ui.sprite_item(ATLAS_NAME, difficulty.OK_RECT, difficulty.OK_TEXELS))
        items.append(ui.sprite_item(ATLAS_NAME, difficulty.CANCEL_RECT, difficulty.CANCEL_TEXELS))
        self.app.interface.set_layer(LAYER_NAME, items)

    def confirm(self):
        level = self.dialog.selection
        self.close()
        if self.on_confirm is not None:
            self.on_confirm(level)

    def cancel(self):
        self.close()
        if self.on_cancel is not None:
            self.on_cancel()

    def on_click(self, x, y):
        """
        True while the window is open: it takes every click.
        """
        if not self.is_open:
            return False
        action = self.dialog.click(x, y)
        if action == 'select':
            self.app.ui_sound(SOUND_SELECT)
            self.update()
        elif action == 'confirm':
            self.app.ui_sound(SOUND_OK)
            self.confirm()
        elif action == 'cancel':
            self.app.ui_sound(SOUND_CANCEL)
            self.cancel()
        return True

    def on_key(self, name):
        """
        True while the window is open: it takes every key; Escape cancels it.
        """
        if not self.is_open:
            return False
        if self.dialog.on_key(name) == 'cancel':
            self.app.ui_sound(SOUND_CANCEL)
            self.cancel()
        return True
