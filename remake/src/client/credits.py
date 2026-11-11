"""
The credits of the main menu (engine.credits) as a layer of the interface: the black screen with the text
of outro.rtf moving up. Escape, Space, Enter or a click end it and the main menu returns.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import credits as credits_module
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'credits'

BACKGROUND_COLOR = (0.0, 0.0, 0.0, 1.0)
TEXT_COLOR = (228, 215, 167)
TEXT_BOX = (100, 700)

STOP_KEYS = ('escape', 'spacebar', 'enter')

#------------------------------------------------------------------------------


class CreditsView(object):
    """
    The scrolling text of the credits over the black screen, the main menu stands below it.
    """

    def __init__(self, app):
        self.app = app
        self.credits = None
        self.is_open = False

    def open(self):
        credits = credits_module.Credits.from_game(self.app.game_data)
        if credits is None:
            self.app.show_message('no credits')
            return False
        self.credits = credits
        self.is_open = True
        self.app.music.system.set_credits()
        self.update(0.0)
        return True

    def close(self):
        if not self.is_open:
            return
        self.is_open = False
        self.credits = None
        self.app.interface.set_layer(LAYER_NAME, [])
        self.app.music.system.set_main_menu()

    def update(self, dt_ms):
        """
        Moves the text; True while the credits are open.
        """
        if not self.is_open:
            return False
        self.credits.update(dt_ms)
        if self.credits.is_done:
            self.close()
            return False
        left, right = TEXT_BOX
        top = self.credits.offset
        items = [
            ui.box_item((0, 0, ui.WIDTH, ui.HEIGHT), BACKGROUND_COLOR),
            ui.text_item(self.credits.text, (left, top, right, top + self.credits.text_height),
                         ui.FONT_MEDIUM, TEXT_COLOR, align='center'),
        ]
        self.app.interface.set_layer(LAYER_NAME, items)
        return True

    def on_click(self, x, y):
        if not self.is_open:
            return False
        self.close()
        return True

    def on_key(self, name):
        if not self.is_open:
            return False
        if name in STOP_KEYS:
            self.close()
        return True
