"""
The tutorial window in the client: engine.tutorial.TutorialWindow drawn as a layer of client.ui.Interface.

The buttons at the bottom are the pieces of the saveload atlas of game.exe (65a025): the narrow panels with
the left arrow, the check and the right arrow, 50 x 20, in the row 30 - 10 above the bottom of the window.
The button of the key Enter (next, or the check on the last page) is bright, the others are at 0.6 (6596b9).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from engine import tutorial
from client import ui

#------------------------------------------------------------------------------

LAYER_NAME = 'tutorial'

TEXT_COLOR = (228, 215, 167)
# the frame of the window is 5 pixels outside its rect: 3 dark pixels and 2 light ones inside them
BORDER_WIDTH = 5
BORDER_HIGH_WIDTH = 2
BORDER_COLOR = (58 / 255.0, 57 / 255.0, 41 / 255.0, 1.0)
BORDER_HIGH_COLOR = (123 / 255.0, 113 / 255.0, 82 / 255.0, 1.0)
# the text wraps a little earlier than the right margin of the window, as in game.exe
TEXT_RIGHT_MARGIN = 16
SHOW_TUTORIAL_LABEL = 'Show tutorial: H'

BUTTON_WIDTH = 50
# the top and the bottom of the buttons above the bottom of the window
BUTTON_TOP_INSET = 30
BUTTON_BOTTOM_INSET = 10
# 65a039: the gap before the first button is (width - 150) / 6, the buttons stand 2 gaps apart
BUTTONS_SPACE = 150
DIM_COLOR = (0.6, 0.6, 0.6, 1.0)
LABEL_BOX_HALF_WIDTH = 80
LABEL_TOP_MARGIN = 24
LABEL_BOTTOM_GAP = 6

BACK = 'back'
NEXT = 'next'
CLOSE = 'close'

BUTTON_ATLAS_NAME = 'saveload'

# the pieces of the atlas in its pixels from the top of the picture: 256 minus the v of game.exe
# (u0, v0, u1, v1) back (2, 0x33, 0x34, 0x47), close (2, 0x5d, 0x34, 0x71) and next (2, 0x48, 0x34, 0x5c);
# v0 is at the top of the button, so the pieces stand upside down as in engine.difficulty
BUTTON_TEXELS = {
    BACK: (2, 205, 52, 185),
    CLOSE: (2, 163, 52, 143),
    NEXT: (2, 184, 52, 164),
}

#------------------------------------------------------------------------------


class TutorialView(object):

    def __init__(self, app):
        self.app = app
        self.window = tutorial.TutorialWindow()
        self.window.on_change = self.update
        # name -> (left, top, right, bottom) of the buttons on the screen
        self.buttons = {}

    @property
    def is_open(self):
        return self.window.is_open

    def text(self, name):
        return (self.app.game_data.text(tutorial.TEXT_PREFIX + name) or '').replace('\r', '').strip('\n')

    def show(self, name):
        """
        SendStringEvent "tutorial <name>"; nothing while the hints are off (the option show_tutorial,
        ShowTutorial of the registry of game.exe, and --tutorial of the remake).
        """
        if not self.app.tutorial_enabled():
            return
        self.window.add(tutorial.load(name, self.app.game_data.text))
        # the hint of the keys of the demo must not cover the tutorial window
        if self.window.is_open:
            self.app.show_help('', 0)

    def update(self):
        interface = self.app.interface
        screen = self.window.screen
        self.buttons = {}
        if screen is None:
            interface.set_layer(LAYER_NAME, [])
            return
        # the hint of the keys of the demo must not cover the tutorial window
        self.app.show_help('', 0)
        head = self.text(self.window.tutorial.head_name)
        title = self.text(screen.title_name)
        text = self.text(screen.text_name)
        line_count = None
        if screen.position is not None and screen.position[3] is None:
            left, top, right = screen.position[:3]
            width = max(right - left, tutorial.MIN_WIDTH) - 2 * tutorial.TEXT_MARGIN - TEXT_RIGHT_MARGIN
            line_count = int(interface.text_height(text, width) / tutorial.LINE_HEIGHT) + 1
        left, top, right, bottom = screen.window_rect(line_count)
        items = [ui.box_item((left, top, right, bottom), tutorial.BACKGROUND_COLOR)]
        items.extend(ui.bevel_items((left, top, right, bottom), BORDER_WIDTH, BORDER_HIGH_WIDTH,
                                    BORDER_COLOR, BORDER_HIGH_COLOR))
        for x0, y0, x1, y1 in screen.rectangles:
            items.extend(frame_items((x0, y0, x1, y1), tutorial.RECTANGLE_BORDER, tutorial.RECTANGLE_COLOR))
        items.append(ui.text_item(title, (left, top + tutorial.TITLE_BOX[1], right, top + tutorial.TITLE_BOX[3]),
                                  ui.FONT_LARGE, tutorial.HEAD_COLOR, align='center'))
        counter = '%s (%d/%d)' % (head, self.window.page + 1, len(self.window.tutorial.screens))
        items.append(ui.text_item(counter, (left, top + tutorial.HEAD_BOX[1], right, top + tutorial.HEAD_BOX[3]),
                                  ui.FONT_MEDIUM, tutorial.HEAD_COLOR, align='center'))
        text_right = right - tutorial.TEXT_MARGIN - TEXT_RIGHT_MARGIN
        items.append(ui.text_item(text, (left + tutorial.TEXT_MARGIN, top + tutorial.HEAD_HEIGHT,
                                         text_right, bottom - tutorial.BUTTONS_HEIGHT),
                                  ui.FONT_MEDIUM, TEXT_COLOR))
        y0 = bottom - BUTTON_TOP_INSET
        gap = (right - left - BUTTONS_SPACE) // 6
        step = 2 * gap + BUTTON_WIDTH
        places = {
            BACK: left + gap,
            CLOSE: left + gap + step,
            NEXT: left + gap + 2 * step,
        }
        centers = {name: x + BUTTON_WIDTH // 2 for name, x in places.items()}
        labels = {CLOSE: SHOW_TUTORIAL_LABEL}
        shown = [CLOSE]
        pages = self.window.tutorial.screens
        if self.window.has_back():
            shown.append(BACK)
            labels[BACK] = self.text(pages[self.window.page - 1].title_name)
        if self.window.has_next():
            shown.append(NEXT)
            labels[NEXT] = self.text(pages[self.window.page + 1].title_name)
        for name in shown:
            center = centers[name]
            items.append(ui.text_item(labels[name],
                                      (center - LABEL_BOX_HALF_WIDTH, y0 - LABEL_TOP_MARGIN,
                                       center + LABEL_BOX_HALF_WIDTH, y0 - LABEL_BOTTOM_GAP),
                                      ui.FONT_MEDIUM, TEXT_COLOR, align='center', valign='bottom'))
            rect = (places[name], y0, places[name] + BUTTON_WIDTH, bottom - BUTTON_BOTTOM_INSET)
            self.buttons[name] = rect
            # Enter presses next, or the check on the last page: that button is bright
            bright = name == NEXT or (name == CLOSE and not self.window.has_next())
            items.append(ui.sprite_item(BUTTON_ATLAS_NAME, rect, BUTTON_TEXELS[name],
                                        None if bright else DIM_COLOR))
        interface.set_layer(LAYER_NAME, items)

    def on_click(self, x, y):
        """
        True while the window is open: it takes every click.
        """
        if not self.is_open:
            return False
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.press(name)
                break
        return True

    def press(self, name):
        if name == NEXT:
            self.app.ui_sound('buttons/Tutorial/Next.wav')
            self.window.forward()
        elif name == BACK:
            self.app.ui_sound('buttons/Tutorial/Ok.wav')
            self.window.back()
        elif name == CLOSE:
            self.app.ui_sound('buttons/Tutorial/Ok.wav')
            self.window.close()

    def on_key(self, name):
        """
        The keys of the open window: Enter turns the pages and closes the last one,
        Escape closes the window, as in game.exe.
        """
        if not self.is_open:
            if name == 'h' and self.window.last is not None:
                self.window.show_last()
                return True
            return False
        if name in ('enter', 'return'):
            if self.window.has_next():
                self.window.forward()
            else:
                self.window.close()
        elif name == 'escape':
            self.window.close()
        return True

#------------------------------------------------------------------------------


def frame_items(rect, border, rgba):
    """
    Four boxes along the inner side of the rect.
    """
    x0, y0, x1, y1 = rect
    return [
        ui.box_item((x0, y0, x1, y0 + border), rgba),
        ui.box_item((x0, y1 - border, x1, y1), rgba),
        ui.box_item((x0, y0 + border, x0 + border, y1 - border), rgba),
        ui.box_item((x1 - border, y0 + border, x1, y1 - border), rgba),
    ]
