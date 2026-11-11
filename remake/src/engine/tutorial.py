"""
The tutorial window of game.exe (658dd0 opens it, 6593e0 lays out a page).

SendStringEvent( 0, "tutorial zt1" ) of a script (5f8070) puts the name into a queue, the window shows
the text "tutorial <name>" of texts.res:

    zt1_head                      the head, "tutor zt1_head" is "Initial information"
    #screen                       a page
    it001_head_screen1            its title, "tutor it001_head_screen1"
    #position 155 34 645 22       left, top, right of the window and the lines of the text (optional)
    #rectangle 11 109 191 491     a red frame over the screen which points at a part of the interface (any count)
    #text
    it001_text_screen1            the text of the page, "tutor it001_text_screen1"

A tutorial without "tutor <head>" or without pages is not shown (5f74e0 closes the window).

The window (6593e0):
- without #position it is 200, 50 - 600, 560; the width is at least 300, the height is
  lines * 16 + 110 (one line when the count is not given) and at most 600;
- the head part is 55 high: the title of the page at y 8..30 in the large font and "<head> (<page>/<pages>)"
  at y 30..50 in the medium font, both in the color 0x82f5ff (COLORREF, RGB ff f5 82);
- the text goes from 15 to the width - 15 below the head part, the buttons are in the last 55;
- the background is 0xff282313, the frames of #rectangle are 0x82ff0000 with the border 5;
- the button "back" is hidden on the first page, "next" on the last one.

The game is paused while the window is open, the key H shows the last tutorial again (zt1 says so).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

TUTORIAL_PREFIX = 'tutorial '
TEXT_PREFIX = 'tutor '

DEFAULT_RECT = (200, 50, 600, 560)
MIN_WIDTH = 300
LINE_HEIGHT = 16
FRAME_HEIGHT = 110
MAX_HEIGHT = 600

HEAD_HEIGHT = 55
BUTTONS_HEIGHT = 55
TITLE_BOX = (0, 8, None, 30)
HEAD_BOX = (0, 30, None, 50)
TEXT_MARGIN = 15

HEAD_COLOR = (0xff, 0xf5, 0x82)
BACKGROUND_COLOR = (0x28 / 255.0, 0x23 / 255.0, 0x13 / 255.0, 1.0)
RECTANGLE_COLOR = (1.0, 0.0, 0.0, 0x82 / 255.0)
RECTANGLE_BORDER = 5

#------------------------------------------------------------------------------


class Screen(object):
    """
    A page of a tutorial: the names of its title and text in texts.res ("tutor <name>").
    """

    def __init__(self, title_name='', position=None, rectangles=(), text_name=''):
        self.title_name = title_name
        # (left, top, right, lines or None), None when the page has no #position
        self.position = position
        self.rectangles = list(rectangles)
        self.text_name = text_name

    def __repr__(self):
        return 'Screen(%s %r)' % (self.title_name, self.position)

    def window_rect(self, line_count=None):
        """
        (left, top, right, bottom) of the window; line_count is the measured text when #position has no count.
        """
        if self.position is None:
            return DEFAULT_RECT
        left, top, right, lines = self.position
        if lines is None:
            lines = line_count if line_count is not None else 1
        if right - left < MIN_WIDTH:
            right = left + MIN_WIDTH
        bottom = top + lines * LINE_HEIGHT + FRAME_HEIGHT
        if bottom - top > MAX_HEIGHT:
            bottom = top + MAX_HEIGHT
        return (left, top, right, bottom)


class Tutorial(object):

    def __init__(self, name, head_name, screens):
        self.name = name
        self.head_name = head_name
        self.screens = list(screens)

    def __repr__(self):
        return 'Tutorial(%s, %d screens)' % (self.name, len(self.screens))


class TutorialWindow(object):
    """
    The queue of the tutorials which the scripts asked for and the page of the one which is shown.
    """

    def __init__(self):
        self.queue = []
        self.tutorial = None
        self.page = 0
        # the tutorial which was shown last, for the key H and the page it was closed on
        self.last = None
        self.last_page = 0
        # called when a tutorial is shown or the window is closed
        self.on_change = None

    @property
    def is_open(self):
        return self.tutorial is not None

    @property
    def screen(self):
        if self.tutorial is None:
            return None
        return self.tutorial.screens[self.page]

    def changed(self):
        if self.on_change is not None:
            self.on_change()

    def add(self, tutorial):
        """
        A tutorial of a script: it is shown at once or after those which are already waiting.
        """
        if tutorial is None or not tutorial.screens:
            return
        self.queue.append(tutorial)
        if self.tutorial is None:
            self.show_next()

    def show_next(self):
        self.tutorial = self.queue.pop(0) if self.queue else None
        if self.tutorial is not None:
            self.last = self.tutorial
        self.page = 0
        self.changed()

    def show_last(self):
        """
        The key H: the last tutorial from the page it was closed on (658dd0 keeps the page of the window).
        """
        if self.tutorial is None and self.last is not None:
            self.tutorial = self.last
            self.page = min(self.last_page, len(self.tutorial.screens) - 1)
            self.changed()

    def has_back(self):
        return self.tutorial is not None and self.page > 0

    def has_next(self):
        return self.tutorial is not None and self.page < len(self.tutorial.screens) - 1

    def back(self):
        if self.has_back():
            self.page -= 1
            self.changed()

    def forward(self):
        if self.has_next():
            self.page += 1
            self.changed()

    def go_to(self, page):
        """
        The page of an icon at the bottom of the window.
        """
        if self.tutorial is not None and 0 <= page < len(self.tutorial.screens):
            self.page = page
            self.changed()
            return True
        return False

    def close(self):
        if self.tutorial is not None:
            self.last_page = self.page
            self.show_next()

#------------------------------------------------------------------------------


def parse(name, text):
    """
    Tutorial of the text "tutorial <name>", None when it has no head.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines or lines[0].startswith('#'):
        return None
    head_name = lines[0]
    screens = []
    screen = None
    expect_text = False
    for line in lines[1:]:
        words = line.split()
        if words[0] == '#screen':
            screen = Screen()
            screens.append(screen)
            expect_text = False
            continue
        if screen is None:
            continue
        if words[0] == '#position':
            numbers = _numbers(words[1:])
            if len(numbers) >= 3:
                screen.position = (numbers[0], numbers[1], numbers[2], numbers[3] if len(numbers) > 3 else None)
        elif words[0] == '#rectangle':
            numbers = _numbers(words[1:])
            if len(numbers) >= 4:
                screen.rectangles.append(tuple(numbers[:4]))
        elif words[0] == '#text':
            expect_text = True
        elif expect_text:
            screen.text_name = line
            expect_text = False
        elif not screen.title_name:
            screen.title_name = line
    return Tutorial(name, head_name, screens)


def load(name, text_of):
    """
    The tutorial "tutorial <name>" with text_of(text_name) giving the texts of texts.res, None when there is none.
    """
    text = text_of(TUTORIAL_PREFIX + name)
    if not text:
        return None
    tutorial = parse(name, text)
    if tutorial is None or not text_of(TEXT_PREFIX + tutorial.head_name):
        return None
    return tutorial


def _numbers(words):
    result = []
    for word in words:
        try:
            result.append(int(word))
        except ValueError:
            break
    return result
