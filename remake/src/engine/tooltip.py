"""
The tooltips of the windows of game.exe (65e540, partly): a region keeps the name of its text of texts.res
("tip option_gamma"), the tooltip comes when the mouse stands over the region for the delay of the option
"Pop-up tips delay" (ToolTip Time, 0.25 s a step of the slider, 0 switches the tooltips off) and goes away
when the mouse leaves the region. The fade of game.exe (the states 2 and 3 of 65e540) is not here.

The place (65e540): the box of the text is moved from the mouse by the offset of the cursor; a box beyond
the right edge of the 800 x 600 screen is moved left, a box beyond the bottom is moved up so that its bottom
is at the mouse.

The look, measured on the frames of game.exe (plan 3): a frame of one pixel 0xffb231 around the background
0x291c00, the first line of the text (the title) in the color of the frame, the other lines in 0xe4d7a7,
the lines 15 pixels apart, 6 pixels between the frame and the text.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

STEP_MS = 250.0
OFFSET = (24, 24)
SCREEN = (800, 600)

FRAME_COLOR = (255, 178, 49)
BACKGROUND_COLOR = (41, 28, 0)
TITLE_COLOR = (255, 178, 49)
TEXT_COLOR = (228, 215, 167)
LINE_HEIGHT = 15
PADDING = 6
TOP_PADDING = 4

#------------------------------------------------------------------------------


def lines_of(text):
    """
    The lines of a text of texts.res without the carriage returns, the spaces at their ends and the empty
    lines at the end.
    """
    lines = [line.rstrip() for line in (text or '').replace('\r', '').split('\n')]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def box(mouse_x, mouse_y, width, height):
    """
    65e540 (partly): the box (left, top, right, bottom) of a tooltip of the size for the mouse.
    """
    left, top = mouse_x + OFFSET[0], mouse_y + OFFSET[1]
    right, bottom = left + width, top + height
    if right > SCREEN[0]:
        left -= right - SCREEN[0]
        if left < 0:
            left = 0
        right = left + width
    if bottom > SCREEN[1]:
        top -= bottom - mouse_y
        if top < 0:
            top = 0
        bottom = top + height
    return left, top, right, bottom


def size(lines, line_width):
    """
    The size of the box of the lines; line_width(text) is the width of a line in the pixels of the screen.
    """
    width = max([line_width(line) for line in lines] or [0])
    return width + PADDING * 2, len(lines) * LINE_HEIGHT + TOP_PADDING * 2


class Tooltip(object):
    """
    The state of the tooltip of one window: the region under the mouse, the time over it and whether the
    tooltip is up.
    """

    def __init__(self):
        self.tip = None
        self.mouse = (0, 0)
        self.hover_ms = 0.0
        self.shown = False

    def __repr__(self):
        return 'Tooltip(%s, %s)' % (self.tip, 'shown' if self.shown else 'hidden')

    def hover(self, tip, x, y):
        """
        The mouse is at (x, y) over the region of the tip (None outside the regions); True when the
        shown tooltip goes away.
        """
        self.mouse = (x, y)
        if tip == self.tip:
            return False
        hidden = self.shown
        self.tip = tip
        self.hover_ms = 0.0
        self.shown = False
        return hidden

    def hide(self):
        """
        A click: the tooltip goes away and comes back only after the mouse leaves the region.
        """
        hidden = self.shown
        self.shown = False
        self.hover_ms = -1.0
        return hidden

    def update(self, dt_ms, delay_ms):
        """
        True when the tooltip comes up in this frame.
        """
        if self.tip is None or self.shown or self.hover_ms < 0.0 or delay_ms <= 0.0:
            return False
        self.hover_ms += dt_ms
        if self.hover_ms >= delay_ms:
            self.shown = True
            return True
        return False


def delay_ms(slider):
    """
    The delay of the slider "Pop-up tips delay": ToolTip Time is the slider * 0.25 s (652460).
    """
    return float(slider) * STEP_MS
