"""
The credits of the main menu: res/outro.res holds credits.scr ("#scroll 25 outro.rtf") and the text of
the credits outro.rtf (found by the function which loads "outro.res" and reads "credits.scr", 62b7xx).

The screen shows the text of the RTF which moves up at the speed of the #scroll line, over the black
screen, with the Credits track of res/music.reg. Escape ends it and the main menu returns.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import rtf

#------------------------------------------------------------------------------

OUTRO_ARCHIVE = 'res/outro.res'
SCRIPT_NAME = 'credits.scr'
TEXT_NAME = 'outro.rtf'

SCROLL_DIRECTIVE = '#scroll'

DEFAULT_SPEED = 25.0

# the line of the text on the screen of 800 x 600
LINE_HEIGHT = 16.0
SCREEN_HEIGHT = 600.0

#------------------------------------------------------------------------------


def parse_script(data):
    """
    (speed, file name) of credits.scr; the file name may be missing.
    """
    speed = DEFAULT_SPEED
    text_name = TEXT_NAME
    for line in bytes(data).decode(rtf.ENCODING, errors='replace').splitlines():
        words = line.split()
        if len(words) >= 2 and words[0].lower() == SCROLL_DIRECTIVE:
            try:
                speed = float(words[1])
            except ValueError:
                speed = DEFAULT_SPEED
            if len(words) >= 3:
                text_name = words[2]
    return speed, text_name


class Credits(object):
    """
    The scrolling text: offset is the top of the text on the screen, 600 at the start.
    """

    def __init__(self, text, speed=DEFAULT_SPEED, height=SCREEN_HEIGHT):
        self.text = text
        self.speed = speed
        self.offset = height
        self.height = height

    def __repr__(self):
        return 'Credits(%d lines, speed %.1f)' % (self.line_count, self.speed)

    @property
    def line_count(self):
        return self.text.count('\n') + 1 if self.text else 0

    @property
    def text_height(self):
        return self.line_count * LINE_HEIGHT

    @property
    def is_done(self):
        return self.offset + self.text_height < 0.0

    def update(self, dt_ms):
        """
        Moves the text up; True when it has left the screen.
        """
        self.offset -= self.speed * dt_ms / 1000.0
        return self.is_done

    @classmethod
    def from_game(cls, game_data):
        """
        The credits of res/outro.res; None when the archive has no credits.scr or the text.
        """
        archive = game_data.archive(OUTRO_ARCHIVE)
        if SCRIPT_NAME not in archive:
            return None
        speed, text_name = parse_script(archive.read(SCRIPT_NAME))
        if text_name not in archive:
            return None
        return cls(rtf.plain_text(archive.read(text_name)), speed)
