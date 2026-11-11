"""
The flying numbers over the units: the lost health and the gained experience (game.exe CFlyingHitPoints,
the list of 5e1870, drawn by 5e18e0 when the option show_flying_hp (79be64) is on).

A number comes from the client part of a unit (52ef90 (partly)): when the share of the health of the unit fell,
the lost health truncated (the share before minus the share now, times the maximum) flies with the flags
of the last blow (+0x6f8 of the unit, 5254b0 and 525070): 1 BACKSTAB (a blow with a factor of the backstab),
2 CRITICAL (a blow at the part 0, the skull). The units of the party show the experience they gain (66a440,
66a560): round(1.04 ^ (intelligence - 25) * share) with the flag 8 EXP, half a second later.

5e18e0 draws an entry while its age is 0..3000 ms and forgets it after 3000 ms. Its box stands in the middle of
the rectangle of the head ("hd", else "bd", else "hp") on the screen and goes 120 pixels right and 100 up
during the 3 seconds, fading out. The digits are 8 x 12 pixels; every flag adds a word of 64 x 10 under them
(the box is then 64 wide at least). The color is cyan for EXP, red for a unit of a player, green for the others.
The glyphs are in the texture "numbers" of menus.res; the texture is upside down in the file, as the v of
Direct3D goes from its first row.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------

TEXTURE_NAME = 'numbers'

LIFE_MS = 3000
EXP_DELAY_MS = 500

BACKSTAB = 1
CRITICAL = 2
EXP = 8

# 73fdc4, 740094: the way of a number during its life in pixels
DRIFT_X = 120.0
DRIFT_Y = 100.0

DIGIT_WIDTH = 8
DIGIT_HEIGHT = 12
WORD_WIDTH = 64
WORD_HEIGHT = 10

# 740088, 74008c: the digits are 4 in a row of the texture, the rows go from v = 1 down
DIGIT_U = 0.125
DIGIT_V = 0.15625

# the words of the flags in their order under the digits: (flag, v at the top, v at the bottom)
WORDS = (
    (BACKSTAB, 0.5390625, 0.390625),
    (CRITICAL, 0.390625, 0.2421875),
    (EXP, 0.2421875, 0.09375),
)

# (r, g, b) 0..1
COLOR_EXP = (0x40 / 255.0, 1.0, 1.0)
COLOR_PLAYER = (1.0, 0x40 / 255.0, 0x40 / 255.0)
COLOR_OTHER = (0x40 / 255.0, 1.0, 0x40 / 255.0)

# 6e8160 with 73e128, 73e124 in 66a440
INTELLIGENCE_FACTOR = 1.04
PLAIN_INTELLIGENCE = 25

#------------------------------------------------------------------------------


def lost_health(old_part, new_part, max_health):
    """
    52ef90 (partly): the number of a fall of the share of the health, None when the share did not fall.
    """
    if new_part >= old_part:
        return None
    return int((old_part - new_part) * max_health)


def shown_experience(share, intelligence):
    """
    66a440: the number of the experience a unit of the party gains from its share.
    """
    return int(round(INTELLIGENCE_FACTOR ** (intelligence - PLAIN_INTELLIGENCE) * share))


def hit_flags(hit):
    """
    5254b0 and 525070: the flags of a blow (engine.combat.Hit) for its number.
    """
    flags = 0
    if hit.backstab != 1.0:
        flags |= BACKSTAB
    if hit.location == 0:
        flags |= CRITICAL
    return flags


def quads(value, flags):
    """
    The box of a number: (width, height, [(x, y, w, h, u0, v_top, u1, v_bottom)]) with x, y from the top left
    corner of the box down and right.
    """
    text = '%d' % value
    width = len(text) * DIGIT_WIDTH
    height = DIGIT_HEIGHT
    words = [(v_top, v_bottom) for flag, v_top, v_bottom in WORDS if flags & flag]
    if words:
        width = max(width, WORD_WIDTH)
        height += WORD_HEIGHT * len(words)
    result = []
    x = (width - len(text) * DIGIT_WIDTH) * 0.5
    for char in text:
        if not char.isdigit():
            # the minus of a negative number is no glyph of the texture
            x += DIGIT_WIDTH
            continue
        digit = ord(char) - ord('0')
        u0 = (digit & 3) * DIGIT_U
        v_top = 1.0 - (digit >> 2) * DIGIT_V
        result.append((x, 0.0, DIGIT_WIDTH, DIGIT_HEIGHT, u0, v_top, u0 + DIGIT_U, v_top - DIGIT_V))
        x += DIGIT_WIDTH
    y = float(DIGIT_HEIGHT)
    for v_top, v_bottom in words:
        result.append(((width - WORD_WIDTH) * 0.5, y, WORD_WIDTH, WORD_HEIGHT, 0.0, v_top, 1.0, v_bottom))
        y += WORD_HEIGHT
    return width, height, result


class Entry(object):

    def __init__(self, unit, value, flags, time_ms):
        self.unit = unit
        self.value = value
        self.flags = flags
        self.time_ms = time_ms

    def __repr__(self):
        return 'Entry(%d, flags %d)' % (self.value, self.flags)


class FlyingNumbers(object):
    """
    The numbers in flight; times are in milliseconds of the real time, as timeGetTime of 5e18e0.
    """

    def __init__(self):
        self.entries = []

    def __repr__(self):
        return 'FlyingNumbers(%d)' % len(self.entries)

    def add(self, unit, value, flags, now_ms):
        """
        5e1870 (called through CMapObjectClientSpecific::vf16 5117f0 and 56ce40, which take only the units):
        a number of the unit; an EXP number starts half a second later.
        """
        if unit is None or value < 0:
            return
        start = now_ms + EXP_DELAY_MS if flags & EXP else now_ms
        self.entries.append(Entry(unit, value, flags, start))

    def update(self, now_ms):
        """
        5e18e0: the entries older than their life are forgotten.
        """
        self.entries = [e for e in self.entries if now_ms - e.time_ms <= LIFE_MS]

    def shown(self, now_ms):
        """
        [(entry, t 0..1 of its life)] of the entries to draw now.
        """
        result = []
        for entry in self.entries:
            t = (now_ms - entry.time_ms) / float(LIFE_MS)
            if 0.0 <= t <= 1.0:
                result.append((entry, t))
        return result


def place(center_x, center_y, t, width, height):
    """
    The top left corner of the box of a number at t of its life; y goes down the screen like in game.exe.
    """
    x = center_x + math.trunc(t * DRIFT_X)
    y = center_y - math.trunc(t * DRIFT_Y)
    return x - width * 0.5, y - height * 0.5


def color(entry, of_player):
    if entry.flags & EXP:
        return COLOR_EXP
    return COLOR_PLAYER if of_player else COLOR_OTHER
