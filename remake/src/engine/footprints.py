"""
The footprints of the units (56d4c0, the decal objects of the type 0x4d and the textures footprints1..4).

game.exe makes a footprint decal when the particles of a step of a unit reach the ground (56d4c0 takes
the record of the tile type from the table 7c89f8), so the footprints stay on the land where the unit
walks. The remake has no world decals; this module keeps the prints of the steps of the units itself:
a print every 0.55 units of the way, the left and the right foot in turn, the texture "footprints"
(a 128 x 128 atlas of 8 x 8 prints) and a life of 300 ticks.

The exact records of the table of the tile types and the life of the decals of game.exe are not
decoded: the table is built from a "footprints" registry which is not in the files of the game.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

#------------------------------------------------------------------------------

# the print of a step: the way between the prints, the offset of a foot to the side and forward
STEP_DISTANCE = 0.55
SIDE_OFFSET = 0.13
FORWARD_OFFSET = 0.10
FOOTPRINT_SIZE = 0.34
# how long a print stays and fades at its end
FOOTPRINT_LIFE = 300
FOOTPRINT_FADE = 60
# the atlas of the prints and its grid
TEXTURE_NAME = 'footprints'
ATLAS_COLUMNS = 8

#------------------------------------------------------------------------------


class Footprint(object):

    def __init__(self, x, y, z, angle, cell, life=FOOTPRINT_LIFE):
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle
        self.cell = cell
        self.life = life
        self.age = 0

    def __repr__(self):
        return 'Footprint(%.1f, %.1f, cell %d)' % (self.x, self.y, self.cell)

    @property
    def alpha(self):
        left = self.life - self.age
        if left <= 0:
            return 0.0
        return min(1.0, left / float(FOOTPRINT_FADE))


class FootprintField(object):
    """
    The prints of the units of a map: it remembers the way of every unit and puts the prints on the land.
    """

    def __init__(self, ground=None, rng=None, life=FOOTPRINT_LIFE):
        self.ground = ground or (lambda x, y: 0.0)
        self.rng = rng or random.Random()
        self.life = life
        self.prints = []
        self.tracks = {}
        self.foot = {}

    def __repr__(self):
        return 'FootprintField(%d prints)' % len(self.prints)

    def clear(self):
        self.prints = []
        self.tracks = {}
        self.foot = {}

    def update(self, units):
        """
        One tick: the units which walked far enough leave a print; the old prints go away.
        """
        for unit in units:
            if unit is None or unit.fighter is None or unit.is_dead:
                continue
            key = id(unit)
            previous = self.tracks.get(key)
            if previous is None:
                self.tracks[key] = (unit.x, unit.y)
                continue
            distance = math.hypot(unit.x - previous[0], unit.y - previous[1])
            self.tracks[key] = (unit.x, unit.y)
            if distance <= 0.0 or distance > 5.0:
                # a turn on the place or a teleport, not a step
                continue
            walked = self.foot.get(key, 0.0) + distance
            while walked >= STEP_DISTANCE:
                walked -= STEP_DISTANCE
                self.spawn(unit)
            self.foot[key] = walked
        alive = []
        for print_ in self.prints:
            print_.age += 1
            if print_.age < print_.life:
                alive.append(print_)
        self.prints = alive

    def spawn(self, unit):
        """
        One print at the foot of the unit, the left and the right in turn.
        """
        left = not self.foot.get(('step', id(unit)), False)
        self.foot[('step', id(unit))] = left
        side = -SIDE_OFFSET if left else SIDE_OFFSET
        forward = FORWARD_OFFSET if left else -FORWARD_OFFSET
        yaw = getattr(unit, 'yaw', 0.0)
        # the unit looks along (sin(yaw), -cos(yaw)) like engine.unit.world_matrix
        fx, fy = math.sin(yaw), -math.cos(yaw)
        rx, ry = -fy, fx
        x = unit.x + rx * side + fx * forward
        y = unit.y + ry * side + fy * forward
        z = self.ground(x, y)
        angle = yaw + self.rng.uniform(-0.12, 0.12)
        cell = self.rng.randrange(ATLAS_COLUMNS * ATLAS_COLUMNS)
        print_ = Footprint(x, y, z, angle, cell, self.life)
        self.prints.append(print_)
        if len(self.prints) > 800:
            self.prints = self.prints[-800:]
        return print_


def cell_uv(cell, columns=ATLAS_COLUMNS):
    """
    (u0, v0, u1, v1) of a print in the atlas, v from the bottom as the particles of the remake.
    """
    column, row = cell % columns, cell // columns
    step = 1.0 / columns
    return (column * step, 1.0 - (row + 1) * step, (column + 1) * step, 1.0 - row * step)
