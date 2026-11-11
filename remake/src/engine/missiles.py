"""
The arrows of the bows and the crossbows, as game.exe flies them.

54f450 delivers a blow of the server: with a bow or a crossbow in the hands (the weapon types 5 and 6) it does
not give the packet of the blow (54f630) to the target at once, but makes the effect 0x5b, CEffectArrowFigure:
at the point of the shooter (vcall 0xc0 of the unit), the id of the target at +0x150, of the shooter at +0x15c,
and the 15 values of the packet at +0x160.

The flight (67aa70, every tick on the server and on the client): the arrow homes on the target, there is
no ballistics. The velocity +0x98 is the way to the point of the target - its position, raised by +0x130 and half
of +0x14c of the unit - scaled so that its horizontal part is the step 0.667 (7433d4), twice that for 0x5b,
so the arrow of a bow flies 1.333 map units per tick (24 per second). When the horizontal distance is less
than the step, the arrow has come: the server (vf06, 67bde0) gives the packet to the target (vcall 0x7c of
the unit, as a blow of the hand) and the effect ends. On the client the arrow does not go lower than 1 above
the ground: the vertical step becomes (ground + 1 - z) - min(that, 1) / 2. When the target is gone,
the arrow stops where it is.

The look (vf23, 67bee0; vf24, 67beb0): the figure "efar1" (efar1.lnk, the part efar1arrow.fig of figures.res)
with the texture "arrow01", put at the position of the arrow and turned along its velocity.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

from eilib import adb

#------------------------------------------------------------------------------

# 7433d4, doubled for the effect 0x5b
ARROW_STEP = 2.0 * 0.6666667

# 7433d8: the lowest height of an arrow above the ground
MIN_HEIGHT = 1.0

ARROW_MODEL = 'efar1'
ARROW_TEXTURE = 'arrow01'

# 54f450: the weapon types which shoot (adb.WEAPONS)
SHOOTING_WEAPONS = (adb.WEAPONS.index('BOW'), adb.WEAPONS.index('CROSSBOW'))

# the range of a shooter whose weapon item is not known: the short bow of items.idb (the remake's choice)
DEFAULT_RANGE = 12.0

#------------------------------------------------------------------------------


class Arrow(object):
    """
    An arrow in flight: the shooter, the target (engine.unit.Unit), the blow it carries (engine.combat.Hit).
    ground_height(x, y) gives the height of the land, None for no land.
    """

    def __init__(self, shooter, target, hit, position, ground_height=None, step=ARROW_STEP, model_name=ARROW_MODEL):
        self.shooter = shooter
        # map units a tick, the missile of a spell (CEffectArrow 0x4e) flies at half the step of an arrow
        self.step = step
        # the figure of the missile, None for the missile of a spell: its particles are its look
        self.model_name = model_name
        # engine.particles.Emitter behind the missile of a spell and (type, scale) of the flash where it hits
        self.trail = None
        self.flash = None
        # the spell of a missile of magic, for its light where it hits
        self.spell = None
        # engine.magic.SpellSound of a missile of a spell: the ones to play, the one which flies with it
        self.sounds = []
        self.loop = None
        self.target = target
        self.hit = hit
        self.x, self.y, self.z = position
        self.velocity = (0.0, 0.0, 0.0)
        self.ground_height = ground_height
        self.arrived = False
        self.update_velocity(1.0)

    def __repr__(self):
        return 'Arrow(%.1f,%.1f,%.1f -> %s)' % (self.x, self.y, self.z, self.target)

    def update_velocity(self, ticks):
        """
        67aa70: the velocity for the next ticks, the arrow comes when the target is nearer than the step.
        """
        step = self.step * ticks
        target = self.target
        if target is None:
            self.velocity = (0.0, 0.0, 0.0)
            self.arrived = True
            return
        tx, ty, tz = target_point(target)
        dx, dy, dz = tx - self.x, ty - self.y, tz - self.z
        horizontal = math.hypot(dx, dy)
        if horizontal < step:
            self.velocity = (0.0, 0.0, 0.0)
            self.arrived = True
            return
        k = step / horizontal
        vz = dz * k
        if self.ground_height is not None:
            ground = self.ground_height(self.x, self.y)
            if ground is not None and self.z < ground + MIN_HEIGHT:
                lift = ground + MIN_HEIGHT - self.z
                vz = lift - min(lift, MIN_HEIGHT) * 0.5
        self.velocity = (dx * k, dy * k, vz)

    def update(self, ticks):
        """
        One step of the flight (vf06): the velocity, then the move. True while the arrow flies.
        """
        if self.arrived:
            return False
        self.update_velocity(ticks)
        vx, vy, vz = self.velocity
        self.x += vx
        self.y += vy
        self.z += vz
        return not self.arrived

    def yaw(self):
        """
        The turn of the figure around z: the arrow lies along y and looks at -y like the units.
        """
        vx, vy, _ = self.velocity
        return math.atan2(vx, -vy)

    def pitch(self):
        vx, vy, vz = self.velocity
        return math.atan2(vz, math.hypot(vx, vy))

    def world_matrix(self):
        """
        vf24: the figure at the position of the arrow, its -y along the velocity.
        """
        yaw, pitch = self.yaw(), self.pitch()
        c, s = math.cos(yaw), math.sin(yaw)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # the rotation around x by -pitch raises the -y of the figure to +z
        cp, sp = math.cos(-pitch), math.sin(-pitch)
        tilt = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        m = np.identity(4)
        m[0:3, 0:3] = turn @ tilt
        m[0:3, 3] = (self.x, self.y, self.z)
        return m

#------------------------------------------------------------------------------


def target_point(unit):
    """
    The point an arrow flies to: the position of the unit raised by half of its height (+0x14c).
    """
    return unit.x, unit.y, unit.z + unit.height() * 0.5


def is_shooting(weapon_code):
    return weapon_code in SHOOTING_WEAPONS
