"""
The weather of a zone: the rain, the snow and the tornado of "#weather" in map.txt (45ea20).

The rain and the snow are not particles: CWeather of the client (4e3c20 makes it, 4e5680 draws it) fills a grid
of 81 x 81 cells around the eye of the camera, 0.5 apart (the offsets 40, -40, 39, ... -1, 0 of +0x94). Every cell
has one drop whose place in its fall is a hash of the cell plus the time; the drop changes every 200 ticks of
the rain (1000 of the snow) and is drawn only when that number & 0x3f is below the density (+0xbc). A drop of the
rain (4e4240) is a thin triangle: a base 0.066 wide across the view and the top 1 higher, against the wind; it
falls by the wind (WS_WIND_DIR * WS_WIND_STR, 4e41f0) minus 0.5 a tick and stays within 30 of the height of the
eye. A flake of the snow (4e4c80) is a small triangle 0.2 wide falling at 0.1 a tick. The texture is "rainsnow":
the rain takes one of the four streaks of its bottom row (v 0.75..1), the snow a flake of the row above it.
Precipitation computes these triangles with numpy for client.scene.

The tornado keeps the particles of the type RainSnow (CParticle 0x2033, 48eea0/48f300/48f380):
game.exe makes an emitter of the type 0x200f, 0x2031, 0x2032 or 0x2033 (485540) with the texture
"rainsnow" (500a00(6, "RainSnow")); its particles are born around the sources of the emitter, fly by
their speed and fall by the scale of the emitter, their life comes from the frames of the type. The
remake keeps the same shape of the emission with the numbers read from the type: the particles live
around the camera, fall to the ground and are born again above it, so the rain goes on while the
camera moves. The exact shares of the type (the sources around the point, the table of the alpha of
48f300) are not reproduced; the comments mark the reading of the remake.

The weather of a zone is none, rain, snow or tornado (map.txt "#weather", default rain). A cave has no
weather. The sound of the rain and of the tornado is a loop of sfx.res chosen by client.weather.

The kind of the zone is only what may fall there: the server (58ccd0) starts and stops it. A zone opens dry
(595c70 sets +0xe4 to 0); every 64 ticks of the server (58dcd0: (tick & 0x3f) == 3) a counter goes down and,
when it is below 1, a dry sky starts to rain or snow when a random number % 101 is 0, a wet one stops when a
random number % 31 is 0; after a change the counter is 10 (7c2dec). The change goes to the clients (58b750 ->
message 0x70 -> 4e3fc0) with the time of the change: 20 ticks for the rain, 40 for the snow. The client (4e40b0)
turns the density of the cells of the rain (+0xbc, 0..64) and the volume of its loop by (1 - cos(pi t)) / 2.
While it rains, every check of the server throws a lightning at a random point of the map: from 2 to 510 on
both axes, from 40 above the land down to 5 under it, the start moved by up to 2 (58ccd0, 578e30).
A tornado zone has no such cycle: the server makes its whirls (the effect 0x65) around the map instead; the
remake keeps the tornado of the zone on all the time.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

import numpy as np

#------------------------------------------------------------------------------

from engine import particles

#------------------------------------------------------------------------------

NONE = 'none'
RAIN = 'rain'
SNOW = 'snow'
TORNADO = 'tornado'
KINDS = (NONE, RAIN, SNOW, TORNADO)

# the type of CParticle of all the weather (0x2033, case 0x200f/0x2031..0x2033 of 485540)
CODE = 0x2033

# 58dcd0, 58ccd0: the check of the weather of the server and its chances
CHECK_TICKS = 64
CHECK_PHASE = 3
START_CHANCE = 101
STOP_CHANCE = 31
HOLD_CHECKS = 10
# 58ccd0: the ticks of the change of the weather sent to the clients
CHANGE_TICKS = {
    RAIN: 20.0,
    SNOW: 40.0,
}
# 4e40b0: the density of the cells of the rain
LEVELS = 64
# 58ccd0: the lightning of the rain over the map
STRIKE_MIN = 2.0
STRIKE_SPAN = 508.0
STRIKE_SHIFT = 2.0
STRIKE_TOP = 40.0
STRIKE_DEPTH = 5.0

# the texture of the type (500a00(6, "RainSnow"))
TEXTURE_NAME = 'rainsnow'

# 4e3c20: the offsets of the cells of the grid, the far ones first; 4e38a0: the side of a cell
GRID_OFFSETS = tuple(v for n in range(40, 0, -1) for v in (n, -n)) + (0, )
CELL = 0.5
# 4e38c0, 4e38e0, 4e3900: the height of the drops at the middle of their fall and the half of their band
# around the eye of the camera
FALL_MIDDLE = 40.0
BAND = 30.0
# 4e4240 (rain) and 4e4c80 (snow): the hashes of the cells (78dea4, 78df24 and the factors of x and y), the ticks
# of one fall, the fall in a tick added to the wind
RAIN_HASH = (17, 43, 456, 942, 32, 234, 865, 95, 321, 47, 909, 284, 543, 396, 193, 120,
             98, 784, 633, 10, 259, 77, 118, 66, 921, 849, 356, 80, 475, 235, 213, 826)
SNOW_HASH = (5417, 2343, 34456, 12942, 8832, 12234, 45865, 8395, 45321, 8847, 99009, 55284, 76543, 45396,
             12193, 98120, 6798, 79784, 11633, 5710, 54259, 3477, 78118, 1266, 97821, 80949, 35786, 8780,
             78475, 72235, 34213, 92826)
PRECIPITATION = {
    RAIN: {'hash': RAIN_HASH, 'fx': 0xac, 'fy': 0xd, 'period': 200.0, 'fall': 0.5},
    SNOW: {'hash': SNOW_HASH, 'fx': -0x7570, 'fy': 0x45b3, 'period': 1000.0, 'fall': 0.1},
}
# the shift of a drop in its cell (+-1 cell) and in height (+-10)
DROP_SHIFT_Z = 10.0
# 4e4240: the half of the base of a drop and its height; 4e4c80 (7aed88): the side of a flake
DROP_HALF_WIDTH = 0.033
DROP_LENGTH = 1.0
FLAKE_SIZE = 0.2
# the sway of a flake: game.exe follows a spline of 128 random knots (4e3980, 4e5660); the remake a sine
FLAKE_SWAY = 0.3

# the radii and the heights of the volume around the camera, in units of the world
RADIUS = 22.0
HEIGHT = 24.0
GROUND_MARGIN = 0.5

# the numbers of the particles (the remake's reading: enough for a dense rain at 0.5 units per tick)
COUNTS = {
    RAIN: 1300,
    SNOW: 900,
    TORNADO: 700,
}

# the speeds of the particles for one tick of the world (55 ms)
RAIN_FALL = 1.35
SNOW_FALL = 0.16
TORNADO_RISE = 0.10
TORNADO_TURN = 0.07
WIND_SHARE = 0.35
SNOW_SWAY = 0.05

# the half of the side of the square of a particle (4976b0 reads the size) and the alpha of its color
SIZES = {
    RAIN: 0.9,
    SNOW: 0.38,
    TORNADO: 0.40,
}
COLORS = {
    RAIN: 0xa0c8d8ff,
    SNOW: 0xd0ffffff,
    TORNADO: 0x60c8c8c8,
}

# the frames of the 4 x 4 atlas of the type: the rows from the bottom are the rain streaks, the
# snowflakes, the dust and the colour puffs (frame_uv counts the rows from the bottom)
FRAMES = {
    RAIN: (0, 1, 2, 3),
    SNOW: (4, 5, 6, 7),
    TORNADO: (8, 9, 10, 11),
}

#------------------------------------------------------------------------------


class WeatherEmitter(particles.Emitter):
    """
    The emitter of the rain, the snow or the tornado: the particles are born in the cylinder around
    the point of the camera and come back above it when they reach the ground.
    """

    def __init__(self, kind, position, rng=None, ground=None, count=None):
        self.kind = kind if kind in (RAIN, SNOW, TORNADO) else RAIN
        self.count = COUNTS[self.kind] if count is None else int(count)
        super(WeatherEmitter, self).__init__(CODE, position, rng=rng, ground=ground)
        self.fill()

    def setup(self, code):
        """
        The numbers of the type: the emitter only lives by its particles, it emits nothing after the fill.
        """
        self.chance = 100
        self.least = 0
        self.distance = 0.0
        self.most = self.count
        self.frames = 16
        self.frame_delay = 0
        self.life = -1
        self.spread = [0.0] * 6
        self.offset = 0.0
        self.scale = 1.0
        self.size = SIZES.get(self.kind, SIZES[RAIN])
        self.lift = 0.0
        self.fade = 0.0
        # the density 0..LEVELS of the cells which are drawn and the cell of each particle (by its id)
        self.level = LEVELS
        self.cells = {}
        self.additive = False
        self.texture_name = TEXTURE_NAME
        self.born, self.moves = self.born_weather, self.move_weather
        self.emitting = False

    def fill(self):
        for index in range(self.most):
            self.emit(index)

    def born_weather(self, p, index):
        # the cell of the particle: it is drawn while the cell is below the density (4e4240)
        self.cells[id(p)] = index % LEVELS
        p.x, p.y, p.z = self.birth_point()
        p.vx, p.vy, p.vz = self.birth_speed()
        p.size = self.shown_size(p)
        p.frame = self.birth_frame()
        p.life = 0
        p.color = COLORS.get(self.kind, COLORS[RAIN])
        return True

    def shown_size(self, p):
        return self.size if self.cells.get(id(p), 0) < self.level else 0.0

    def birth_point(self):
        """
        A point of the cylinder of the emitter; the tornado keeps its radius around the centre.
        """
        rng = self.rng
        angle = rng.random() * 2.0 * math.pi
        radius = math.sqrt(rng.random()) * RADIUS
        cx, cy, cz = self.current
        x = cx + math.cos(angle) * radius
        y = cy + math.sin(angle) * radius
        ground = self.ground(x, y)
        if self.kind == TORNADO:
            z = ground + rng.random() * HEIGHT * 0.5
        else:
            z = ground + HEIGHT * (0.35 + 0.65 * rng.random())
        return x, y, z

    def birth_speed(self):
        rng = self.rng
        wind_x, wind_y = self.wind_xy()
        if self.kind == SNOW:
            return (wind_x * WIND_SHARE + (rng.random() - 0.5) * SNOW_SWAY,
                    wind_y * WIND_SHARE + (rng.random() - 0.5) * SNOW_SWAY,
                    -SNOW_FALL)
        if self.kind == TORNADO:
            return wind_x * 0.1, wind_y * 0.1, 0.0
        return (wind_x * WIND_SHARE + (rng.random() - 0.5) * 0.08,
                wind_y * WIND_SHARE + (rng.random() - 0.5) * 0.08,
                -RAIN_FALL * (0.75 + 0.25 * rng.random()))

    def wind_xy(self):
        """
        The wind of the scene (WS_WIND_DIR and WS_WIND_STR). The emitter has no link to the scene, so the
        client gives the direction in self.wind (x, y, strength, 0).
        """
        if len(self.wind) >= 3 and (self.wind[0] != 0.0 or self.wind[1] != 0.0):
            return self.wind[0] * self.wind[2], self.wind[1] * self.wind[2]
        return 0.0, 0.0

    def move_weather(self, p):
        p.x += p.vx
        p.y += p.vy
        p.z += p.vz
        if self.kind == TORNADO:
            self.move_tornado(p)
        ground = self.ground(p.x, p.y)
        cx, cy, _cz = self.current
        if (p.z <= ground + GROUND_MARGIN or p.z > ground + HEIGHT or
                math.hypot(p.x - cx, p.y - cy) > RADIUS):
            self.respawn(p)
        p.size = self.shown_size(p)
        return True

    def birth_frame(self):
        frames = FRAMES.get(self.kind, FRAMES[RAIN])
        return frames[self.rng.randrange(len(frames))]

    def move_tornado(self, p):
        """
        The tornado: the particles turn around the centre and rise (the remake's reading).
        """
        cx, cy, cz = self.current
        dx, dy = p.x - cx, p.y - cy
        radius = math.hypot(dx, dy) or 1.0
        turn = TORNADO_TURN * (0.4 + 0.6 * radius / RADIUS)
        p.x += -dy / radius * turn
        p.y += dx / radius * turn
        p.z += TORNADO_RISE

    def respawn(self, p):
        """
        The particle is born again above the camera; the last state is dropped so the drawing does not
        stretch it from the old place.
        """
        p.x, p.y, p.z = self.birth_point()
        p.vx, p.vy, p.vz = self.birth_speed()
        p.size = self.shown_size(p)
        p.frame = self.birth_frame()
        p.color = COLORS.get(self.kind, COLORS[RAIN])
        p.keep()

#------------------------------------------------------------------------------


class Precipitation(object):
    """
    The rain or the snow of CWeather (4e4240, 4e4c80): the triangles of the drops of the grid around the eye.
    """

    def __init__(self, kind, level=LEVELS):
        self.kind = kind
        self.numbers = PRECIPITATION[kind]
        self.level = level
        # the wind of the map: WS_WIND_DIR * WS_WIND_STR (4e41f0)
        self.wind = (0.0, 0.0, 0.0)
        self.texture_name = TEXTURE_NAME
        offsets = np.asarray(GRID_OFFSETS, dtype=np.int64)
        # the outer loop goes by y, the inner by x: the far cells come first
        self.offset_y, self.offset_x = [a.ravel() for a in np.meshgrid(offsets, offsets, indexing='ij')]
        self.across_y = np.abs(self.offset_y) < np.abs(self.offset_x)
        self.hash_table = np.asarray(self.numbers['hash'], dtype=np.int64)

    def __repr__(self):
        return 'Precipitation(%s, level %d)' % (self.kind, self.level)

    def drops(self, eye, time):
        """
        The drops shown now: x, y, z of their bottoms, the numbers of their falls and the across_y flags.
        """
        numbers = self.numbers
        wx, wy, wz = self.wind
        vz = wz - numbers['fall']
        ex, ey, ez = (float(v) for v in eye)
        lead = (ez - FALL_MIDDLE) / vz
        grid_x = int(round((ex - wx * lead) / CELL))
        grid_y = int(round((ey - wy * lead) / CELL))
        ix = self.offset_x + grid_x
        iy = self.offset_y + grid_y
        cell_hash = (self.hash_table[(ix * ix + iy) & 0x1f] + ix * numbers['fx'] + iy * numbers['fy']) & 0xffff
        value = cell_hash + float(time)
        period = numbers['period']
        fall = np.round(value / period).astype(np.int64)
        shown = (fall & 0x3f) < self.level
        if not shown.any():
            return None
        ix, iy, value, fall, across = ix[shown], iy[shown], value[shown], fall[shown], self.across_y[shown]
        phase = value - fall * period
        z = phase * vz + FALL_MIDDLE
        low, high = max(ez - BAND, 0.0), ez + BAND
        if self.kind == RAIN:
            z = low + np.mod(z - low, high - low)
            keep = np.ones(len(z), dtype=bool)
        else:
            keep = (z >= low) & (z <= high)
        x = ((fall * -0x3b & 0xff) - 128) * CELL / 128.0 + phase * wx + ix * CELL
        y = ((fall * 0x43 & 0xff) - 128) * CELL / 128.0 + phase * wy + iy * CELL
        z = z + ((fall * -0x7f & 0xff) - 128) * DROP_SHIFT_Z / 128.0
        if self.kind == SNOW:
            x = x + np.sin(phase * 0.05 + ix) * FLAKE_SWAY
            y = y + np.cos(phase * 0.04 + iy) * FLAKE_SWAY
        return x[keep], y[keep], z[keep], fall[keep], across[keep]

    def triangles(self, eye, time, right=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
        """
        The vertices (n * 3, 9) of the triangles in the format of the particles: x, y, z, u, v, r, g, b, a.
        right and up are the axes of the view for the flakes of the snow.
        """
        drops = self.drops(eye, time)
        if drops is None or len(drops[0]) == 0:
            return np.zeros((0, 9), dtype=np.float32)
        x, y, z, fall, across = drops
        count = len(x)
        out = np.ones((count, 3, 9), dtype=np.float32)
        column = (fall & 3) * 0.25
        if self.kind == RAIN:
            wx, wy, wz = self.wind
            vz = wz - self.numbers['fall']
            speed = math.sqrt(wx * wx + wy * wy + vz * vz) or 1.0
            dx = np.where(across, 0.0, DROP_HALF_WIDTH)
            dy = np.where(across, DROP_HALF_WIDTH, 0.0)
            out[:, 0, 0], out[:, 0, 1], out[:, 0, 2] = x + dx, y + dy, z
            out[:, 1, 0], out[:, 1, 1], out[:, 1, 2] = x - dx, y - dy, z
            out[:, 2, 0] = x - wx / speed * DROP_LENGTH
            out[:, 2, 1] = y - wy / speed * DROP_LENGTH
            out[:, 2, 2] = z + DROP_LENGTH
            v_base, v_top = 0.75, 1.0
        else:
            right = np.asarray(right, dtype=np.float64) * FLAKE_SIZE
            up = np.asarray(up, dtype=np.float64) * FLAKE_SIZE
            center = np.stack((x, y, z), axis=1)
            # 4e4c80: the corners -0.5 and +0.5 of the side across, -0.3 and +0.7 of it up
            out[:, 0, 0:3] = center - right * 0.5 - up * 0.3
            out[:, 1, 0:3] = center + right * 0.5 - up * 0.3
            out[:, 2, 0:3] = center + up * 0.7
            v_base, v_top = 0.5, 0.74
        out[:, 0, 3], out[:, 0, 4] = column, v_base
        out[:, 1, 3], out[:, 1, 4] = column + 0.25, v_base
        out[:, 2, 3], out[:, 2, 4] = column + 0.125, v_top
        return out.reshape(count * 3, 9)


class Weather(object):
    """
    The weather of a map: the kind of "#weather", the drops of the rain or the snow (Precipitation) or the
    particles of the tornado (none for the calm).
    """

    def __init__(self, kind, position, rng=None, ground=None, level=LEVELS):
        self.kind = kind if kind in KINDS else NONE
        self.rng = rng or random.Random()
        self.emitters = []
        self.precipitation = None
        if self.kind in PRECIPITATION:
            self.precipitation = Precipitation(self.kind, level=level)
        elif self.kind != NONE:
            emitter = WeatherEmitter(self.kind, position, rng=self.rng, ground=ground)
            emitter.level = level
            self.emitters.append(emitter)

    def set_level(self, level):
        if self.precipitation is not None:
            self.precipitation.level = level
        for emitter in self.emitters:
            emitter.level = level

    def __repr__(self):
        return 'Weather(%s, %d emitters)' % (self.kind, len(self.emitters))

    @property
    def active(self):
        return bool(self.emitters) or self.precipitation is not None

    def update(self, position):
        """
        One tick of the world: the emitters follow the camera and their particles move.
        """
        for emitter in self.emitters:
            emitter.update(position)


class Schedule(object):
    """
    The rain or the snow of a zone over time, as the server of game.exe runs it (58ccd0).
    """

    def __init__(self, zone_kind, rng=None):
        self.zone_kind = zone_kind if zone_kind in KINDS else NONE
        self.rng = rng or random.Random()
        # what falls now: NONE, or the kind of the zone
        self.falling = NONE
        self.counter = 0
        self.ticks = 0
        # the tick of the change and its length (the fade of the client)
        self.change_tick = 0
        self.change_ticks = 0.0

    def __repr__(self):
        return 'Schedule(%s, falling %s)' % (self.zone_kind, self.falling)

    @property
    def cycles(self):
        return self.zone_kind in (RAIN, SNOW)

    def tick(self):
        """
        One tick of the server. Returns the points of the lightning (start, end) thrown on this tick
        and whether what falls changed.
        """
        self.ticks += 1
        if self.ticks % CHECK_TICKS != CHECK_PHASE or not self.cycles:
            return [], False
        changed = self.check()
        strikes = []
        if self.falling == RAIN:
            strikes.append(self.strike_points())
        return strikes, changed

    def check(self):
        self.counter -= 1
        if self.counter >= 1:
            return False
        if self.falling == NONE:
            if self.rng.randrange(START_CHANCE) != 0:
                return False
            self.change(self.zone_kind)
        else:
            if self.rng.randrange(STOP_CHANCE) != 0:
                return False
            self.change(NONE)
        self.counter = HOLD_CHECKS
        return True

    def change(self, falling):
        self.falling = falling
        self.change_tick = self.ticks
        self.change_ticks = CHANGE_TICKS.get(self.zone_kind, CHANGE_TICKS[RAIN])

    def level(self):
        """
        4e40b0: the density 0..LEVELS of what falls, turned by (1 - cos(pi t)) / 2 during a change.
        """
        if self.change_ticks <= 0.0:
            return LEVELS if self.falling != NONE else 0
        t = (self.ticks - self.change_tick) / self.change_ticks
        if t >= 1.0:
            return LEVELS if self.falling != NONE else 0
        wave = math.cos(t * math.pi)
        part = (1.0 - wave) * 0.5 if self.falling != NONE else (1.0 + wave) * 0.5
        return int(round(part * LEVELS))

    def strike_points(self):
        """
        58ccd0: a point of the map and the start of the bolt above it (x, y and the shift of z from the land).
        """
        rng = self.rng
        x = rng.random() * STRIKE_SPAN + STRIKE_MIN
        y = rng.random() * STRIKE_SPAN + STRIKE_MIN
        start_x = x + rng.random() * 2.0 * STRIKE_SHIFT - STRIKE_SHIFT
        start_y = y + rng.random() * 2.0 * STRIKE_SHIFT - STRIKE_SHIFT
        return (start_x, start_y, STRIKE_TOP - STRIKE_DEPTH), (x, y, -STRIKE_DEPTH)

#------------------------------------------------------------------------------


def from_zone(zone):
    """
    The kind of the weather of a zone of map.txt, the calm for a cave or a zone without it.
    """
    if zone is None:
        return NONE
    if zone.is_cave:
        return NONE
    return zone.weather if zone.weather in KINDS else RAIN
