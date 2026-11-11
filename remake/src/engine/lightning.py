"""
The lightning bolts of game.exe (CLightning of the client, 0x466000-0x468a00).

The server effect CEffectLightning (object 0x4d) keeps the two ends (+0x14c, +0x158), the power (+0x164)
and the demon flag (+0x174, 67a670/67a880); on the client its CLightning (+0x168) is made by 4680d0 and,
while the effect follows a unit and an object (67a850: the hand of the caster by 532860, the target 1 above
its feet), the ends move every tick (vf06 67a8b0 -> 467e10).

A bolt (records of 0xa8 bytes: the main one at +0x5c, six branches at +0x104) is a path of points (466820):
    - the step from the start to the end is cut in 48 segments when it is shorter than 10, round(length * 5)
      cleared of the low two bits (4..400) up to 15, 72 above 15 (the length then counts as 15);
    - two axes across it, a = (d.y + 0.0666 d.z, 0.013 d.z - d.x, -0.0666 d.x - 0.013 d.y) and b = a x d,
      both as long as the length;
    - point i = start + i * step + noise_a[i] * a + noise_b[i] * b (466ce0), the noise less its straight
      line from the first to the last point, so the ends stay on the path; only the count of points is
      drawn, the last one is one step short of the end.

The noise (4662c0) is read from gfx.res "lightning.dat", 32 rows of 400 floats: the rows 0..15 are slow
waves, 16..31 small ripples. A record (466190) is two random rows (rnd & 15) with the weights of a random
direction of the plane times the jaggedness. A bolt keeps three records for each axis and a time (+0x90);
the time of the world in ticks / 4 adds a new record each whole step, the three are mixed by the quadratic
B-spline of the part of the step (0.5 t^2, 0.5 + t - t^2, 0.5 (1 - t)^2), and a new record of the ripples
(rows + 16, weight 1) is added at every call, so the bolt crawls slowly and trembles at each drawing.

The power (the +0x50c of the bolt) sets the kind (4680d0):
    - above 0 (the lightning arc, 5 * the visual strength): the jaggedness 1.5, the width = the power,
      branches from 0.2, 0.3, 0.5, 0.8, 0.9, 0.9 of the way (counted from the end) to the end, their width
      (rnd * 0.8 + 0.7) * power / 2; the ends of the branches sway along the axes of the main bolt by
      sin(time * rate / 4) of the table 787c8c times the part of the way and +0x510 = 0.05 (4672e0);
      the noise is renewed at every drawing, the texture band is random (rnd & 3), a glowing ball is drawn
      at the end;
    - 0 (the lightning wall, the storm): the jaggedness 2, the width 4, six copies of the bolt moved by a
      random step up to 0.05 of the length in each axis, widths rnd * 0.8 * 3.5 + 2.45; balls at both ends;
    - below 0 (the invoke lightning, -7 * the strength, from 30 above the ground): the jaggedness 1, the
      width -1.5 * power, +0x510 = 0.3 (0.05 when the height is 30), the shape is made once, no ball and
      the texture is drawn by its first column only.

The drawing (467ad0): every bolt is a band of its points facing the camera, the texture "Lightning"
("DemonLightning" for the demons) has four bands, the band is chosen by u along the way (the length * 0.6)
and v of a quarter; the ball (467590) is 8 squares of "SmtgLight" ("DemonLight") from the point toward the
camera, i / 8 of the size away with the size sqrt(1 - (i / 8)^2) of |end - start| * 0.11 (0.5..1.2 for the
arc), the alpha 0xa0, the frames of the 4 x 4 atlas going by 2.

The random numbers of game.exe come from its table (7a9d38); here a random.Random gives them.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import random

import numpy

#------------------------------------------------------------------------------

# gfx.res "lightning.dat": the rows of the noise, the slow ones and then the ripples
NOISE_FILE_NAME = 'lightning.dat'
NOISE_ROWS = 32
NOISE_POINTS = 400
RIPPLE_ROW = 16

# 4672e0: the time of the noise is the time of the world in ticks / 4
NOISE_TICKS = 4.0
# 4662c0: the most steps of the noise which one call catches up
NOISE_MOST_STEPS = 100.0

# 466820: the segments of a path
SHORT_LENGTH = 10.0
LONG_LENGTH = 15.0
SHORT_SEGMENTS = 48
LONG_SEGMENTS = 72
SEGMENTS_PER_UNIT = 5.0
LEAST_SEGMENTS = 4
MOST_SEGMENTS = 400
AXIS_TILT_A = -0.0666
AXIS_TILT_B = 0.013

BRANCHES = 6
# 4680d0: where the branches of a bolt with power start, the part of the way from the end
BRANCH_FRACTIONS = (0.2, 0.3, 0.5, 0.8, 0.9, 0.9)
# 787c8c: the rates of the sway of the ends of the branches along the two axes
BRANCH_SWAY_RATES = ((2.0, 3.0), (2.5, 1.8), (0.3, 0.8), (1.2, 1.3), (3.0, 5.1), (2.1, 1.76))
SWAY_RATE_SCALE = 0.25

# +0x510: the sway of the branches (0.1 by vf02 before 4680d0)
POWER_SWAY = 0.05
NEGATIVE_SWAY = 0.3
STORM_HEIGHT = 30.0

# the jaggedness of the noise by the kind
POWER_JAGGEDNESS = 1.5
ZERO_JAGGEDNESS = 2.0
NEGATIVE_JAGGEDNESS = 1.0

# the widths
NEGATIVE_WIDTH_SCALE = 1.5
ZERO_WIDTH = 4.0
ZERO_BRANCH_WIDTH_RANGE = 0.8 * 3.5
ZERO_BRANCH_WIDTH_LEAST = 2.45
POWER_BRANCH_WIDTH_RANGE = 0.8
POWER_BRANCH_WIDTH_LEAST = 0.7
ZERO_BRANCH_OFFSET = 0.05

# 467ad0: the texture along the way, its bands
TEXTURE_LENGTH_SCALE = 0.6
TEXTURE_BANDS = 4
BAND_HEIGHT = 0.25

# 467590: the ball of light
BALL_SIZE_SCALE = 0.11
BALL_LEAST = 0.5
BALL_MOST = 1.2
BALL_SQUARES = 8
BALL_ALPHA = 0xa0
BALL_FRAMES = 15

TEXTURES = ('Lightning', 'SmtgLight')
DEMON_TEXTURES = ('DemonLightning', 'DemonLight')

# the remake's reading: the width of a band in the units of the world is the width of game.exe / 10
# (its renderer takes it in the dll; the arc of the plain spell is then 0.4 wide)
WIDTH_SCALE = 0.1

#------------------------------------------------------------------------------

_noise_table = numpy.zeros((NOISE_ROWS, NOISE_POINTS), dtype=numpy.float32)

#------------------------------------------------------------------------------


def load_noise(data):
    """
    Takes the bytes of gfx.res "lightning.dat" (468a00 reads it into 79ce68).
    """
    global _noise_table
    table = numpy.frombuffer(bytes(data), dtype='<f4')
    if table.size < NOISE_ROWS * NOISE_POINTS:
        return False
    _noise_table = table[:NOISE_ROWS * NOISE_POINTS].reshape(NOISE_ROWS, NOISE_POINTS).astype(numpy.float32)
    return True


def noise_table():
    return _noise_table


def random_record(rng, scale):
    """
    466190: two rows of the slow noise and the weights of a random direction of the plane times the scale.
    """
    rows = []
    weights = []
    for _ in range(2):
        rows.append(rng.getrandbits(32) & 0xf)
        weights.append(rng.getrandbits(32) * 2.0 ** -31 - 1.0)
    length = math.sqrt(weights[0] * weights[0] + weights[1] * weights[1])
    if length > 0.0:
        weights = [w / length * scale for w in weights]
    return (rows[0], rows[1], weights[0], weights[1])


def segments_of(length):
    """
    466820: the count of the segments of a path and the length its axes get.
    """
    if length < SHORT_LENGTH:
        count = SHORT_SEGMENTS
    elif length > LONG_LENGTH:
        length = LONG_LENGTH
        count = LONG_SEGMENTS
    else:
        count = min(int(round(length * SEGMENTS_PER_UNIT)) & ~3, MOST_SEGMENTS)
    return max(count, LEAST_SEGMENTS), length


def scaled(vector, length):
    size = float(numpy.sqrt(numpy.dot(vector, vector)))
    if size <= 0.0:
        return numpy.zeros(3)
    return vector * (length / size)

#------------------------------------------------------------------------------


class Bolt(object):
    """
    A record of 0xa8 bytes of CLightning: a path, its noise and its width.
    """

    def __init__(self):
        # [0..2] the start, [3..5] the axis a, [6..8] the axis b, [9..11] the step
        self.start = numpy.zeros(3)
        self.side = numpy.zeros(3)
        self.up = numpy.zeros(3)
        self.step = numpy.zeros(3)
        # +0x98: the count of the points
        self.count = LEAST_SEGMENTS
        # +0x30..+0x5c and +0x60..+0x8c: three records of each axis, the newest first
        self.records_a = [(0, 0, 0.0, 0.0)] * 3
        self.records_b = [(0, 0, 0.0, 0.0)] * 3
        # +0x90: the time of the newest records
        self.time = 0.0
        # +0xa4
        self.width = 0.0
        self.points = numpy.zeros((self.count, 3))

    def __repr__(self):
        return 'Bolt(%d points, width %.2f)' % (self.count, self.width)

    @property
    def end(self):
        return self.start + self.step * self.count

    def aim(self, start, end):
        """
        466820: the path from the start to the end.
        """
        self.start = numpy.asarray(start, dtype=numpy.float64)
        d = numpy.asarray(end, dtype=numpy.float64) - self.start
        self.count, length = segments_of(float(numpy.sqrt(numpy.dot(d, d))))
        a = numpy.array((d[1] - d[2] * AXIS_TILT_A, d[2] * AXIS_TILT_B - d[0],
                         d[0] * AXIS_TILT_A - d[1] * AXIS_TILT_B))
        b = numpy.cross(a, d)
        self.side = scaled(a, length)
        self.up = scaled(b, length)
        self.step = d / self.count
        if len(self.points) != self.count:
            self.points = numpy.resize(self.points, (self.count, 3))

    def seed(self, rng, scale, time):
        """
        4680d0: the time and the six first records.
        """
        self.time = time
        self.records_a = [random_record(rng, scale) for _ in range(3)]
        self.records_b = [random_record(rng, scale) for _ in range(3)]

    def noise(self, rng, time, scale):
        """
        4662c0: the noise of both axes at the time.
        """
        t = time - self.time
        if t > NOISE_MOST_STEPS:
            t = NOISE_MOST_STEPS
        if t > 1.0:
            while True:
                self.records_a = [random_record(rng, scale)] + self.records_a[:2]
                self.records_b = [random_record(rng, scale)] + self.records_b[:2]
                self.time += 1.0
                t -= 1.0
                if not t > 1.0:
                    break
        mix = (0.5 * t * t, 0.5 + t - t * t, 0.5 - t + 0.5 * t * t)
        table = _noise_table
        count = min(self.count, NOISE_POINTS)
        result = []
        for records in (self.records_a, self.records_b):
            values = numpy.zeros(count)
            for (row0, row1, w0, w1), weight in zip(records, mix):
                values += table[row0, :count] * (w0 * weight) + table[row1, :count] * (w1 * weight)
            row0, row1, w0, w1 = random_record(rng, scale)
            values += table[RIPPLE_ROW + row0, :count] * w0 + table[RIPPLE_ROW + row1, :count] * w1
            result.append(values)
        return result

    def shape(self, rng, time, scale):
        """
        466ce0: the points of the path with the noise less its straight line.
        """
        noise_a, noise_b = self.noise(rng, time, scale)
        count = len(noise_a)
        steps = numpy.arange(count, dtype=numpy.float64)
        across_a = noise_a - noise_a[0] - steps * ((noise_a[-1] - noise_a[0]) / count)
        across_b = noise_b - noise_b[0] - steps * ((noise_b[-1] - noise_b[0]) / count)
        self.points = (self.start + steps[:, None] * self.step + across_a[:, None] * self.side +
                       across_b[:, None] * self.up)
        return self.points

    def length(self):
        return float(numpy.sqrt(numpy.dot(self.step, self.step))) * self.count

#------------------------------------------------------------------------------


class Lightning(object):
    """
    CLightning: the main bolt and six branches between two points.
    """

    def __init__(self, start, end, power, demon=False, rng=None, time=0.0):
        self.rng = rng or random.Random()
        self.power = power
        self.demon = demon
        self.texture_name, self.ball_texture_name = DEMON_TEXTURES if demon else TEXTURES
        self.main = Bolt()
        self.branches = [Bolt() for _ in range(BRANCHES)]
        # the time of the world in ticks
        self.ticks = time
        self.fractions = BRANCH_FRACTIONS
        self.start = numpy.asarray(start, dtype=numpy.float64)
        self.end = numpy.asarray(end, dtype=numpy.float64)
        self.init()

    def __repr__(self):
        return 'Lightning(power %.2f, %d points)' % (self.power, self.main.count)

    @property
    def jaggedness(self):
        if self.power > 0.0:
            return POWER_JAGGEDNESS
        if self.power == 0.0:
            return ZERO_JAGGEDNESS
        return NEGATIVE_JAGGEDNESS

    def init(self):
        """
        4680d0.
        """
        rng = self.rng
        start, end = self.start, self.end
        time = self.ticks / NOISE_TICKS
        self.sway = POWER_SWAY if self.power > 0.0 else NEGATIVE_SWAY
        if abs(abs(end[2] - start[2]) - STORM_HEIGHT) < 0.01:
            self.sway = POWER_SWAY
        power = abs(self.power)
        jaggedness = self.jaggedness
        self.main.aim(start, end)
        self.main.seed(rng, jaggedness, time)
        self.main.width = power
        if self.power < 0.0:
            self.main.width = power * NEGATIVE_WIDTH_SCALE
        if power == 0.0:
            self.main.width = ZERO_WIDTH
            for branch in self.branches:
                offset = self.random_offset()
                branch.aim(start + offset, end + offset)
                branch.seed(rng, jaggedness, time)
                branch.width = rng.getrandbits(32) * 2.0 ** -32 * ZERO_BRANCH_WIDTH_RANGE + ZERO_BRANCH_WIDTH_LEAST
        else:
            for branch, fraction in zip(self.branches, self.fractions):
                branch.aim(start * fraction + end * (1.0 - fraction), end)
                branch.seed(rng, jaggedness, time)
                branch.width = ((rng.getrandbits(32) * 2.0 ** -32 * POWER_BRANCH_WIDTH_RANGE +
                                 POWER_BRANCH_WIDTH_LEAST) * power * 0.5)
        self.shape(self.ticks)

    def random_offset(self):
        """
        4680d0, 467e10: a step up to 0.05 of the length in each axis (the y first, then the x and the z).
        """
        d = self.end - self.start
        reach = float(numpy.sqrt(numpy.dot(d, d))) * ZERO_BRANCH_OFFSET
        y = self.rng.getrandbits(32) * 2.0 ** -32 * 2.0 * reach - reach
        x = self.rng.getrandbits(32) * 2.0 ** -32 * 2.0 * reach - reach
        z = self.rng.getrandbits(32) * 2.0 ** -32 * 2.0 * reach - reach
        return numpy.array((x, y, z))

    def follow(self, start, end):
        """
        467e10: the ends moved (every tick while the effect follows its units).
        """
        self.start = numpy.asarray(start, dtype=numpy.float64)
        self.end = numpy.asarray(end, dtype=numpy.float64)
        self.main.aim(self.start, self.end)
        if self.power == 0.0:
            for branch in self.branches:
                offset = self.random_offset()
                branch.aim(self.start + offset, self.end + offset)
        else:
            for branch, fraction in zip(self.branches, self.fractions):
                branch.aim(self.start * fraction + self.end * (1.0 - fraction), self.end)
        if self.power < 0.0:
            self.shape(self.ticks)

    def shape(self, ticks):
        """
        4672e0: the points of all bolts at the time of the world in ticks.
        """
        time = ticks / NOISE_TICKS
        jaggedness = self.jaggedness
        main = self.main
        main.shape(self.rng, time, jaggedness)
        if self.power != 0.0:
            end = main.end
            for branch, fraction, (rate_a, rate_b) in zip(self.branches, self.fractions, BRANCH_SWAY_RATES):
                sway_a = self.sway * fraction * math.sin(time * rate_a * SWAY_RATE_SCALE)
                sway_b = self.sway * fraction * math.sin(time * rate_b * SWAY_RATE_SCALE)
                target = end + main.side * sway_a + main.up * sway_b
                index = min(int(round((1.0 - fraction) * main.count)), main.count - 1)
                branch.start = numpy.array(main.points[index])
                branch.step = (target - branch.start) / branch.count
        for branch in self.branches:
            branch.shape(self.rng, time, jaggedness)

    def update(self, ticks=1.0):
        self.ticks += ticks

    def bands(self, fraction=0.0):
        """
        467ad0: (points, width in the units of the world, the length of the texture along the way, the band)
        of every bolt to draw now; a bolt with power gets new noise at every drawing.
        """
        if self.power >= 0.0:
            self.shape(self.ticks + fraction)
        result = []
        for bolt in [self.main] + self.branches:
            if self.power >= 0.0:
                band = self.rng.getrandbits(32) & (TEXTURE_BANDS - 1)
                along = bolt.length() * TEXTURE_LENGTH_SCALE
            else:
                band = 0
                along = 0.0
            result.append((bolt.points, bolt.width * WIDTH_SCALE, along, band))
        return result

    def balls(self):
        """
        467ad0: the points and the sizes of the balls of light (467590).
        """
        start = self.main.start
        end = self.main.end
        d = end - start
        size = float(numpy.sqrt(numpy.dot(d, d))) * BALL_SIZE_SCALE
        if self.power > 0.0:
            size = min(max(size, BALL_LEAST), BALL_MOST)
        result = []
        if self.power >= 0.0:
            result.append((tuple(float(v) for v in end), size))
        if self.power == 0.0:
            result.append((tuple(float(v) for v in start), size))
        return result


def ball_squares(center, size, toward_camera, frame):
    """
    467590: the 8 squares of a ball of light, (center, size, frame of the 4 x 4 atlas, alpha 0..255).
    """
    squares = []
    for i in range(BALL_SQUARES):
        t = i / float(BALL_SQUARES)
        point = tuple(center[k] + toward_camera[k] * t * size for k in range(3))
        squares.append((point, math.sqrt(1.0 - t * t) * size, (frame + 2 * i) % BALL_FRAMES, BALL_ALPHA))
    return squares
