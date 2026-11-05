"""
Map of the zones where units can walk, shared by the server and the client.

zonemap.png of the biome has SCALE times more pixels along each side than heightmap.png, one pixel of the map
of the land is SCALE x SCALE pixels of the zonemap. Every pixel keeps the distance from its center to the nearest
border of the closed area: steep mountains, deep water, trees, buildings, see server.land.build_zonemap().
The border is DISTANCE_BORDER, a brighter pixel is that many DISTANCE_STEP segments away from the closed area,
a darker one is inside of it. Far from the border the distance is cut to what fits into one byte.

A unit is a circle: it can stand at the position when the distance there is not smaller than its radius.
The server and the client check the movement with the same code and the same file, so the client knows
in advance where its hero is going to be stopped, see lib.physics.move().
Module needs only Pillow, the server has no numpy.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import io
import math

from PIL import Image

#------------------------------------------------------------------------------

from lib import const

#------------------------------------------------------------------------------

# value of a pixel of zonemap.png right at the border of the closed area
DISTANCE_BORDER = 128

# distance in segments of one step of a pixel value, the values from 0 to 255 cover about 2 segments each way
DISTANCE_STEP = 1.0 / 64.0

# pixels of zonemap.png along one segment of the map, see server.land.build_zonemap()
SCALE = 4

# half of the width of the hero figure, in the units of the models. A unit with Template.scale is bigger
# or smaller as wide as its scale along x and y of the model, see unit_size()
UNIT_RADIUS = 0.82

# the distance does not change around the position when its gradient is smaller, the way out is not known there
FLAT_GRADIENT = 0.000001

# a unit is pushed out of the closed area not further than that in one push, in segments
MAX_PUSH = 0.25

#------------------------------------------------------------------------------


def unit_size(u):
    """
    How many times the unit is wider than a unit of the usual size: the larger scale of its template along x and y of the model.
    """
    if u is None:
        return 1.0
    scale = u.template.scale
    return max(float(scale[0]), float(scale[1]))


def unit_radius(size):
    """
    Radius in segments of a unit of that size, see unit_size(): how far it keeps from the closed area.
    """
    return UNIT_RADIUS * size / const.SEGMENT_SIZE

#------------------------------------------------------------------------------


class ZoneMap(object):
    """
    Distances to the closed area, see the module description. Positions are continuous map coordinates
    x = w + shift_w, y = h + shift_h, the map wraps around at its edges.
    """

    def __init__(self, width, height, scale, data):
        # size of the map in segments
        self.width = width
        self.height = height
        self.scale = scale
        # one byte per pixel, row by row
        self.data = data
        self.pixels_width = width * scale
        self.pixels_height = height * scale

    @classmethod
    def from_png(cls, png_data, width, height):
        """
        Reads zonemap.png of the biome, width and height are the size of the map in segments.
        """
        im = Image.open(io.BytesIO(png_data)).convert('L')
        scale = im.size[0] // width
        if scale < 1 or im.size != (width * scale, height * scale):
            raise ValueError(f'zonemap.png is {im.size[0]}x{im.size[1]}, but it must be {width}x{height} times {SCALE}')
        if _Debug:
            print(f'zonemap is {im.size[0]}x{im.size[1]}, {scale} pixels per segment')
        return cls(width, height, scale, im.tobytes())

    def distance(self, x, y):
        """
        Returns (distance, gradient_x, gradient_y): distance in segments from the position to the closed area,
        negative inside of it, and how fast it grows along x and y. Values of the 4 pixels around the position
        are interpolated between their centers.
        """
        sx = x * self.scale - 0.5
        sy = y * self.scale - 0.5
        i = int(math.floor(sx))
        j = int(math.floor(sy))
        u = sx - i
        v = sy - j
        row0 = (j % self.pixels_height) * self.pixels_width
        row1 = ((j + 1) % self.pixels_height) * self.pixels_width
        i0 = i % self.pixels_width
        i1 = (i + 1) % self.pixels_width
        data = self.data
        d00 = data[row0 + i0]
        d10 = data[row0 + i1]
        d01 = data[row1 + i0]
        d11 = data[row1 + i1]
        value = (d00 * (1.0 - u) + d10 * u) * (1.0 - v) + (d01 * (1.0 - u) + d11 * u) * v
        k = DISTANCE_STEP * self.scale
        gradient_x = ((d10 - d00) * (1.0 - v) + (d11 - d01) * v) * k
        gradient_y = ((d01 - d00) * (1.0 - u) + (d11 - d10) * u) * k
        return (value - DISTANCE_BORDER) * DISTANCE_STEP, gradient_x, gradient_y

    def is_open(self, x, y, radius=0.0):
        """
        A unit with that radius in segments can stand at the position, see unit_radius().
        """
        return self.distance(x, y)[0] >= radius

    def push(self, x, y, radius, gap, from_x=None, from_y=None):
        """
        Returns the movement which takes a unit with that radius out of the closed area the shortest way,
        or None when it is outside already. gap is the free space left between the unit and the border.
        When the distance does not change around the position the way out is not known and the unit goes back
        to (from_x, from_y). A push is not longer than MAX_PUSH, a unit deep inside is pushed again, see physics.slide().
        """
        distance, gradient_x, gradient_y = self.distance(x, y)
        if distance >= radius:
            return None
        rate = math.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
        if rate < FLAT_GRADIENT:
            if from_x is None or from_y is None:
                return 0.0, 0.0
            return from_x - x, from_y - y
        length = min((radius - distance) / rate + gap, MAX_PUSH)
        return gradient_x / rate * length, gradient_y / rate * length
