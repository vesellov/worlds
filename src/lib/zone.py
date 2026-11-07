"""
Map of the zones where units can walk, shared by the server and the client.

zonemap.png of the biome is a black and white mask, see server.land.build_zonemap(): black pixels are the closed land,
steep mountains and deep water, where units can not walk. It has SCALE times more pixels along each side
than heightmap.png, one pixel of the map of the land is SCALE x SCALE pixels of the mask.
Trees and buildings are not in the file: when the biome is loaded the server and the client both build the same
map of the distances from the mask, plants.bin and buildings.json, see ZoneMap.build(). It is built only with
the exact integer math and the basic floating point operations, so every device gets exactly the same bytes.

Every pixel of the built map keeps the distance from its center to the nearest border of the closed area.
The border is DISTANCE_BORDER, a bigger value is that many DISTANCE_STEP segments away from the closed area,
a smaller one is inside of it. Far from the border the distance is cut to what fits into one byte.

A unit is a circle: it can stand at the position when the distance there is not smaller than its radius.
The server and the client check the movement with the same code and the same map, so the client knows
in advance where its hero is going to be stopped, see lib.physics.move().
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import io
import math

import numpy as np
from PIL import Image

#------------------------------------------------------------------------------

from lib import const
from lib import mth

#------------------------------------------------------------------------------

# value of a pixel of the map right at the border of the closed area
DISTANCE_BORDER = 128

# distance in segments of one step of a pixel value, the values from 0 to 255 cover about 2 segments each way
DISTANCE_STEP = 1.0 / 64.0

# distances further than that from the border, in segments, are cut to the smallest or the largest value
DISTANCE_RANGE = DISTANCE_BORDER * DISTANCE_STEP

# pixels of zonemap.png along one segment of the map, see server.land.build_zonemap()
SCALE = 4

# half of the width of the hero figure, in the units of the models. A unit with Template.scale is bigger
# or smaller as wide as its scale along x and y of the model, see unit_size()
UNIT_RADIUS = 0.82

# the distance does not change around the position when its gradient is smaller, the way out is not known there
FLAT_GRADIENT = 0.000001

# a unit is pushed out of the closed area not further than that in one push, in segments
MAX_PUSH = 0.25

# plants with such names are solid, units can not walk through them: trees and stumps,
# other plants (grass, reed, bushes, mushrooms ...) units walk through
SOLID_PLANT_NAME_SUFFIXES = ('_tree', '_stump')

# radius of the trunk near the ground (up to 2.0 above it, a bit higher than the hero), in the units of the models.
# Measured on the "trunk" part for the 8 basic variants of the model: coefficients 0.0 and 1.0 in the order of
# mth.trilinear(), the first one is the model with all coefficients 0.0. The radius of a plant is interpolated
# from them by the coefficients of the plant, the same way the client builds the model of the plant
TRUNK_RADIUS = {
    'nafltr21': (1.35, 2.71, 1.35, 1.35, 2.71, 5.41, 2.71, 2.71),
    'nafltr22': (0.58, 1.15, 0.58, 0.58, 0.86, 1.73, 0.86, 0.86),
    'nafltr56': (0.29, 1.11, 0.43, 0.43, 0.44, 1.61, 0.65, 0.65),
    'nafltr57': (0.25, 0.76, 0.62, 0.62, 0.38, 1.16, 0.94, 0.94),
    'nafltr68': (0.29, 1.11, 0.43, 0.43, 0.44, 1.61, 0.65, 0.65),
    'nafltr69': (0.25, 0.76, 0.62, 0.62, 0.38, 1.16, 0.94, 0.94),
    'nafltr75': (0.49, 1.97, 0.98, 0.98, 0.98, 3.94, 1.97, 1.97),
}

# used for a solid plant which model is not in TRUNK_RADIUS, in the units of the models
DEFAULT_TRUNK_RADIUS = 0.5

# free space between the figure of a unit and a trunk, about half of the hero size, in the units of the models
TRUNK_GAP = 0.5

# radius in segments of the area of a building where units can not walk, they also keep their own radius from it.
# "k" of the building in buildings.json, see server.land.build_capitals() and build_towers()
BUILDING_RADIUS = {
    'capital': 0.44,
    'tower': 0.24,
    'flag': 0.05,
}
DEFAULT_BUILDING_RADIUS = 0.15

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


def read_mask(mask_png_data, width, height):
    """
    Closed pixels of zonemap.png as a numpy array of booleans, rows go along h, and the pixels per segment.
    width and height are the size of the map in segments.
    """
    im = Image.open(io.BytesIO(mask_png_data)).convert('L')
    scale = im.size[0] // width
    if scale < 1 or im.size != (width * scale, height * scale):
        raise ValueError(f'zonemap.png is {im.size[0]}x{im.size[1]}, but it must be {width}x{height} times {SCALE}')
    return np.asarray(im) < 128, scale


def squared_distances(features, reach):
    """
    Squared distance in pixels from the center of every pixel to the center of the nearest pixel where features is True.
    It is exact up to reach pixels, further it is larger than reach * reach. The map wraps around at its edges.
    Only integer math: the distances along the columns first, then the nearest of them along the rows.
    """
    far = reach + 1
    along = np.where(features, 0, far).astype(np.int16)
    for _ in range(reach):
        along = np.minimum(along, np.minimum(np.roll(along, 1, axis=0), np.roll(along, -1, axis=0)) + 1)
    along_squared = along * along
    del along
    result = along_squared.copy()
    for step in range(1, reach + 1):
        result = np.minimum(result, np.minimum(np.roll(along_squared, step, axis=1), np.roll(along_squared, -step, axis=1)) + step * step)
    return result


def land_distances(closed, scale):
    """
    Signed distance in segments from the center of every pixel to the border of the closed area, negative inside of it.
    """
    reach = int(math.ceil(DISTANCE_RANGE * scale + 0.5))
    # the distances are measured from center to center, the border is half of a pixel from the centers next to it
    outside = np.sqrt(squared_distances(closed, reach).astype(np.float64)) - 0.5
    inside = 0.5 - np.sqrt(squared_distances(~closed, reach).astype(np.float64))
    return np.where(closed, inside, outside) / scale


def add_circle(distances, x, y, radius, scale):
    """
    Round obstacle at (x, y) with that radius in segments: the distance to it is exact, not made of pixels.
    """
    rows_count, columns_count = distances.shape
    reach = radius + DISTANCE_RANGE
    columns = np.arange(int(math.floor((x - reach) * scale)), int(math.ceil((x + reach) * scale)) + 1)
    rows = np.arange(int(math.floor((y - reach) * scale)), int(math.ceil((y + reach) * scale)) + 1)
    dx = (columns + 0.5) / scale - x
    dy = (rows + 0.5) / scale - y
    circle = np.sqrt(dy[:, None] * dy[:, None] + dx[None, :] * dx[None, :]) - radius
    # the map wraps around at its edges
    window = np.ix_(rows % rows_count, columns % columns_count)
    distances[window] = np.minimum(distances[window], circle)


def trunk_radius(template, coefs):
    """
    Radius in segments of the area around a solid plant where units can not walk, coefs is the text "c1:c2:c3"
    of the plant. Returns None when the template was not measured, see TRUNK_RADIUS.
    """
    if template not in TRUNK_RADIUS:
        return None
    radius = mth.trilinear(TRUNK_RADIUS[template], mth.quantize_coefs([float(c) for c in coefs.split(':')]))
    return (radius + TRUNK_GAP) / const.SEGMENT_SIZE


def solid_objects(plants, buildings):
    """
    (x, y, radius) in segments of the trees and the buildings, plants and buildings are the data of plants.bin
    and buildings.json. Also returns the templates of the solid plants which were not measured.
    """
    objects = []
    unknown = set()
    # keys of plants are "template#texture#parts", every plant is "c1:c2:c3 x y direction plant_name template_key"
    for plant_key, plants_list in (plants or {}).items():
        template = plant_key.split('#')[0].lower()
        for plant_coded in plants_list:
            coefs, x, y, _, plant_name, _ = plant_coded.split(' ')
            if not plant_name.endswith(SOLID_PLANT_NAME_SUFFIXES):
                continue
            radius = trunk_radius(template, coefs)
            if radius is None:
                unknown.add(template)
                radius = (DEFAULT_TRUNK_RADIUS + TRUNK_GAP) / const.SEGMENT_SIZE
            objects.append((float(x), float(y), radius))
    for building in (buildings or []):
        x = float(building['x'])
        y = float(building['y'])
        # a building of a zone of the original game says which circles around it units can not walk:
        # [dx, dy, radius] in segments, an empty list when units walk through it, see server.land.map_buildings()
        if 'a' in building:
            for dx, dy, radius in building['a']:
                objects.append((x + float(dx), y + float(dy), float(radius)))
            continue
        objects.append((x, y, BUILDING_RADIUS.get(building['k'], DEFAULT_BUILDING_RADIUS)))
    return objects, sorted(unknown)

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
    def build(cls, mask_png_data, width, height, plants=None, buildings=None):
        """
        Builds the map from zonemap.png of the biome, the trees of plants and the buildings, see solid_objects().
        width and height are the size of the map in segments.
        """
        closed, scale = read_mask(mask_png_data, width, height)
        distances = land_distances(closed, scale)
        del closed
        objects, unknown = solid_objects(plants, buildings)
        for x, y, radius in objects:
            add_circle(distances, x, y, radius, scale)
        if unknown:
            if _Debug:
                print(f'WARNING, trunks of {unknown} were not measured, see TRUNK_RADIUS, {DEFAULT_TRUNK_RADIUS} is used')
        values = np.clip(np.round(DISTANCE_BORDER + distances / DISTANCE_STEP), 0, 255).astype(np.uint8)
        if _Debug:
            print(f'zonemap is {width * scale}x{height * scale}, {scale} pixels per segment, {len(objects)} trees and buildings')
        return cls(width, height, scale, values.tobytes())

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
