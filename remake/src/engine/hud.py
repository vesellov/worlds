"""
The panels of the battle screen of game.exe (635210 creates them), drawn from the atlas battle00.mmp.

Every element is a flat piece of the 3D interface (client.gl.interface_point) at the depth z, a smaller z is
nearer to the eye, so z orders the elements and the head models of the portraits. The texels of the atlas
are counted in 1/256 of the texture whatever its size, v from the bottom of the picture: game.exe puts
u / 256 and 1 - v / 256 into the vertices and moves them by half a texel (5e70e0).

The elements of game.exe:
    5e70e0(z, scale, x0, y0, x1, y1, u0, v0, u1, v1)              a rect of the screen with a rect of the texture
    5e7d60(z, scale, x0, y0, x1, y1, border, u0, v0, u1, v1)      a frame: 4 trapezoids along the edges, each
                                                                   maps the strip u0..u1 (along) x v0..v1 (outer to inner)
    5e8c60(z, scale, x, y, radius, segments, u, v, tex_radius)     a disc cut from the texture around (u, v)
    5e91f0(z, scale, x, y, radius, segments, width, u0, v0, u1, v1) a ring, every segment maps the strip
    5e99d0(z, scale, x0, y0, x1, y1, x2, y2, u0, v0, u1, v1, u2, v2) a triangle
The scale only makes the size animation possible, the size on the screen does not depend on it.
An element without a texture is drawn with its color (0xd4, ARGB), the backgrounds are black with alpha 0x50 or 0xb4.

Panels (the window rect, the tooltips "tip N" of texts.res):
    Information window   6456a0  0, 0 - 185, 225     10100 open/close, 10101..10104 the modes
    Text window          649ab0  180, 0 - 620, 100   10200 open/close, 10201 messages, 10202 collected items
    Mini-map             641ee0  615, 0 - 800, 165   10300 open/close, 10301 zoom in, 10302 zoom out, 10303 camera
    Pause dial           6484e0  the corner 800, 600 10400 pause, 10402 normal speed, 10403 fast, 10401 quests
    Mode dial            649150  the corner 0, 600   10500 aggressive/defensive, 10501..10504 crawl, sneak, walk, run
    Portraits            640020  56 x 90 slots in a row centered at x = 400 from y = 510
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

BATTLE_TEXTURE = 'battle00'

TEXEL = 1.0 / 256.0
HALF_TEXEL = 0.5 / 256.0

WHITE = (1.0, 1.0, 1.0, 1.0)
# the colors 0x50000000 and 0xb4000000 of the backgrounds
SHADE = (0.0, 0.0, 0.0, 0x50 / 255.0)
# 6456a0 and 620920: a button which is not chosen is drawn with the half of its color
DIM = (0.5, 0.5, 0.5, 1.0)
DARK_SHADE = (0.0, 0.0, 0.0, 0xb4 / 255.0)

# the strip of the frames around the panels (5e7d60)
PANEL_FRAME_TEXELS = (4, 55, 90, 60)
PANEL_FRAME_BORDER = 5

TIP_PREFIX = 'tip %d'

TIP_INFO = 10100
TIP_INFO_GENERAL = 10101
TIP_INFO_HIT_LOCATIONS = 10102
TIP_INFO_ATTRIBUTES = 10103
TIP_INFO_SPELL_EFFECTS = 10104

# the modes of the information window (6456a0 keeps the mode at +0x180), their buttons are dimmed to 0.5
# when not chosen; the updates are 6477a0, 647b30, 647000 and 646b50
INFO_GENERAL = 0
INFO_HIT_LOCATIONS = 1
INFO_ATTRIBUTES = 2
INFO_SPELL_EFFECTS = 3
INFO_MODE_ACTIONS = ('info general', 'info hit locations', 'info attributes', 'info spell effects')
# the text of the window is drawn in the box 20, 0 - 180, 220 (5fcc00), in the color 0xa7d7e4 (BGR)
INFO_TEXT_ORIGIN = (20, 0)
INFO_TEXT_COLOR = (228, 215, 167)
INFO_BROKEN_COLOR = (255, 0, 0)
# 6477a0: the numbers of the health and the stamina next to their bars
INFO_HEALTH_NUMBER = (130, 185, 160, 200)
INFO_MANA_NUMBER = (130, 200, 160, 215)
# 647000: the lines of the attributes, 15 pixels apart
INFO_ATTRIBUTES_LINE = (5, 35, 155, 50)
INFO_LINE_STEP = 15
INFO_HEALTH_LABEL = 'string infounit_0'
INFO_MANA_LABEL = 'string infounit_1'
INFO_ATTACK_LABEL = 'string infounit_3'
INFO_DEFENSE_LABEL = 'string infounit_4'
INFO_DAMAGE_LABEL = 'string infounit_9'
INFO_LABELS = (INFO_HEALTH_LABEL, INFO_MANA_LABEL, INFO_ATTACK_LABEL, INFO_DEFENSE_LABEL, INFO_DAMAGE_LABEL)
# 647b30: the hit locations (tables 742434 and 74244c), their bars are 50 texels long and turn yellow
# below 2/3 of the health and red below 1/3 (the texels move by 7 rows); the label is 60 x 15
HIT_LOCATION_POINTS = ((50, 35), (50, 105), (95, 70), (5, 70), (95, 140), (5, 140))
HIT_LOCATION_BAR = 50
HIT_LOCATION_ROW_SHIFT = 7
HIT_LOCATION_LABEL = 'string hl_%s'
TIP_TEXT = 10200
TIP_TEXT_MESSAGES = 10201
TIP_TEXT_ITEMS = 10202
TIP_MINIMAP = 10300
TIP_ZOOM_IN = 10301
TIP_ZOOM_OUT = 10302
TIP_DEFAULT_CAMERA = 10303
TIP_PAUSE = 10400
TIP_QUESTS = 10401
TIP_NORMAL_SPEED = 10402
TIP_FAST_SPEED = 10403
TIP_AGGRESSIVE = 10500
TIP_CRAWL = 10501
TIP_SNEAK = 10502
TIP_WALK = 10503
TIP_RUN = 10504

# the dials in the corners (6484e0, 649150): an inner disc of the radius 34 and sectors from 35 to 80
DIAL_INNER_RADIUS = 34.0
DIAL_SECTOR_INNER = 35.0
DIAL_SECTOR_OUTER = 80.0
DIAL_SEGMENTS = 16
# the arrow of a dial stands on this radius and points to the corner (6488a0: 800 + 73 cos, 600 - 73 sin)
DIAL_ARROW_RADIUS = 73.0
DIAL_ARROW_POINTS = ((0.0, 0.0), (17.0, 11.0), (17.0, -11.0))
DIAL_ARROW_TEXELS = ((104, 19), (115, 2), (93, 2))

PAUSE_DIAL_CENTER = (800.0, 600.0)
# the texture of the pause dial: the ring of the sun, the moon and the stars around 174, 81 of battle00
PAUSE_DIAL_TEXTURE_CENTER = (174, 81)
PAUSE_DIAL_SECTORS = (TIP_PAUSE, TIP_NORMAL_SPEED, TIP_FAST_SPEED)
# the ring of the hour turns by (hour + 15) * pi / 12 (6488a0)
HOUR_TURN_OFFSET = 15.0
# the pause makes the ring pulse: 1.4 - (0.7 + 0.3 cos(2 pi t)), t in seconds
PULSE_BASE = 0.7
PULSE_AMPLITUDE = 0.3
PULSE_TOP = 1.4

MODE_DIAL_CENTER = (0.0, 600.0)
MODE_DIAL_TEXTURE_CENTER = (2, 254)
MODE_DIAL_SECTORS = (TIP_CRAWL, TIP_SNEAK, TIP_WALK, TIP_RUN)
MODE_DIAL_SECTOR_COLOR = (0.7, 0.7, 0.7, 1.0)

PORTRAIT_WIDTH = 56
PORTRAIT_TOP = 510
PORTRAIT_FACE_BOTTOM = 580
PORTRAIT_BOTTOM = 600
PORTRAIT_CENTER_X = 400
# the head model of a portrait (640020): at the middle of the slot, y 548, 7 units deep, scale 0.3 (0.22 under 50 pixels)
PORTRAIT_HEAD_Y = 548.0
PORTRAIT_HEAD_DEPTH = 7.0
PORTRAIT_HEAD_SCALE = 0.3
PORTRAIT_HEAD_SMALL_SCALE = 0.22
PORTRAIT_SMALL_WIDTH = 50
PORTRAIT_FRAME_TEXELS = (27, 96, 90, 101)
PORTRAIT_FRAME_BORDER = 3

# 6456a0: the figure of the unit in the information window, in front of the panel (z 10) and behind its bars (z 4);
# the window renders it into the region (20, 0, 180, 220) with its own camera, the remake puts the model itself
# at the depth of the interface and sizes it to that region (chosen by eye, not from game.exe)
INFO_FIGURE_POINT = (92.0, 172.0)
INFO_FIGURE_DEPTH = 8.0
INFO_FIGURE_SCALE = 0.63
# the model of a unit stands along +z and faces -y, the interface goes down along +y and away along +z:
# +90 degrees about x stand it up and turn it to the eye
INFO_FIGURE_ROTATION = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)

# 645f40 and 646b50: the icons of the spells on the unit in the mode of the effects, a column of 24 x 24
# at x 28..52, the middle at y 50, the step 30
SPELL_ICON_RECT = (28, 38, 52, 62)
SPELL_ICON_STEP = 30
SPELL_ICON_SLOTS = 6
SPELL_ICON_Z = 5
HEALTH_BAR_TEXELS = (9, 9, 73, 22)
MANA_BAR_TEXELS = (9, 35, 73, 48)

# 6407a0: the head of a portrait is "infa" + model + build + face number + "face", the texture "face" + sex + number;
# the build of the body by the second coefficient of the complection (unit + 0xc8): "th" (thin) under 0.2,
# "fa" (fat) over 0.8, else "me"
HEAD_MODEL_FORMAT = 'infa%s%s%dface'
FACE_TEXTURE_FORMAT = 'face%s%02d'
BUILD_THIN = 'th'
BUILD_MEDIUM = 'me'
BUILD_FAT = 'fa'
BUILD_THIN_BELOW = 0.2
BUILD_FAT_ABOVE = 0.8

# the text window shows the messages here
TEXT_BOX = (190, 25, 610, 95)

# the mini-map (641ee0, its update 642b00 -> 642e40 and 643d80): the texture of #maps of map.txt covers
# max(width, height) meters of the map from (0, 0), a sector is 32 texels; u = x / size, v = y / size
MINIMAP_CENTER = (700.0, 80.0)
MINIMAP_SECTOR_TEXELS = 32
# the elements of the map: one unit of their space is 80 pixels (the disc, the marks) or 100 (the square),
# it is half of the size of the map in meters divided by the zoom
MINIMAP_UNIT_PIXELS = 80.0
MINIMAP_SQUARE_UNIT_PIXELS = 100.0
MINIMAP_DISC_RADIUS = 0.8
MINIMAP_DISC_SEGMENTS = 16
MINIMAP_SQUARE_COLOR = (0.8, 0.8, 0.8, 1.0)
# while a zoom button is held the zoom goes 2.5 times a second, from 1 to size * 0.05
MINIMAP_ZOOM_SPEED = 2.5
MINIMAP_ZOOM_LIMIT = 0.05
# the marks of the units: an arrow 0.07 ahead and 0.04 back at 135 degrees, shown closer than sqrt(0.56)
MARK_TIP = 0.07
MARK_BACK = 0.04
MARK_BACK_ANGLE = math.radians(135.0)
MARK_RADIUS_SQUARED = 0.56
# the colors of the marks are pixels of battle00 (641ee0): our party, others, enemies
MARK_OWN = 0
MARK_NEUTRAL = 1
MARK_ENEMY = 2
MARK_UVS = ((0.9375, 0.984375), (0.953125, 0.984375), (0.96875, 0.984375), (0.984375, 0.984375))
# the compass in the middle: the model in25arrow 8.1 deep at 694, 84, scale 0.8, turned by the camera,
# its texture is battle00 moved by (0.2227, 0.1406)
COMPASS_MODEL = 'in25arrow'
COMPASS_POINT = (694.0, 84.0)
COMPASS_DEPTH = 8.1
COMPASS_SCALE = 0.8
COMPASS_TEXTURE_OFFSET = (0.2227, 0.1406)

#------------------------------------------------------------------------------


class Element(object):
    """
    Triangles of (x, y, u, v): the screen point and the texel of the atlas, at the depth z
    (CInterfaceWindow::NewElement 5ff3c0 makes one).
    """

    def __init__(self, z, triangles, texture_name=BATTLE_TEXTURE, color=WHITE, name=None, raw_uv=False):
        """
        raw_uv: the u and v of the triangles are already the texture coordinates of game.exe (the mini-map).
        """
        self.z = float(z)
        self.triangles = triangles
        self.texture_name = texture_name
        self.color = tuple(color)
        self.name = name
        self.raw_uv = raw_uv

    def __repr__(self):
        return 'Element(%s z=%g %d triangles)' % (self.name or self.texture_name, self.z, len(self.triangles))

    def vertices(self):
        """
        [(x, y, u, v), ...] of the triangles, u and v as the shader wants them (the texture from its first row).
        """
        result = []
        for triangle in self.triangles:
            for x, y, u, v in triangle:
                if self.raw_uv:
                    result.append((x, y, u, v))
                else:
                    result.append((x, y, u * TEXEL + HALF_TEXEL, 1.0 - v * TEXEL - HALF_TEXEL))
        return result

    def bounds(self):
        xs = [p[0] for t in self.triangles for p in t]
        ys = [p[1] for t in self.triangles for p in t]
        return min(xs), min(ys), max(xs), max(ys)

#------------------------------------------------------------------------------


class Region(object):
    """
    A place of the mouse: a polygon of the screen with a tooltip and an action. game.exe makes it by
    CInterfaceWindow::NewRegion (5ff830), adds the points by CInterfaceRegion::AddPoint (5fca70) and
    the tooltip "tip N" by SetTooltip (5fc730).
    """

    def __init__(self, points, tip=None, action=None):
        self.points = [tuple(p) for p in points]
        self.tip = tip
        self.action = action

    def __repr__(self):
        return 'Region(%s, %s)' % (self.action, self.tip)

    def contains(self, x, y):
        inside = False
        n = len(self.points)
        for i in range(n):
            x0, y0 = self.points[i]
            x1, y1 = self.points[(i + 1) % n]
            if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
        return inside

#------------------------------------------------------------------------------


class Panel(object):

    def __init__(self, name, elements=None, regions=None):
        self.name = name
        self.elements = elements or []
        self.regions = regions or []

#------------------------------------------------------------------------------


def quad(z, rect, texels=None, color=WHITE, texture_name=BATTLE_TEXTURE, name=None):
    """
    5e70e0: the rect (x0, y0, x1, y1) of the screen with the texels (u0, v0, u1, v1), without texels only the color.
    """
    x0, y0, x1, y1 = rect
    if texels is None:
        texture_name = None
        texels = (0, 0, 0, 0)
    u0, v0, u1, v1 = texels
    if v1 < v0:
        # game.exe takes one texel off a flipped rect
        v0, v1 = v0 - 1, v1 - 1
    a, b, c, d = (x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1), (x0, y1, u0, v1)
    return Element(z, [(a, b, c), (a, c, d)], texture_name, color, name)


def texture_quad(z, rect, texture_name, color=WHITE, name=None):
    """
    A quad of a whole texture, not a piece of an atlas (the icons of the spells, 646b50).
    """
    x0, y0, x1, y1 = rect
    a, b, c, d = (x0, y0, 0, 0), (x1, y0, 1, 0), (x1, y1, 1, 1), (x0, y1, 0, 1)
    return Element(z, [(a, b, c), (a, c, d)], texture_name, color, name, raw_uv=True)


def spell_icon_rect(index):
    """
    The rect of the icon of the slot of 645f40: (28, 38, 52, 62), then 30 pixels lower.
    """
    x0, y0, x1, y1 = SPELL_ICON_RECT
    top = y0 + index * SPELL_ICON_STEP
    return (x0, top, x1, y1 + index * SPELL_ICON_STEP)


def spell_icons(texture_names):
    """
    646b50: the textures of the spells on the unit in the slots of the column, the gaps stay empty.
    texture_names: texture name or None for a slot, the first SPELL_ICON_SLOTS of them are drawn.
    """
    elements = []
    for index, texture_name in enumerate(list(texture_names)[:SPELL_ICON_SLOTS]):
        if texture_name is None:
            continue
        elements.append(texture_quad(SPELL_ICON_Z, spell_icon_rect(index), texture_name, name='spell effect'))
    return elements


def frame(z, rect, border, texels, color=WHITE, name=None):
    """
    5e7d60: four trapezoids along the edges (top, right, bottom, left going around), the outer edge takes v0,
    the inner v1, the strip runs from u0 to u1 along every edge.
    """
    x0, y0, x1, y1 = rect
    u0, v0, u1, v1 = texels
    outer = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    inner = [(x0 + border, y0 + border), (x1 - border, y0 + border), (x1 - border, y1 - border), (x0 + border, y1 - border)]
    triangles = []
    for i in range(4):
        j = (i + 1) % 4
        a = outer[i] + (u0, v0)
        b = inner[i] + (u0, v1)
        c = outer[j] + (u1, v0)
        d = inner[j] + (u1, v1)
        triangles.append((a, b, c))
        triangles.append((b, d, c))
    return Element(z, triangles, BATTLE_TEXTURE, color, name)


def hollow_frame(z, rect, border, texels, color=WHITE, texture_name=BATTLE_TEXTURE, name=None):
    """
    5e8450: the rect without its middle, 8 triangles between the outer edge and the edge moved in by the border;
    the inner texels are moved in by the same part of the texel rect.
    """
    x0, y0, x1, y1 = rect
    u0, v0, u1, v1 = texels
    du = (u1 - u0) * border / float(x1 - x0)
    dv = (v1 - v0) * border / float(y1 - y0)
    outer = [(x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1), (x0, y1, u0, v1)]
    inner = [(x0 + border, y0 + border, u0 + du, v0 + dv), (x1 - border, y0 + border, u1 - du, v0 + dv),
             (x1 - border, y1 - border, u1 - du, v1 - dv), (x0 + border, y1 - border, u0 + du, v1 - dv)]
    triangles = []
    for i in range(4):
        j = (i + 1) % 4
        triangles.append((outer[i], inner[i], outer[j]))
        triangles.append((inner[i], inner[j], outer[j]))
    return Element(z, triangles, texture_name, color, name)


def tiles(z, rect, texels, columns, rows, color=WHITE, texture_name=BATTLE_TEXTURE, name=None):
    """
    5e7680: the rect split into columns x rows cells, every cell shows the same texels.
    """
    x0, y0, x1, y1 = rect
    width = (x1 - x0) / float(columns)
    height = (y1 - y0) / float(rows)
    triangles = []
    for row in range(rows):
        for column in range(columns):
            cell = (x0 + column * width, y0 + row * height, x0 + (column + 1) * width, y0 + (row + 1) * height)
            triangles.extend(quad(z, cell, texels).triangles)
    return Element(z, triangles, texture_name, color, name)


def disc(z, center, radius, segments, texture_center, texture_radius, angle=0.0, color=WHITE, texture_name=BATTLE_TEXTURE,
         name=None):
    """
    5e8c60: a polygon of the segments cut from the texture around texture_center, turned by the angle
    (clockwise on the screen, the quaternion about +z of game.exe).
    """
    cx, cy = center
    tu, tv = texture_center
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for i in range(segments):
        t = 2.0 * math.pi * i / segments
        x, y = math.cos(t) * radius, math.sin(t) * radius
        points.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a,
                       tu + math.cos(t) * texture_radius, tv - math.sin(t) * texture_radius))
    triangles = [(points[0], points[i], points[i + 1]) for i in range(1, segments - 1)]
    return Element(z, triangles, texture_name, color, name)


def ring(z, center, radius, segments, width, texels, color=WHITE, name=None):
    """
    5e91f0: segments of a ring from radius - width to radius, every one maps the strip (outer v0, inner v1).
    """
    cx, cy = center
    u0, v0, u1, v1 = texels
    triangles = []
    for i in range(segments):
        t0 = 2.0 * math.pi * i / segments
        t1 = 2.0 * math.pi * (i + 1) / segments
        a = (cx + math.cos(t0) * radius, cy + math.sin(t0) * radius, u0, v0)
        b = (cx + math.cos(t0) * (radius - width), cy + math.sin(t0) * (radius - width), u0, v1)
        c = (cx + math.cos(t1) * radius, cy + math.sin(t1) * radius, u1, v0)
        d = (cx + math.cos(t1) * (radius - width), cy + math.sin(t1) * (radius - width), u1, v1)
        triangles.append((a, b, c))
        triangles.append((b, d, c))
    return Element(z, triangles, BATTLE_TEXTURE, color, name)


def triangle(z, points, texels, color=WHITE, name=None):
    """
    5e99d0: one triangle of the screen with its texels.
    """
    return Element(z, [tuple(tuple(p) + tuple(t) for p, t in zip(points, texels))], BATTLE_TEXTURE, color, name)


def rect_region(rect, tip=None, action=None):
    """
    CInterfaceRegion::SetRect (5fc9e0).
    """
    x0, y0, x1, y1 = rect
    return Region([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], tip, action)


def corner_point(corner, radius, angle):
    """
    The point at the radius from a corner of the screen, the angle goes from the bottom edge up
    (game.exe: 800 - r cos, 600 - r sin for the right corner, r cos, 600 - r sin for the left one).
    """
    cx, cy = corner
    side = -1.0 if cx > 0 else 1.0
    return cx + side * radius * math.cos(angle), cy - radius * math.sin(angle)


def dial_sector_region(corner, angle0, angle1, tip, action=None):
    points = [corner_point(corner, DIAL_SECTOR_INNER, angle1), corner_point(corner, DIAL_SECTOR_INNER, angle0),
              corner_point(corner, DIAL_SECTOR_OUTER, angle0), corner_point(corner, DIAL_SECTOR_OUTER, angle1)]
    return Region(points, tip, action)


def dial_inner_region(corner, tip, action=None):
    points = [corner] + [corner_point(corner, DIAL_INNER_RADIUS, math.pi * 0.5 * i / 4) for i in range(5)]
    return Region(points, tip, action)


def dial_arrow(z, corner, angle):
    """
    The arrow of a dial on the radius 73 at the angle, its tip points to the corner.
    """
    px, py = corner_point(corner, DIAL_ARROW_RADIUS, angle)
    length = math.hypot(corner[0] - px, corner[1] - py)
    fx, fy = (corner[0] - px) / length, (corner[1] - py) / length
    # the local -x of the arrow goes to the corner, the local y across it
    points = [(px - lx * fx - ly * fy, py - lx * fy + ly * fx) for lx, ly in DIAL_ARROW_POINTS]
    return triangle(z, points, DIAL_ARROW_TEXELS, name='arrow')


def sector_angle(index, count):
    """
    The middle of the sector of a quarter of a circle split into count parts.
    """
    return math.pi * 0.5 * (index + 0.5) / count

#------------------------------------------------------------------------------


def info_panel(mode=INFO_GENERAL, health=1.0, mana=None, hit_locations=()):
    """
    The information window of 6456a0 in one of its modes. health and mana are the parts left (mana None hides
    the stamina, 6477a0 shows it only for the own party), hit_locations are the parts of the health
    of the 6 locations of the body.
    """
    elements = [
        quad(60, (20, 0, 180, 220), color=SHADE),
        quad(20, (0, 0, 20, 220), color=DARK_SHADE),
        quad(10, (0, 0, 20, 20), (34, 150, 54, 170), name='info'),
    ]
    for index, texels in enumerate(((47, 62, 67, 92), (69, 62, 89, 92), (3, 94, 23, 124), (233, 112, 253, 142))):
        y = 100 + index * 30
        elements.append(quad(10, (0, y, 20, y + 30), texels, WHITE if index == mode else DIM,
                             name=INFO_MODE_ACTIONS[index]))
    elements.append(frame(10, (-5, -5, 185, 225), PANEL_FRAME_BORDER, PANEL_FRAME_TEXELS))
    if mode == INFO_GENERAL:
        # 6477a0: the bars of the health and the stamina are cut to the part left
        elements.extend(info_bar(4, (40, 190), 100, (64, 154, 164, 159), health, 'health'))
        elements.append(quad(5, (35, 190, 145, 195), (59, 147, 169, 152)))
        if mana is not None:
            elements.extend(info_bar(4, (40, 205), 100, (64, 161, 164, 166), mana, 'mana'))
            elements.append(quad(5, (35, 205, 145, 210), (59, 147, 169, 152)))
    elif mode == INFO_HIT_LOCATIONS:
        for (x, y), part in zip(HIT_LOCATION_POINTS, hit_locations):
            shift = 0
            if part <= 2.0 / 3.0:
                shift = HIT_LOCATION_ROW_SHIFT if part > 1.0 / 3.0 else HIT_LOCATION_ROW_SHIFT * 2
            u0, v0, u1, v1 = 64, 126 + shift, 114, 131 + shift
            elements.extend(info_bar(4, (x + 25, y + 15), HIT_LOCATION_BAR, (u0, v0, u1, v1), part, 'hit location'))
            elements.append(quad(5, (x + 20, y + 15, x + 80, y + 20), (59, 119, 119, 124)))
    regions = [
        rect_region((0, 0, 20, 30), TIP_INFO, 'info'),
        rect_region((0, 100, 20, 130), TIP_INFO_GENERAL, 'info general'),
        rect_region((0, 130, 20, 160), TIP_INFO_HIT_LOCATIONS, 'info hit locations'),
        rect_region((0, 160, 20, 190), TIP_INFO_ATTRIBUTES, 'info attributes'),
        rect_region((0, 190, 20, 220), TIP_INFO_SPELL_EFFECTS, 'info spell effects'),
    ]
    return Panel('info', elements, regions)


def info_bar(z, point, length, texels, part, name):
    """
    A bar of the information window cut to the part (0..1) on the screen and in the texels, none for 0.
    """
    part = min(max(part, 0.0), 1.0)
    if part <= 0.0:
        return []
    x, y = point
    u0, v0, u1, v1 = texels
    return [quad(z, (x, y, x + length * part, y + 5), (u0, v0, u0 + (u1 - u0) * part, v1), name=name)]


def info_texts(mode, labels, health=None, mana=None, hit_locations=(), attack=None, defense=None, damage=None):
    """
    The texts of the information window as (text, screen rect, color). labels: text name -> text of texts.res.
    health and mana are (current, maximum) or None, hit_locations are (name, broken) of the 6 locations,
    attack and defense numbers, damage (min, max), all of them None when not known.
    """
    ox, oy = INFO_TEXT_ORIGIN

    def _rect(rect, dy=0):
        x0, y0, x1, y1 = rect
        return (x0 + ox, y0 + oy + dy, x1 + ox, y1 + oy + dy)

    result = []
    if mode == INFO_GENERAL:
        if health is not None:
            result.append(('%d' % round(max(health[0], 0)), _rect(INFO_HEALTH_NUMBER), INFO_TEXT_COLOR))
        if mana is not None:
            result.append(('%d' % round(max(mana[0], 0)), _rect(INFO_MANA_NUMBER), INFO_TEXT_COLOR))
    elif mode == INFO_HIT_LOCATIONS:
        for (x, y), (name, broken) in zip(HIT_LOCATION_POINTS, hit_locations):
            text = labels.get(HIT_LOCATION_LABEL % name, name)
            result.append((text, _rect((x, y, x + 60, y + 15)), INFO_BROKEN_COLOR if broken else INFO_TEXT_COLOR))
    elif mode == INFO_ATTRIBUTES:
        lines = []
        if health is not None:
            lines.append('%s %d/%d' % (labels.get(INFO_HEALTH_LABEL, ''),
                                       round(max(health[0], 0)), round(health[1])))
        if mana is not None:
            lines.append('%s %d/%d' % (labels.get(INFO_MANA_LABEL, ''),
                                       round(max(mana[0], 0)), round(mana[1])))
        # 647000 puts the speed and the sight here, they are not known yet
        if attack is not None:
            lines.append('%s %d' % (labels.get(INFO_ATTACK_LABEL, ''), attack))
        if defense is not None:
            lines.append('%s %d' % (labels.get(INFO_DEFENSE_LABEL, ''), defense))
        if damage is not None:
            lines.append('%s %d-%d' % (labels.get(INFO_DAMAGE_LABEL, ''), damage[0], damage[1]))
        for i, line in enumerate(lines):
            result.append((line, _rect(INFO_ATTRIBUTES_LINE, i * INFO_LINE_STEP), INFO_TEXT_COLOR))
    return result


def text_panel():
    elements = [
        quad(20, (180, 20, 620, 95), color=SHADE),
        quad(20, (180, 0, 620, 20), color=DARK_SHADE),
        quad(10, (185, 0, 215, 20), (2, 128, 32, 148)),
        quad(10, (215, 0, 245, 20), (2, 150, 32, 170)),
        frame(18, (180, -5, 620, 100), PANEL_FRAME_BORDER, PANEL_FRAME_TEXELS),
    ]
    regions = [
        rect_region((385, 0, 415, 20), TIP_TEXT, 'text'),
        rect_region((185, 0, 215, 20), TIP_TEXT_MESSAGES, 'text messages'),
        rect_region((215, 0, 245, 20), TIP_TEXT_ITEMS, 'text items'),
    ]
    return Panel('text', elements, regions)


def minimap_panel():
    """
    641ee0. The map itself (a square of 160 dimmed to 0.8 and a disc of the radius 64) is drawn by the client.
    """
    elements = [
        quad(20, (620, 0, 780, 160), color=SHADE),
        quad(20, (780, 0, 800, 160), color=DARK_SHADE),
        quad(10, (780, 100, 800, 130), (25, 62, 45, 92)),
        quad(10, (780, 130, 800, 160), (3, 62, 23, 92)),
        quad(10, (780, 70, 800, 100), (34, 118, 54, 148)),
        ring(10, (700, 80), 66, 16, 3, (27, 107, 91, 112)),
        frame(10, (615, -5, 805, 165), PANEL_FRAME_BORDER, PANEL_FRAME_TEXELS),
    ]
    regions = [
        rect_region((780, 0, 800, 30), TIP_MINIMAP, 'minimap'),
        rect_region((780, 100, 800, 130), TIP_ZOOM_IN, 'zoom in'),
        rect_region((780, 130, 800, 160), TIP_ZOOM_OUT, 'zoom out'),
        rect_region((780, 70, 800, 100), TIP_DEFAULT_CAMERA, 'default camera'),
    ]
    return Panel('minimap', elements, regions)


def pause_dial(hour=6.0, speed_index=1, pulse=1.0):
    """
    6484e0 and its update 6488a0: three discs of the ring of the day, the middle one turns with the hour;
    the arrow shows the sector of the speed (0 pause, 1 normal, 2 fast).
    """
    angle = (hour + HOUR_TURN_OFFSET) * math.pi / 12.0
    center, texture_center = PAUSE_DIAL_CENTER, PAUSE_DIAL_TEXTURE_CENTER
    elements = [
        disc(12, center, 80, DIAL_SEGMENTS, texture_center, 80),
        disc(10, center, 65, DIAL_SEGMENTS, texture_center, 65, angle=angle, color=(pulse, pulse, pulse, 1.0), name='hour'),
        disc(8, center, 32, DIAL_SEGMENTS, texture_center, 32),
        dial_arrow(8, center, sector_angle(speed_index, len(PAUSE_DIAL_SECTORS))),
    ]
    regions = [dial_sector_region(center, math.pi / 6.0 * i, math.pi / 6.0 * (i + 1), tip, tip)
               for i, tip in enumerate(PAUSE_DIAL_SECTORS)]
    regions.append(dial_inner_region(center, TIP_QUESTS, TIP_QUESTS))
    return Panel('pause', elements, regions)


def mode_dial(mode_index=2):
    """
    649150: four sectors of the movement (crawl, sneak, walk, run) under a disc, the arrow shows the mode.
    """
    center, (tu, tv) = MODE_DIAL_CENTER, MODE_DIAL_TEXTURE_CENTER
    elements = []
    count = len(MODE_DIAL_SECTORS)
    for i in range(count):
        a0, a1 = math.pi * 0.5 * i / count, math.pi * 0.5 * (i + 1) / count
        points = [center, corner_point(center, DIAL_SECTOR_OUTER, a0), corner_point(center, DIAL_SECTOR_OUTER, a1)]
        texels = [(tu, tv), (tu + DIAL_SECTOR_OUTER * math.cos(a0), tv - DIAL_SECTOR_OUTER * math.sin(a0)),
                  (tu + DIAL_SECTOR_OUTER * math.cos(a1), tv - DIAL_SECTOR_OUTER * math.sin(a1))]
        elements.append(triangle(10, points, texels, color=MODE_DIAL_SECTOR_COLOR, name='sector'))
    elements.append(disc(8, center, 32, DIAL_SEGMENTS, (tu, tv), 32))
    elements.append(dial_arrow(8, center, sector_angle(mode_index, count)))
    regions = [dial_sector_region(center, math.pi * 0.5 * i / count, math.pi * 0.5 * (i + 1) / count, tip, tip)
               for i, tip in enumerate(MODE_DIAL_SECTORS)]
    regions.append(dial_inner_region(center, TIP_AGGRESSIVE, TIP_AGGRESSIVE))
    return Panel('mode', elements, regions)


def portrait_slots(count):
    """
    Left x of the slots of the portraits: a row of 56 pixels wide slots centered at 400 (640020).
    """
    return [PORTRAIT_CENTER_X + i * PORTRAIT_WIDTH - count * PORTRAIT_WIDTH // 2 for i in range(count)]


def portrait(x, health=1.0, mana=1.0):
    """
    The frame and the bars of one portrait, the bars shrink to the right with the health and the mana.
    """
    x1 = x + PORTRAIT_WIDTH
    bar_width = PORTRAIT_WIDTH - 6
    elements = [frame(20, (x, PORTRAIT_TOP, x1, PORTRAIT_BOTTOM), PORTRAIT_FRAME_BORDER, PORTRAIT_FRAME_TEXELS)]
    for value, (y0, y1), texels, name in ((health, (587, 592), HEALTH_BAR_TEXELS, 'health'),
                                          (mana, (592, 597), MANA_BAR_TEXELS, 'mana')):
        value = min(max(value, 0.0), 1.0)
        if value <= 0.0:
            continue
        u0, v0, u1, v1 = texels
        elements.append(quad(7, (x + 3, y0, x + 3 + bar_width * value, y1), (u0, v0, u0 + (u1 - u0) * value, v1), name=name))
    regions = [rect_region((x, PORTRAIT_TOP, x1, PORTRAIT_BOTTOM), None, 'portrait')]
    return Panel('portrait', elements, regions)


def bar_shares(unit):
    """
    641040 (partly): the health (+0x17e of +0x180) and the mana (+0x182 of +0x184) of the unit for its portrait, 0 without
    a maximum.
    """
    fighter = getattr(unit, 'fighter', None)
    if fighter is None:
        return 1.0, 1.0
    health = max(0.0, fighter.health) / fighter.max_health if fighter.max_health > 0.0 else 0.0
    mana = max(0.0, fighter.mana) / fighter.max_mana if fighter.max_mana > 0.0 else 0.0
    return round(health, 3), round(mana, 3)


def portrait_head_scale(width=PORTRAIT_WIDTH):
    return PORTRAIT_HEAD_SCALE if width >= PORTRAIT_SMALL_WIDTH else PORTRAIT_HEAD_SMALL_SCALE


def body_build(fatness):
    if fatness < BUILD_THIN_BELOW:
        return BUILD_THIN
    if fatness > BUILD_FAT_ABOVE:
        return BUILD_FAT
    return BUILD_MEDIUM


def head_model_name(model_name, fatness, face_number):
    """
    infaunhumame1face: the head of a portrait, face_number counts from 1.
    """
    return HEAD_MODEL_FORMAT % (model_name, body_build(fatness), face_number)


def face_texture_name(model_name, face_index):
    """
    facem00: the face of the hero (unhuma "m"), of the heroine (unhufe "f").
    """
    return FACE_TEXTURE_FORMAT % ('f' if model_name == 'unhufe' else 'm', face_index)


def portrait_face_texture(face_rgba):
    """
    The texture the head of a portrait expects: game.exe (640020, 5fffb0) copies the face of 128 x 128 into a cell
    of a 256 x 256 texture and moves the texture coordinates of the head by half of it, so the face of the first
    slot comes out in the lower left quarter (the rows from the first one down).
    """
    height, width = face_rgba.shape[0:2]
    result = np.zeros((height * 2, width * 2, 4), dtype=np.uint8)
    result[:, :, 3] = 255
    result[height:, 0:width] = face_rgba
    return result


def pulse(time_s):
    """
    The brightness of the ring of the pause dial while the game stands.
    """
    return PULSE_TOP - (PULSE_BASE + PULSE_AMPLITUDE * math.cos(2.0 * math.pi * time_s))


class Minimap(object):
    """
    The state of the mini-map: the size of the map and the zoom.
    """

    def __init__(self, texture_name, sectors_x, sectors_y, zoom=1.0):
        self.texture_name = texture_name
        self.width = float(sectors_x * MINIMAP_SECTOR_TEXELS)
        self.height = float(sectors_y * MINIMAP_SECTOR_TEXELS)
        self.size = max(self.width, self.height)
        self.max_zoom = self.size * MINIMAP_ZOOM_LIMIT
        self.zoom = min(zoom, self.max_zoom)

    @property
    def meters_per_unit(self):
        return self.size * 0.5

    def zoom_by(self, direction, dt_s):
        """
        642b00: direction 1 zooms in, -1 out, for the time the button is held.
        """
        factor = MINIMAP_ZOOM_SPEED ** dt_s
        if direction > 0:
            self.zoom = min(self.zoom * factor, self.max_zoom)
        elif direction < 0:
            self.zoom = max(self.zoom / factor, 1.0)

    def to_screen(self, x, y, camera, unit_pixels):
        cx, cy = MINIMAP_CENTER
        scale = unit_pixels / self.meters_per_unit
        return cx + (x - camera[0]) * scale, cy - (y - camera[1]) * scale

    def disc(self, camera):
        """
        642e40: the circle of the radius 0.8 around the camera, cut by the borders of the map.
        """
        reach = self.meters_per_unit / self.zoom
        polygon = []
        for i in range(MINIMAP_DISC_SEGMENTS):
            t = 2.0 * math.pi * i / MINIMAP_DISC_SEGMENTS
            polygon.append((camera[0] + math.cos(t) * MINIMAP_DISC_RADIUS * reach,
                            camera[1] - math.sin(t) * MINIMAP_DISC_RADIUS * reach))
        polygon = clip_polygon(polygon, 0.0, 0.0, self.width, self.height)
        points = []
        for x, y in polygon:
            sx, sy = self.to_screen(x, y, camera, MINIMAP_UNIT_PIXELS * self.zoom)
            points.append((sx, sy, x / self.size, y / self.size))
        triangles = [(points[0], points[i], points[i + 1]) for i in range(1, len(points) - 1)]
        return Element(14, triangles, self.texture_name, name='minimap', raw_uv=True)

    def square(self, camera):
        """
        The map behind the disc: 0.8 units around the camera without the zoom (the 160 x 160 of the panel),
        dimmed to 0.8.
        """
        mpu = self.meters_per_unit * MINIMAP_DISC_RADIUS
        x0 = max(camera[0] - mpu, 0.0)
        x1 = min(camera[0] + mpu, self.width)
        y0 = max(camera[1] - mpu, 0.0)
        y1 = min(camera[1] + mpu, self.height)
        if x0 >= x1 or y0 >= y1:
            return None
        corners = []
        for x, y in ((x0, y1), (x1, y1), (x1, y0), (x0, y0)):
            sx, sy = self.to_screen(x, y, camera, MINIMAP_SQUARE_UNIT_PIXELS)
            corners.append((sx, sy, x / self.size, y / self.size))
        a, b, c, d = corners
        return Element(18, [(a, b, c), (a, c, d)], self.texture_name, MINIMAP_SQUARE_COLOR, name='minimap square',
                       raw_uv=True)

    def marks(self, camera, units):
        """
        643d80: units are (x, y, yaw, kind), the yaw of engine.unit (the forward is (sin, -cos)).
        """
        unit_pixels = MINIMAP_UNIT_PIXELS * self.zoom
        limit = MARK_RADIUS_SQUARED * MINIMAP_UNIT_PIXELS * MINIMAP_UNIT_PIXELS
        cx, cy = MINIMAP_CENTER
        triangles = []
        for x, y, yaw, kind in units:
            sx, sy = self.to_screen(x, y, camera, unit_pixels)
            if (sx - cx) ** 2 + (sy - cy) ** 2 > limit:
                continue
            # the forward on the screen, y goes down
            fx, fy = math.sin(yaw), math.cos(yaw)
            u, v = MARK_UVS[kind]
            points = []
            for length, angle in ((MARK_TIP, 0.0), (MARK_BACK, MARK_BACK_ANGLE), (MARK_BACK, -MARK_BACK_ANGLE)):
                c, s = math.cos(angle), math.sin(angle)
                dx, dy = fx * c - fy * s, fx * s + fy * c
                points.append((sx + dx * length * MINIMAP_UNIT_PIXELS, sy + dy * length * MINIMAP_UNIT_PIXELS, u, v))
            triangles.append(tuple(points))
        return Element(12, triangles, BATTLE_TEXTURE, name='marks', raw_uv=True)

    def elements(self, camera, units):
        result = [self.disc(camera), self.marks(camera, units)]
        square = self.square(camera)
        if square is not None:
            result.append(square)
        return result

#------------------------------------------------------------------------------


def clip_polygon(points, x0, y0, x1, y1):
    """
    The polygon cut by the rect, edge after edge like game.exe 642e40.
    """
    for axis, limit, keep_greater in ((0, x0, True), (1, y0, True), (0, x1, False), (1, y1, False)):
        result = []
        n = len(points)
        for i in range(n):
            current, previous = points[i], points[i - 1]
            current_in = current[axis] >= limit if keep_greater else current[axis] <= limit
            previous_in = previous[axis] >= limit if keep_greater else previous[axis] <= limit
            if current_in != previous_in:
                t = (limit - previous[axis]) / (current[axis] - previous[axis])
                result.append(tuple(previous[k] + (current[k] - previous[k]) * t for k in range(2)))
            if current_in:
                result.append(current)
        points = result
        if not points:
            break
    return points


def compass_angle(camera_yaw_degrees):
    """
    642e40: the compass turns by atan2(-forward x, -forward y) of the camera.
    """
    yaw = math.radians(camera_yaw_degrees)
    return math.atan2(-math.sin(yaw), -math.cos(yaw))


def shifted_texture(rgba, offset):
    """
    The texture sampled with the texture coordinates moved by the offset (u, v), like the UV offset
    of an element of game.exe (+0xcc, +0xd0).
    """
    height, width = rgba.shape[0:2]
    return np.roll(rgba, (-int(round(offset[1] * height)), -int(round(offset[0] * width))), axis=(0, 1))


def battle_panels(hour=6.0, speed_index=1, mode_index=2):
    return [info_panel(), text_panel(), minimap_panel(), pause_dial(hour, speed_index), mode_dial(mode_index)]


def region_at(panels, x, y):
    for panel in panels:
        for region in panel.regions:
            if region.contains(x, y):
                return region
    return None
