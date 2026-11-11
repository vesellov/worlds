"""
The strip of items of game.exe: the row of slots at the bottom of the camp screens (window 621120, built by 620340).

The strip is 100 pixels high across the screen, at y 500 with the texture inventory01 (or at y 0 with inventory02),
its depth is 4 (8 for the upper one):
- the ends (0..50 and 750..800, depth 3.2) show an arrow while the list can go that way (621b10 moves their texels
  by 50 to the right);
- three buttons on each side (30 x 30, depth 2.4) choose the category of the items (620920): the chosen one is bright,
  the others are dimmed to 0.5; the mode of the window (6213e0) moves their texels and chooses their tooltips;
- the slots are a row of 8 cells of 100 pixels (5e7680, depth 8) with a frame over every cell (5e8450, depth 4),
  the items stand between them at the depth 6, as 3D models (5eca10).

The list scrolls by whole slots (620a10): a held arrow adds a slot at once and one more every 0.1 s, the strip
speeds up by 15 slots/s^2 up to 7 slots/s and slows down near the target; with Shift it jumps to the first
or the last item (620dd0). The position never goes over the count minus 7 visible slots.

An item of the strip (5eca10 in the mode 9 of the strip) is its figure "init<we|ar><kind><number><weapon|armor>"
turned by pi about x, its bounds centered in the slot and its larger side scaled to 50 pixels; the texture
"<kind>_<number>.<variant>" (5ec020) is copied into the bottom rows of a 256 x 256 texture, white when there is none.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math

import numpy as np

#------------------------------------------------------------------------------

from eilib import idb
from engine import hud

#------------------------------------------------------------------------------

LOWER_TEXTURE = 'inventory01'
UPPER_TEXTURE = 'inventory02'
LOWER_TOP = 500
UPPER_TOP = 0
LOWER_DEPTH = 4.0
UPPER_DEPTH = 8.0

STRIP_HEIGHT = 100
SLOT_SIZE = 100
SLOTS_LEFT = 50
SLOTS_RIGHT = 750
SLOT_CELLS = 8
VISIBLE_SLOTS = 7
SLOT_TEXELS = (14, 14, 114, 114)
SLOT_FRAME_BORDER = 15

END_WIDTH = 50
END_TEXELS = (142, 14, 192, 114)
# 621b10: the ends with the arrow are 50 texels to the right
END_ARROW_SHIFT = 50

BUTTON_SIZE = 30
BUTTON_TOPS = (5, 35, 65)
LEFT_BUTTON_X = 0
RIGHT_BUTTON_X = 770
LEFT_BUTTON_U = 2
RIGHT_BUTTON_U = 44
BUTTON_ROWS = ((130, 170), (172, 212), (214, 254))
BUTTON_TEXEL_WIDTH = 40

# depths are parts of the depth of the strip (620340)
END_DEPTH = 0.8
BUTTON_DEPTH = 0.6
STRIP_DEPTH = 2.0
ITEM_DEPTH = 1.5

# the arrows (30..50 and 750..770 between 35 and 65 below the top), their tooltips are set by 621120
LEFT_ARROW = (30, 35, 50, 65)
RIGHT_ARROW = (750, 35, 770, 65)
TIP_SCROLL_RIGHT = 20400
TIP_SCROLL_LEFT = 20401

# 6213e0: the mode moves the texels of the buttons and chooses the first tooltip of the six
MODE_ITEMS = 0
MODE_SPELLS = 1
MODE_CONSTRUCTOR = 2
MODE_BUTTON_SHIFT = {MODE_ITEMS: 168, MODE_SPELLS: 84, MODE_CONSTRUCTOR: 0}
MODE_FIRST_TIP = {MODE_ITEMS: 21100, MODE_SPELLS: 21200, MODE_CONSTRUCTOR: 21000}

# the categories of the mode of items (6218b0); the last one shows everything (621530)
CATEGORY_MATERIALS = 0
CATEGORY_RUNES = 1
CATEGORY_READY = 2
CATEGORY_COLLECTED = 3
CATEGORY_QUEST = 4
CATEGORY_ALL = 5
CATEGORIES = 6

# 620a10
SCROLL_REPEAT_S = 0.1
SCROLL_ACCELERATION = 15.0
SCROLL_MAX_SPEED = 7.0
SCROLL_NEAR = 0.5
SCROLL_SNAP = 0.01
SCROLL_SNAP_PENDING = 0.02

SCROLL_NONE = 0
SCROLL_BACK = 1
SCROLL_FORWARD = 2

# 5eca10: the larger side of an item is 50 pixels (0.4816 * 100 * depth * 0.00125 of the interface space)
ITEM_PIXELS = 50.0
ITEM_TEXTURE_SIZE = 256
ITEM_MODEL_FORMAT = 'init%s%s%d%s'
# the tangent of the half of the horizontal angle of view (game.exe 0x740158), 800 pixels are 2 of its units
TAN_HALF_WIDTH = 0.4815746
ITEM_TEXTURE_FORMAT = '%s_%02d.%d'

#------------------------------------------------------------------------------


class InventoryStrip(object):
    """
    The state of the strip: the items, the category, the position of the list of every category and the scrolling.
    """

    def __init__(self, items=(), mode=MODE_ITEMS, upper=False):
        self.items = list(items)
        self.mode = mode
        self.upper = upper
        self.category = CATEGORY_ALL
        self.positions = [0.0] * CATEGORIES
        self.scroll = SCROLL_NONE
        self.scroll_timer = 0.0
        self.pending = 0.0
        self.speed = 0.0

    @property
    def top(self):
        return UPPER_TOP if self.upper else LOWER_TOP

    @property
    def depth(self):
        return UPPER_DEPTH if self.upper else LOWER_DEPTH

    @property
    def texture_name(self):
        return UPPER_TEXTURE if self.upper else LOWER_TEXTURE

    @property
    def position(self):
        return self.positions[self.category]

    def shown_items(self):
        """
        621530: the items of the chosen category, all of them for the last one.
        """
        if self.category == CATEGORY_ALL:
            return list(self.items)
        return [item for item in self.items if item_category(item) == self.category]

    def max_position(self):
        return max(len(self.shown_items()) - VISIBLE_SLOTS, 0)

    def choose_category(self, category):
        self.category = category
        self.scroll = SCROLL_NONE
        self.pending = 0.0
        self.speed = 0.0

    def press_arrow(self, direction, jump=False):
        """
        620dd0: direction -1 is the left arrow, 1 the right one; with jump the list goes to its first or last item.
        """
        if jump:
            self.positions[self.category] = 0.0 if direction < 0 else float(self.max_position())
            self.pending = 0.0
            self.speed = 0.0
            return
        self.scroll = SCROLL_BACK if direction < 0 else SCROLL_FORWARD
        self.scroll_timer = 0.0

    def release(self):
        self.scroll = SCROLL_NONE

    def update(self, dt_s):
        """
        620a10: the held arrow adds whole slots, the position follows them with a speed which grows and falls.
        """
        if self.scroll != SCROLL_NONE:
            steps = 0
            if self.scroll_timer == 0.0:
                steps = 1
                self.scroll_timer = 0.001
            else:
                self.scroll_timer += dt_s
                while self.scroll_timer > SCROLL_REPEAT_S:
                    self.scroll_timer -= SCROLL_REPEAT_S
                    steps += 1
            self.pending += -steps if self.scroll == SCROLL_BACK else steps
        position = self.positions[self.category]
        if self.pending != 0.0 and dt_s > 0.0:
            if abs(self.pending) > SCROLL_NEAR:
                if self.pending <= 0.0:
                    self.speed = max(self.speed - SCROLL_ACCELERATION * dt_s, -SCROLL_MAX_SPEED)
                    speed = max((SCROLL_NEAR + self.pending) / dt_s, self.speed)
                else:
                    self.speed = min(self.speed + SCROLL_ACCELERATION * dt_s, SCROLL_MAX_SPEED)
                    speed = min((self.pending - SCROLL_NEAR) / dt_s, self.speed)
                speed = (speed + self.speed) * 0.5
            else:
                speed = math.copysign(math.sqrt(SCROLL_NEAR * abs(self.pending)) / 0.25, self.pending)
            self.speed = speed
            step = speed * dt_s
            if abs(self.pending) < abs(step):
                step = self.pending
            position += step
            self.pending -= step
            whole = float(int(position))
            if abs(self.pending) < SCROLL_SNAP_PENDING:
                if position - whole < SCROLL_SNAP:
                    position, self.pending = whole, 0.0
                elif position - whole > 1.0 - SCROLL_SNAP:
                    position, self.pending = whole + 1.0, 0.0
        if self.pending == 0.0:
            self.speed = 0.0
        limit = float(self.max_position())
        if position > limit:
            position, self.pending, self.speed = limit, 0.0, 0.0
        if position < 0.0:
            position, self.pending, self.speed = 0.0, 0.0, 0.0
        self.positions[self.category] = position

    def elements(self):
        """
        620340 and 621b10: the elements of the strip at the current position of the list.
        """
        top, depth, texture_name = self.top, self.depth, self.texture_name
        position = self.position
        fraction = position - int(position)
        result = []
        u0, v0, u1, v1 = END_TEXELS
        left_shift = END_ARROW_SHIFT if position > 0.0 else 0
        right_shift = END_ARROW_SHIFT if position < len(self.shown_items()) - VISIBLE_SLOTS else 0
        result.append(hud.quad(depth * END_DEPTH, (0, top, END_WIDTH, top + STRIP_HEIGHT),
                               (u1 + left_shift, v0, u0 + left_shift, v1), texture_name=texture_name, name='left end'))
        result.append(hud.quad(depth * END_DEPTH, (800 - END_WIDTH, top, 800, top + STRIP_HEIGHT),
                               (u0 + right_shift, v0, u1 + right_shift, v1), texture_name=texture_name, name='right end'))
        shift = MODE_BUTTON_SHIFT[self.mode]
        for index in range(CATEGORIES):
            x = LEFT_BUTTON_X if index < 3 else RIGHT_BUTTON_X
            u = (LEFT_BUTTON_U if index < 3 else RIGHT_BUTTON_U) + shift
            row0, row1 = BUTTON_ROWS[index % 3]
            y = top + BUTTON_TOPS[index % 3]
            color = hud.WHITE if index == self.category else hud.DIM
            result.append(hud.quad(depth * BUTTON_DEPTH, (x, y, x + BUTTON_SIZE, y + BUTTON_SIZE),
                                   (u, row0, u + BUTTON_TEXEL_WIDTH, row1), color, texture_name, 'category %d' % index))
        x0 = SLOTS_LEFT - fraction * SLOT_SIZE
        result.append(hud.tiles(depth * STRIP_DEPTH, (x0, top, x0 + SLOT_CELLS * SLOT_SIZE, top + STRIP_HEIGHT),
                                SLOT_TEXELS, SLOT_CELLS, 1, texture_name=texture_name, name='slots'))
        for i in range(SLOT_CELLS):
            x = x0 + i * SLOT_SIZE
            result.append(hud.hollow_frame(depth, (x, top, x + SLOT_SIZE, top + STRIP_HEIGHT), SLOT_FRAME_BORDER,
                                           SLOT_TEXELS, texture_name=texture_name, name='slot frame'))
        return result

    def slots(self):
        """
        621b10: (item, screen x, screen y, depth) of the items in the visible slots.
        """
        items = self.shown_items()
        position = self.position
        first = int(position)
        fraction = position - first
        result = []
        for i in range(SLOT_CELLS):
            index = first + i
            if 0 <= index < len(items):
                x = (i - fraction + 1.0) * SLOT_SIZE
                result.append((items[index], x, self.top + STRIP_HEIGHT * 0.5, self.depth * ITEM_DEPTH))
        return result

    def regions(self):
        """
        620340: the arrows, the six buttons and the slots in this order, the tooltips of 621120 and 6213e0.
        """
        top = self.top
        result = [
            hud.Region(_rect_points(_moved(LEFT_ARROW, top)), TIP_SCROLL_RIGHT, 'scroll left'),
            hud.Region(_rect_points(_moved(RIGHT_ARROW, top)), TIP_SCROLL_LEFT, 'scroll right'),
        ]
        first_tip = MODE_FIRST_TIP[self.mode]
        for index in range(CATEGORIES):
            x = LEFT_BUTTON_X if index < 3 else RIGHT_BUTTON_X
            y = top + BUTTON_TOPS[index % 3]
            result.append(hud.Region(_rect_points((x, y, x + BUTTON_SIZE, y + BUTTON_SIZE)), first_tip + index,
                                     'category %d' % index))
        result.append(hud.Region(_rect_points((SLOTS_LEFT, top, SLOTS_RIGHT, top + STRIP_HEIGHT)), None, 'slots'))
        return result

    def on_click(self, action, jump=False):
        """
        True when the action belongs to the strip.
        """
        if action == 'scroll left':
            self.press_arrow(-1, jump)
        elif action == 'scroll right':
            self.press_arrow(1, jump)
        elif action is not None and action.startswith('category '):
            self.choose_category(int(action.split()[1]))
        elif action != 'slots':
            return False
        return True

#------------------------------------------------------------------------------


def _moved(rect, top):
    x0, y0, x1, y1 = rect
    return (x0, y0 + top, x1, y1 + top)


def _rect_points(rect):
    x0, y0, x1, y1 = rect
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def item_category(item):
    """
    6218b0 for the mode of items: weapons and armors are ready to use; materials, keystones, collected
    and quest items are not read from items.idb yet.
    """
    if isinstance(item.shape, (idb.WeaponShape, idb.ArmorShape)):
        return CATEGORY_READY
    return CATEGORY_ALL


def item_kind(item):
    kinds = idb.WEAPON_KINDS if item.is_weapon else idb.ARMOR_KINDS
    if not 0 <= item.shape.type < len(kinds):
        return None
    return kinds[item.shape.type]


def item_model_name(item):
    """
    The loose figure of an item: "initweax1weapon", "initarbt0armor".
    """
    kind = item_kind(item)
    if kind is None:
        return None
    number = int(item.shape.number)
    if number < 0:
        return None
    if item.is_weapon:
        return ITEM_MODEL_FORMAT % ('we', kind, number, 'weapon')
    return ITEM_MODEL_FORMAT % ('ar', kind, number, 'armor')


def item_texture_name(item):
    kind = item_kind(item)
    if kind is None:
        return None
    return ITEM_TEXTURE_FORMAT % (kind, int(item.shape.number), item.shape.variant)


def item_texture(rgba):
    """
    5ec020: the texture of an item copied into the bottom rows of a 256 x 256 texture, white without it.
    """
    result = np.full((ITEM_TEXTURE_SIZE, ITEM_TEXTURE_SIZE, 4), 255, dtype=np.uint8)
    if rgba is None:
        return result
    height = min(rgba.shape[0], ITEM_TEXTURE_SIZE)
    width = min(rgba.shape[1], ITEM_TEXTURE_SIZE)
    result[:, :, 0:3] = 0
    result[ITEM_TEXTURE_SIZE - height:, 0:width] = rgba[0:height, 0:width]
    return result


def item_rotation():
    """
    5eca10, the strip: pi about x (the quaternion (w, x, y, z)), the y of the interface goes down.
    """
    return (0.0, 1.0, 0.0, 0.0)


def item_scale(width, height, depth):
    """
    The scale of the figure which makes its larger side ITEM_PIXELS pixels at the depth.
    """
    size = max(width, height, 1e-6)
    return hud_pixel(depth) * ITEM_PIXELS / size


def hud_pixel(depth):
    """
    One pixel of the 800 x 600 screen in the space of the interface at the depth (game.exe 5fc450).
    """
    return 2.0 / 800.0 * TAN_HALF_WIDTH * depth
