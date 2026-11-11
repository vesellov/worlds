"""
The paths of the units on the land, as the server of game.exe finds them.

The grid. 5b43c0 splits every cell of the map into 2 x 2 points (64 x 64 points per sector).
The height of a point leans to its corner of the cell: the corner gives 1/2, the two neighbor
corners 1/4 each. Heights are kept in 1/511 of the highest altitude of the .mp; the depth of
the liquid over the land is kept in the same steps, up to 63. Every point also takes the type of
its tile, the types are those of tileDesc.reg (res/aiinfo.res): Speed and CostMul of every type (5b43c0 gives
a type which the file does not list Speed -1 and CostMul 1). Only CostMul is used: the server of game.exe
never reads Speed back (nor does the client), so the tiles do not change the speed of the units.

The codes. Every point has a code 0..15 for each of 8 classes of units (4 bits of a dword).
The classes differ by the height of the unit (7912bc): 2.0, 0.4, 1.0, 1.6, 2.2, 1.7, 3.0, 6.0.
units.udb gives the class of a standing and of a lying unit of every type
(the tags 9 and 10): 3 for the humans, 1 for the wolves, 7 for the trolls, 0 for the flying units.
5b8ba0 computes a code from a price, which starts at 0x400, and a cost, which starts at the CostMul of the tile:
    - the class 0 ignores the liquids and has the cost 1;
    - a liquid deeper than the height of the unit minus one step, or of 63, closes the point;
      deeper than the half of the height gives the price 800 and 30 to the cost, else the price is 0x370;
      the class 1 is closed by any liquid;
    - the price below 0 or the cost above 70 is the code 0 (closed);
    - else the code comes from the table 791164: the column is min(15, (price + 50) / 102),
      the row is 0 for the cost below 2, 1 below 3, 2 below 10, 3 up to 34 and 4 above.
5b89f0 closes a point when one of its neighbors was closed for the class: the cross of 5 points
for the classes 0..4, a diamond of the radius 2 for the classes 5 and 6, of the radius 3 for 7.
5b8710 looks at the steepest of the 4 neighbors: above 60 degrees the point closes for all the classes,
above 40 degrees for all but the class 0; it also closes the classes 5 and 6 at the 4 neighbors and
the class 7 at 13 points around. 5b8fa0 closes 3 points at the borders of the map.

The cost of a step. 5c46c0: the cost of the code of the next point (5b6070: 0 is closed, 14 is 1024,
1 is 25600...) times the cost of the change of the height, >> 10. The diagonal steps are x 0x5a8 >> 10.
The cost of the height is a table of the slope: for the class 0, 1024 up to 60 degrees; for the others, up to
40 degrees, (sin^2 + 1)^2 * 1024 uphill and 1024 / (sin^2 + 1) downhill (the step of a point is 0.5).

The search. 5bb350 tries the straight line first (5c2420), moves a target which cannot be reached to the nearest
open point of the region of the unit (5bac10: rings around the target, first on the way to the unit),
then searches (5b9c90): a coarse path over blocks of 8 x 8 points (5b0d90), refined window by window by
a Dijkstra in 32 x 32 points (5c25d0), and smoothed over the 8 directions (5c46c0). Here the same costs are
searched by one A* over the whole grid and the path is straightened by the straight lines.

The objects. 5b6a80 puts every part of an object on the grid. 5179e0 gives a part flags by its name:
"BASE..." is a floor (0x400), "CROWN..." is the kind 2 (0x90), "EMPTY..." is skipped (0x890), the other parts
are the kind 0. The 6 faces of the box of the part (.fig bounds of the proportions, placed by the pose of the part
and the object; the object stands on the height of its grid point, 5b77f0, or on the land kept for a point
under a floor, 5b3160, which is the land under the object within 0.02 here) are sampled about 3 times per cell; a face whose normal looks
up gives the highest height of its points (5b2d10), the others the lowest (5b2e10), in the steps of the grid, up
to 0x3fc. A point which only faces of one sort cover keeps the low 0x7fffffff or the high 0 and stops nobody: so
the sides of a turned box do not make walls around it.
A floor (5b6740) raises the point to its top, makes it a road (tile type 7) and drops the liquid.
In 5b8ba0 an obstacle of the kind 0 or 1 which overlaps the column of the unit (from the land, from the liquid
for the class 0, up to the height of the class) by the part f of that height adds 290 f (20 f) to the cost and
multiplies the price by 0.6 ^ (7 f) (0.6 ^ (2 f)); the kind 3 doubles the cost, the kinds 2 and 4 change nothing.
So a trunk closes the points under it for every class whose column it overlaps by about a quarter.

The turns. 5c46c0 smooths the path over the 8 directions and adds (n - 1) * 10 * 60 for a turn by n * 45 degrees
(n > 1), counting from the way the unit looks; here the A* path is searched again with these costs
over the points next to it (the corridor of 5c46c0 is the pairs of directions of 7c3048, here 3 x 3).
The last step: after the smoothing, 5c46c0 adds (1 - k) * 2048 to the paths which come to the target
by a diagonal step, k = 2 * |x - round(x)| of the target in the points (5bb350 passes it, and it takes the x twice
where the y was surely meant: so only x counts). A target in the middle of its point (k = 1) costs nothing,
one on the border of the points prefers a straight last step.
Not here yet: the units as obstacles.
The lava is a liquid over the land, so it closes the points by its depth like the water.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import heapq
import math

import numpy as np

#------------------------------------------------------------------------------

from eilib import mapdata
from engine import quat

#------------------------------------------------------------------------------

POINTS_PER_CELL = 2

POINT_SIZE = 1.0 / POINTS_PER_CELL

HEIGHT_STEPS = 511

MAX_DEPTH = 0x3f

CLASSES_COUNT = 8

CLASS_HEIGHTS = (2.0, 0.4, 1.0, 1.6, 2.2, 1.7, 3.0, 6.0)

FLYING_CLASS = 0

HUMAN_CLASS = 3

START_PRICE = 0x400

DEEP_PRICE = 800

SHALLOW_PRICE = 0x370

DEEP_COST = 30

MAX_COST = 70

# 791164: the codes by the row of the cost and the column of the price
CODE_TABLE = (
    (0, 0, 1, 1, 1, 1, 4, 7, 11, 12, 14, 14, 15, 15, 15, 15),
    (0, 0, 1, 1, 1, 1, 4, 7, 11, 12, 13, 13, 13, 13, 13, 13),
    (0, 0, 1, 1, 1, 1, 4, 7, 8, 9, 10, 10, 10, 10, 10, 10),
    (0, 0, 1, 1, 1, 1, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6),
    (0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3),
)

# 5b6070: the cost of a step onto a point of a code, -1 is closed
CODE_COSTS = (-1, 0x6400, 0x77ec, 0x8bd8, 0x2ff8, 0x37f0, 0x3ffc, 0xdfc, 0xfff, 0x11fd, 0x1400,
              0x666, 0x732, 0x800, 0x400, 0x4cc)

# 5b9580: the table of the units which do not care about the land (+0x43c of the unit)
FREE_CODE_COSTS = (-1, ) + (0x400, ) * 15

WALK_SLOPE_DEGREES = 40.0

FLY_SLOPE_DEGREES = 60.0

DIAGONAL = 0x5a8

# 7912dc, 79130c, 79137c: the points around which a closed point closes
CROSS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
DIAMOND_2 = tuple((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) <= 2)
DIAMOND_3 = ((-3, 0), (3, 0), (0, -3), (0, 3)) + tuple(
    (dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) <= 4 and max(abs(dx), abs(dy)) <= 2)
CLASS_SPREADS = (CROSS, CROSS, CROSS, CROSS, CROSS, DIAMOND_2, DIAMOND_2, DIAMOND_3)

# 7916bc and 7916dc: the neighbors of the slope and the ring of the class 7
SLOPE_NEIGHBORS = ((0, 1), (0, -1), (1, 0), (-1, 0))
SLOPE_RING = ((0, 2), (-1, 1), (0, 1), (1, 1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1, -1), (0, -1), (1, -1),
              (0, -2))

BORDER_POINTS = 3

# 5c5cc0: the directions 1..8, the even ones are diagonal
DIRECTIONS = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))
STATES_PER_POINT = len(DIRECTIONS) + 1

# 5c46c0: a path which comes to the target by a diagonal step costs (1 - k) * 2048 more (73fe40)
DIAGONAL_END_COST = 2048.0

# 5c46c0: a turn costs (45-degree steps - 1) * +0x24 * 60, +0x24 is 10.0 (set by 5b9580)
TURN_STEP_FACTOR = 10.0
TURN_COST = int(round(TURN_STEP_FACTOR * 60.0))

DEFAULT_TILE = 15

# 5b43c0: Speed and CostMul of a tile type which tileDesc.reg does not list
UNLISTED_TILE_SPEED = -1
UNLISTED_TILE_COST = 1

# 5bac10: the turn of the rays by 45 degrees (73fdac)
TURN_45 = 0.707

# 5ba9d0 looks for a free point by the table of offsets of 5bfc40 sorted by the distance, its size is not read yet
PUSH_OUT_POINTS = 32

# limit -> the offsets of ring_offsets()
RING_OFFSETS = {}

KIND_SOLID = 0
KIND_LIGHT = 1
KIND_CROWN = 2
KIND_DOUBLE = 3
KIND_FLOOR = 4

# 7912a4, 7912b0: what the part of the height covered by an obstacle adds to the cost and the power of PRICE_BASE
OBSTACLE_COSTS = {KIND_SOLID: 290, KIND_LIGHT: 20}
OBSTACLE_POWERS = {KIND_SOLID: 7, KIND_LIGHT: 2}

# 73fd50, its logarithm is kept in 7c2f7c by 5b2fb0
PRICE_BASE = 0.6

MAX_OBSTACLE_HEIGHT = 0x3fc

# the low of a point which only the tops of a part cover (5b2d10)
NO_LOW = 0x7fffffff

# 5b6a80: samples of a face per unit of its length
FACE_SAMPLES = 1.5

ROAD_TILE = 7

#------------------------------------------------------------------------------


class TileDescription(object):

    def __init__(self, name, speed=1024, cost=2):
        self.name = name
        self.speed = speed
        self.cost = cost

    def __repr__(self):
        return 'TileDescription(%s, speed %d, cost %d)' % (self.name, self.speed, self.cost)

#------------------------------------------------------------------------------


class PassGrid(object):
    """
    The heights, the liquids and the codes of the points of a map for the 8 classes of units.
    """

    def __init__(self, terrain, tile_descriptions=None, parts=()):
        """
        parts: (name, corners) of the parts of the objects of the map, corners are the 8 corners of the box
        in the map, see object_parts().
        """
        self.terrain = terrain
        header = terrain.header
        self.max_altitude = header.max_altitude or 1.0
        self.scale = HEIGHT_STEPS / self.max_altitude
        self.width = header.sectors_x * mapdata.SECTOR_CELLS * POINTS_PER_CELL
        self.height = header.sectors_y * mapdata.SECTOR_CELLS * POINTS_PER_CELL
        self.tiles = tile_descriptions or default_tile_descriptions()
        self.heights, self.depths, self.tile_types = self.build_points()
        self.obstacles = self.put_parts(parts)
        self.codes = np.zeros((CLASSES_COUNT, self.height, self.width), dtype=np.uint8)
        for unit_class in range(CLASSES_COUNT):
            self.codes[unit_class] = spread_closed(self.class_codes(unit_class), CLASS_SPREADS[unit_class])
        self.close_slopes()
        self.close_borders()
        self._step_tables = {}

    def build_points(self):
        cells_x = self.width // POINTS_PER_CELL
        cells_y = self.height // POINTS_PER_CELL
        land = np.zeros((cells_y + 1, cells_x + 1))
        water = np.full((cells_y + 1, cells_x + 1), -1e9)
        tile_types = np.full((self.height, self.width), DEFAULT_TILE, dtype=np.int64)
        known_types = np.asarray(self.terrain.header.tile_types or [DEFAULT_TILE], dtype=np.int64)
        n = mapdata.SECTOR_CELLS
        points_per_tile = n * POINTS_PER_CELL // mapdata.SECTOR_TILES
        for (sx, sy), sector in self.terrain.sectors.items():
            land[sy * n:sy * n + n + 1, sx * n:sx * n + n + 1] = sector.land[:, :, 2]
            if sector.water is not None:
                water[sy * n:sy * n + n + 1, sx * n:sx * n + n + 1] = sector.water[:, :, 2]
            index = sector.land_tiles[:, :, 0] * 64 + sector.land_tiles[:, :, 1]
            types = np.where(index < len(known_types), known_types[np.minimum(index, len(known_types) - 1)],
                             DEFAULT_TILE)
            block = np.repeat(np.repeat(types, points_per_tile, axis=0), points_per_tile, axis=1)
            p = n * POINTS_PER_CELL
            tile_types[sy * p:sy * p + p, sx * p:sx * p + p] = block
        heights = np.rint(corner_points(land) * self.scale).astype(np.int64)
        surface = np.rint(corner_points(np.maximum(water, land)) * self.scale).astype(np.int64)
        depths = np.clip(surface - heights, 0, MAX_DEPTH)
        return heights, depths, tile_types & 0x1f

    def put_parts(self, parts):
        """
        The floors go into the heights and the tiles; the other parts become the obstacles:
        arrays of the points (x, y), the lowest and the highest heights and the kinds.
        """
        xs, ys, lows, highs, kinds = [], [], [], [], []
        for name, corners in parts:
            kind = part_kind(name)
            if kind is None:
                continue
            points, up = sample_box(np.asarray(corners, dtype=np.float64))
            # 5b6a80: round(2 x - 0.5), rounded to the even number as the FPU does
            px = np.rint(points[:, 0] * POINTS_PER_CELL - 0.5).astype(np.int64)
            py = np.rint(points[:, 1] * POINTS_PER_CELL - 0.5).astype(np.int64)
            inside = (px >= 0) & (py >= 0) & (px < self.width) & (py < self.height)
            if not inside.any():
                continue
            px, py, up = px[inside], py[inside], up[inside]
            pz = np.clip(np.rint(points[inside, 2] * self.scale), 0, MAX_OBSTACLE_HEIGHT).astype(np.int64)
            index = py * self.width + px
            order = np.argsort(index, kind='stable')
            index, pz, up = index[order], pz[order], up[order]
            unique, first = np.unique(index, return_index=True)
            # 5b2d10 makes the record of a point with the low 0x7fffffff, 5b2e10 with the high 0:
            # a point under the faces of one sort only stops nobody
            low = np.minimum.reduceat(np.where(up, NO_LOW, pz), first)
            high = np.maximum.reduceat(np.where(up, pz, 0), first)
            ux, uy = unique % self.width, unique // self.width
            if kind == KIND_FLOOR:
                self.heights[uy, ux] = np.maximum(self.heights[uy, ux], high)
                self.tile_types[uy, ux] = ROAD_TILE
                self.depths[uy, ux] = 0
                continue
            xs.append(ux)
            ys.append(uy)
            lows.append(low)
            highs.append(high)
            kinds.append(np.full(len(unique), kind, dtype=np.int64))
        if not xs:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, empty, empty
        return tuple(np.concatenate(a) for a in (xs, ys, lows, highs, kinds))

    def unit_height(self, unit_class):
        return int(round(CLASS_HEIGHTS[unit_class] * self.scale))

    def class_codes(self, unit_class):
        """
        The codes of 5b8ba0 for the land and the liquids, before the closed points spread.
        """
        costs = np.array([self.tiles[t].cost if t in self.tiles else 2 for t in range(32)], dtype=np.int64)
        price = np.full(self.heights.shape, START_PRICE, dtype=np.int64)
        if unit_class == FLYING_CLASS:
            cost = np.ones(self.heights.shape, dtype=np.int64)
        else:
            cost = costs[self.tile_types]
            height = self.unit_height(unit_class) if unit_class != 1 else 0
            depth = self.depths
            wet = depth != 0
            closed = wet & ((depth == MAX_DEPTH) | (height - 1 <= depth))
            if height < 0x400:
                deep = wet & ~closed & (depth > height // 2)
                shallow = wet & ~closed & ~deep
                price[deep] = DEEP_PRICE
                cost = cost + np.where(deep, DEEP_COST, 0)
                price[shallow] = SHALLOW_PRICE
            price[closed] = -1
        cost = np.array(cost, dtype=np.int64)
        doubled = self.add_obstacles(unit_class, price, cost)
        cost = np.where(doubled, cost * 2, cost)
        column = np.minimum(15, (price + 50) // 102)
        row = np.where(cost < 2, 0, np.where(cost < 3, 1, np.where(cost < 10, 2, np.where(cost <= 34, 3, 4))))
        codes = np.asarray(CODE_TABLE, dtype=np.uint8)[row, np.clip(column, 0, 15)]
        codes[(price < 0) | (cost > MAX_COST)] = 0
        return codes

    def add_obstacles(self, unit_class, price, cost):
        """
        5b8ba0: the obstacles over the column of the unit change the price and the cost in place;
        returns the points where the kind 3 doubles the cost.
        """
        xs, ys, lows, highs, kinds = self.obstacles
        doubled = np.zeros(price.shape, dtype=bool)
        if len(xs) == 0:
            return doubled
        height = self.unit_height(unit_class)
        bottom = self.heights[ys, xs]
        if unit_class == FLYING_CLASS:
            bottom = bottom + self.depths[ys, xs]
        top = bottom + height
        for kind in (KIND_SOLID, KIND_LIGHT):
            mask = (kinds == kind) & (lows <= top) & (bottom <= highs)
            if not mask.any() or height <= 0:
                continue
            part = (np.minimum(highs, top) - np.maximum(lows, bottom))[mask] / float(height)
            x, y = xs[mask], ys[mask]
            np.add.at(cost, (y, x), np.rint(OBSTACLE_COSTS[kind] * part).astype(np.int64))
            factors = np.ones(price.shape)
            np.multiply.at(factors, (y, x), PRICE_BASE ** (OBSTACLE_POWERS[kind] * part))
            open_price = price >= 0
            price[open_price] = np.rint(price[open_price] * factors[open_price]).astype(np.int64)
        mask = (kinds == KIND_DOUBLE) & (lows <= top) & (bottom <= highs)
        doubled[ys[mask], xs[mask]] = True
        return doubled

    def close_slopes(self):
        """
        5b8710: the steep points close, and the big classes keep away from them.
        """
        h = self.heights
        steepest = np.zeros(h.shape, dtype=np.int64)
        for dx, dy in SLOPE_NEIGHBORS:
            steepest = np.maximum(steepest, np.abs(shift(h, dx, dy, edge=True) - h))
        walk_limit = slope_limit(WALK_SLOPE_DEGREES, self.scale)
        fly_limit = slope_limit(FLY_SLOPE_DEGREES, self.scale)
        too_steep = steepest >= fly_limit
        steep = (steepest >= walk_limit) & ~too_steep
        inner = np.zeros(h.shape, dtype=bool)
        inner[2:-2, 2:-2] = True
        too_steep &= inner
        steep &= inner
        self.codes[:, too_steep] = 0
        self.codes[1:, steep] = 0
        either = too_steep | steep
        near = np.zeros(h.shape, dtype=bool)
        for dx, dy in SLOPE_NEIGHBORS:
            near |= shift(either, -dx, -dy)
        self.codes[5:7, near] = 0
        ring = np.zeros(h.shape, dtype=bool)
        for dx, dy in SLOPE_RING:
            ring |= shift(either, -dx, -dy)
        self.codes[7, ring] = 0

    def close_borders(self):
        b = BORDER_POINTS
        self.codes[:, :b, :] = 0
        self.codes[:, -b:, :] = 0
        self.codes[:, :, :b] = 0
        self.codes[:, :, -b:] = 0

    def point(self, x, y):
        return (min(max(int(math.floor(x * POINTS_PER_CELL)), 0), self.width - 1),
                min(max(int(math.floor(y * POINTS_PER_CELL)), 0), self.height - 1))

    def position(self, px, py):
        return ((px + 0.5) * POINT_SIZE, (py + 0.5) * POINT_SIZE)

    def is_open(self, unit_class, px, py):
        return 0 <= px < self.width and 0 <= py < self.height and self.codes[unit_class, py, px] != 0

    def step_tables(self, unit_class, free=False):
        """
        8 flat lists (one per number - 1 of DIRECTIONS) of the cost of the step from a point (y * width + x)
        to its neighbor, -1 where the step is closed: the cost of the code of the neighbor
        times the cost of the slope >> 10, a diagonal step times DIAGONAL >> 10 (5c46c0).
        """
        key = (unit_class, free)
        if key not in self._step_tables:
            slopes = SlopeCosts(unit_class, self.scale)
            limit = MAX_OBSTACLE_HEIGHT + 1
            slope_table = np.array([slopes(d) for d in range(-limit, limit + 1)], dtype=np.int64)
            code_costs = np.asarray(FREE_CODE_COSTS if free else CODE_COSTS, dtype=np.int64)
            heights = self.heights.astype(np.int64)
            tables = []
            for number, (dx, dy) in enumerate(DIRECTIONS):
                cost = code_costs[shift(self.codes[unit_class], dx, dy)]
                delta = np.clip(shift(heights, dx, dy, edge=True) - heights, -limit, limit)
                slope = slope_table[delta + limit]
                step = (slope * cost) >> 10
                if number % 2 == 1:
                    step = (step * DIAGONAL) >> 10
                step[(cost < 0) | (slope < 0)] = -1
                tables.append(step.ravel().tolist())
            self._step_tables[key] = tables
        return self._step_tables[key]

    def find_path(self, unit_class, start, goal, free=False, direction=None):
        """
        The points of the map (x, y) from the start to the goal, None when there is no path.
        The goal may move to the nearest open point. direction is the number (1..8, see DIRECTIONS) of the way
        the unit looks at the start: the first step pays for turning away from it too.
        """
        sx, sy = self.point(*start)
        goal_point = self.point(*goal)
        end_factor = target_factor(goal[0])
        tables = self.step_tables(unit_class, free)
        slopes = SlopeCosts(unit_class, self.scale)
        code_costs = FREE_CODE_COSTS if free else CODE_COSTS
        cheapest = slopes.cheapest() * min(c for c in code_costs if c > 0) >> 10
        points = reached = None
        if self.is_open(unit_class, *goal_point):
            points, reached = self.search(tables, cheapest, (sx, sy), goal_point)
        if points is None:
            if reached is None:
                reached = self.reachable(tables, (sx, sy))
            target = self.reachable_target(unit_class, (sx, sy), goal_point, reached)
            if target is None or target == (sx, sy):
                return None
            points, _ = self.search(tables, cheapest, (sx, sy), target)
            if points is None:
                return None
            # the target moved to the middle of an open point
            end_factor = 1.0
        gx, gy = points[-1]
        points = self.smooth(tables, points, direction, end_factor)
        straight = self.straighten(unit_class, points)
        return [self.position(px, py) for px, py in straight[1:]] if len(straight) > 1 else [self.position(gx, gy)]

    def search(self, tables, cheapest, start, goal):
        """
        A* over the points by the costs of the steps (step_tables()): the points from the start to the goal
        (None when it is not reached) and the indexes of the points which the search reached (all the points
        which can be reached when the goal is not).
        """
        width = self.width
        gx, gy = goal
        start_index = start[1] * width + start[0]
        goal_index = gy * width + gx
        moves = [(number, dy * width + dx, dx, dy) for number, (dx, dy) in enumerate(DIRECTIONS)]
        diagonal_extra = DIAGONAL - 1024
        heappush, heappop = heapq.heappush, heapq.heappop
        came = {start_index: -1}
        best = {start_index: 0}
        queue = [(0, 0, start_index)]
        while queue:
            _, spent, index = heappop(queue)
            if index == goal_index:
                break
            if spent > best.get(index, spent):
                continue
            px, py = index % width, index // width
            for number, offset, dx, dy in moves:
                step = tables[number][index]
                if step < 0:
                    continue
                total = spent + step
                n_index = index + offset
                if total < best.get(n_index, total + 1):
                    best[n_index] = total
                    came[n_index] = index
                    ax, ay = abs(px + dx - gx), abs(py + dy - gy)
                    if ax < ay:
                        ax, ay = ay, ax
                    heappush(queue, (total + cheapest * (ax * 1024 + ay * diagonal_extra) // 1024, total, n_index))
        if goal_index not in came:
            return None, came
        points = []
        index = goal_index
        while index != -1:
            points.append((index % width, index // width))
            index = came[index]
        points.reverse()
        return points, came

    def reachable(self, tables, start):
        """
        The indexes of the points which can be reached from the start by the steps of step_tables().
        """
        width = self.width
        offsets = [dy * width + dx for dx, dy in DIRECTIONS]
        start_index = start[1] * width + start[0]
        seen = {start_index}
        queue = [start_index]
        while queue:
            index = queue.pop()
            for number, offset in enumerate(offsets):
                if tables[number][index] < 0:
                    continue
                n_index = index + offset
                if n_index not in seen:
                    seen.add(n_index)
                    queue.append(n_index)
        return seen

    def reachable_target(self, unit_class, start, goal, reached):
        """
        5bac10 (with the search of the region): the target of a path which cannot be reached moves to the open
        point which can, on the rings around the target from the nearest one (up to the distance to the unit),
        first on the way to the unit, then turned by 45, -45, -90, 90, 135, -135 and 180 degrees from it.
        """
        ux, uy = start
        gx, gy = goal
        dx, dy = gx - ux, gy - uy
        distance = max(abs(dx), abs(dy))
        if distance == 0:
            return start
        if abs(dy) < abs(dx):
            vx, vy = math.copysign(1.0, dx), dy / float(abs(dx))
        else:
            vx, vy = dx / float(abs(dy)), math.copysign(1.0, dy)
        ax, ay = (vx - vy) * TURN_45, (vx + vy) * TURN_45
        scale = 1.0 / max(abs(ax), abs(ay))
        if scale < 1.0:
            ax, ay, vx, vy = ax * scale, ay * scale, vx * scale, vy * scale
        rays = ((vx, vy), (ax, ay), (ay, -ax), (vy, -vx), (-vy, vx), (-ay, ax), (-ax, -ay), (-vx, -vy))
        width = self.width
        for r in range(distance + 1):
            for rx, ry in rays if r else rays[:1]:
                px = gx - int(round(r * rx))
                py = gy - int(round(r * ry))
                if self.is_open(unit_class, px, py) and py * width + px in reached:
                    return px, py
        return None

    def nearest_open(self, unit_class, x, y, limit=PUSH_OUT_POINTS):
        """
        5ba9d0: the open point nearest to x, y (the point itself first) by the offsets sorted by the distance,
        as a position; None when there is none within limit points.
        """
        cx, cy = self.point(x, y)
        for dx, dy in ring_offsets(limit):
            if self.is_open(unit_class, cx + dx, cy + dy):
                return self.position(cx + dx, cy + dy)
        return None

    def smooth(self, tables, points, direction=None, end_factor=1.0):
        """
        The path again over the points next to the found one (5c46c0): a state is a point and the direction
        of the step to it, so the turns pay turn_cost(); the first step turns from the direction of the unit.
        A diagonal step onto the last point costs (1 - end_factor) * DIAGONAL_END_COST more (target_factor()).
        """
        if len(points) < 3:
            return points
        width = self.width
        corridor = set()
        for px, py in points:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = px + dx, py + dy
                    if 0 <= x < width and 0 <= y < self.height:
                        corridor.add(y * width + x)
        moves = [(number, dy * width + dx) for number, (dx, dy) in enumerate(DIRECTIONS)]
        turns = [[turn_cost(previous, number + 1) for number in range(len(DIRECTIONS))]
                 for previous in range(STATES_PER_POINT)]
        goal_index = points[-1][1] * width + points[-1][0]
        end_cost = int(round((1.0 - end_factor) * DIAGONAL_END_COST))
        start_state = (points[0][1] * width + points[0][0]) * STATES_PER_POINT + (direction or 0)
        came = {start_state: -1}
        best = {start_state: 0}
        queue = [(0, start_state)]
        found = None
        heappush, heappop = heapq.heappush, heapq.heappop
        while queue:
            spent, state = heappop(queue)
            index, came_direction = divmod(state, STATES_PER_POINT)
            if index == goal_index:
                found = state
                break
            if spent > best.get(state, spent):
                continue
            turn_row = turns[came_direction]
            for number, offset in moves:
                step = tables[number][index]
                if step < 0:
                    continue
                n_index = index + offset
                if n_index not in corridor:
                    continue
                total = spent + step + turn_row[number]
                if n_index == goal_index and number % 2 == 1:
                    total += end_cost
                n_state = n_index * STATES_PER_POINT + number + 1
                if total < best.get(n_state, total + 1):
                    best[n_state] = total
                    came[n_state] = state
                    heappush(queue, (total, n_state))
        if found is None:
            return points
        result = []
        state = found
        while state != -1:
            index = state // STATES_PER_POINT
            result.append((index % width, index // width))
            state = came[state]
        result.reverse()
        return result

    def line_is_open(self, unit_class, a, b):
        """
        All the points under the straight line from a to b are open (5c2420 checks the straight path).
        """
        for x, y in line_points(a, b):
            if not self.is_open(unit_class, x, y):
                return False
        return True

    def straighten(self, unit_class, points):
        """
        Drops the points which the straight lines between the others pass by.
        """
        if len(points) < 3:
            return points
        result = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            far = len(points) - 1
            while far > anchor + 1 and not self.line_is_open(unit_class, points[anchor], points[far]):
                far -= 1
            result.append(points[far])
            anchor = far
        return result

#------------------------------------------------------------------------------


class SlopeCosts(object):
    """
    The cost of a step by the change of the height in the steps of the grid (the tables +0xd8 and +0xec of 5b43c0).
    """

    def __init__(self, unit_class, scale):
        self.unit_class = unit_class
        self.scale = scale
        self.table = {}

    def __call__(self, delta):
        value = self.table.get(delta)
        if value is None:
            value = slope_cost(delta, self.scale, self.unit_class == FLYING_CLASS)
            self.table[delta] = value
        return value

    def cheapest(self):
        limit = slope_limit(FLY_SLOPE_DEGREES if self.unit_class == FLYING_CLASS else WALK_SLOPE_DEGREES, self.scale)
        return min(self(-d) for d in range(0, max(limit, 1)))

#------------------------------------------------------------------------------


def part_kind(part_name):
    """
    The kind of the obstacle of a part of an object by its name (5179e0), None for the parts which are skipped.
    """
    name = part_name.upper()
    if name.startswith('EMPTY'):
        return None
    if name.startswith('CROWN'):
        return KIND_CROWN
    if name.startswith('BASE'):
        return KIND_FLOOR
    return KIND_SOLID


def object_parts(figure_model, world_matrix):
    """
    (name, corners) of the visible parts of a figure placed by the 4x4 world_matrix (the land under the object
    already in it): the 8 corners of the .fig box of every part in the map.
    """
    result = []
    pose = figure_model.skeleton.pose({}, 0.0)
    for part_name in figure_model.skeleton.visible:
        fig = figure_model.figure(part_name)
        if fig is None or fig.min is None:
            continue
        low, high = fig.variant_bounds(figure_model.coefs)
        if np.any(low > high):
            # a part without vertices keeps +FLT_MAX..-FLT_MAX
            continue
        position, rotation = pose[part_name]
        part_matrix = np.identity(4)
        part_matrix[0:3, 0:3] = quat.to_matrix(rotation)
        part_matrix[0:3, 3] = position
        matrix = np.asarray(world_matrix) @ part_matrix
        corners = np.array([(x, y, z, 1.0) for x in (low[0], high[0]) for y in (low[1], high[1])
                            for z in (low[2], high[2])])
        result.append((part_name, (corners @ matrix.T)[:, :3]))
    return result


def sample_box(corners):
    """
    Points on the 6 faces of a box given by its 8 corners (in the order of object_parts()),
    about FACE_SAMPLES per unit of the length of a face side and never fewer than 3 (5b6a80),
    and for every point whether its face looks up: such faces give the tops of the obstacle (5b2d10),
    the others the bottoms (5b2e10).
    """
    faces = ((0, 1, 2), (4, 5, 6), (0, 1, 4), (2, 3, 6), (0, 2, 4), (1, 3, 5))
    center = corners.mean(axis=0)
    samples, ups = [], []
    for origin, a, b in faces:
        o = corners[origin]
        u = corners[a] - o
        v = corners[b] - o
        normal = np.cross(u, v)
        if np.dot(normal, o + (u + v) * 0.5 - center) < 0:
            normal = -normal
        nu = int(round(2.0 * math.hypot(u[0], u[1]) * FACE_SAMPLES)) + 2
        nv = int(round(2.0 * math.hypot(v[0], v[1]) * FACE_SAMPLES)) + 2
        tu = (np.arange(nu) + 0.5) / nu
        tv = (np.arange(nv) + 0.5) / nv
        grid_u, grid_v = np.meshgrid(tu, tv)
        samples.append(o + grid_u.reshape(-1, 1) * u + grid_v.reshape(-1, 1) * v)
        ups.append(np.full(nu * nv, normal[2] > 1e-6))
    return np.concatenate(samples), np.concatenate(ups)


def default_tile_descriptions():
    return {i: TileDescription(name) for i, name in enumerate(mapdata.TILE_TYPES)}


def tile_descriptions(registry):
    """
    TileDescription of every tile type of tileDesc.reg (as parsed by eilib.reg), by the number of the type.
    """
    if not registry:
        return default_tile_descriptions()
    result = {i: TileDescription(name, UNLISTED_TILE_SPEED, UNLISTED_TILE_COST)
              for i, name in enumerate(mapdata.TILE_TYPES)}
    for section, keys in registry.items():
        try:
            number = int(section)
        except ValueError:
            continue
        result[number] = TileDescription(keys.get('Name', ''), int(keys.get('Speed', 1024)), int(keys.get('CostMul', 2)))
    return result


def corner_points(z):
    """
    The 2 x 2 points of every cell from the heights of its corners (5b43c0): the corner gives 1/2,
    the two neighbor corners 1/4 each.
    """
    h00, h10, h01, h11 = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    rows, cols = h00.shape
    out = np.zeros((rows * 2, cols * 2))
    out[0::2, 0::2] = h00 * 0.5 + (h10 + h01) * 0.25
    out[0::2, 1::2] = h10 * 0.5 + (h00 + h11) * 0.25
    out[1::2, 0::2] = h01 * 0.5 + (h00 + h11) * 0.25
    out[1::2, 1::2] = h11 * 0.5 + (h10 + h01) * 0.25
    return out


def slope_degrees(delta, scale):
    return math.degrees(math.atan2(abs(delta) / scale, POINT_SIZE))


def slope_cost(delta, scale, flying=False):
    degrees = slope_degrees(delta, scale)
    if flying:
        return 0x400 if degrees <= FLY_SLOPE_DEGREES else -1
    if degrees > WALK_SLOPE_DEGREES:
        return -1
    k = math.sin(math.radians(degrees)) ** 2 + 1.0
    if delta < 0:
        return int(round(1024.0 / k))
    return int(round(1024.0 * k * k))


def slope_limit(degrees, scale):
    """
    The first change of the height in steps which is steeper than the angle.
    """
    return int(math.floor(math.tan(math.radians(degrees)) * POINT_SIZE * scale)) + 1


def shift(array, dx, dy, edge=False):
    """
    array[y + dy, x + dx] at every (x, y): the neighbor in the direction; the borders repeat or are zero.
    """
    out = np.empty_like(array) if edge else np.zeros_like(array)
    h, w = array.shape
    if edge:
        out[...] = array
    ys = slice(max(0, -dy), h - max(0, dy))
    xs = slice(max(0, -dx), w - max(0, dx))
    ys2 = slice(max(0, dy), h - max(0, -dy))
    xs2 = slice(max(0, dx), w - max(0, -dx))
    out[ys, xs] = array[ys2, xs2]
    return out


def spread_closed(codes, offsets):
    """
    5b89f0: a point closes when the point at one of the offsets around it is closed.
    """
    closed = codes == 0
    near = np.zeros(codes.shape, dtype=bool)
    for dx, dy in offsets:
        near |= shift(closed, dx, dy)
    result = codes.copy()
    result[near] = 0
    return result


def target_factor(x):
    """
    k of the cost of the last diagonal step (5bb350 -> 5c46c0): 2 * |p - round(p)| of the x of the target
    in the points, 1 in the middle of a point, 0 on its border.
    """
    p = x * POINTS_PER_CELL
    return min(1.0, 2.0 * abs(p - round(p)))


def turn_cost(previous, direction):
    """
    The cost of turning from one direction of the steps to another (5c46c0): nothing for a turn by 45 degrees,
    TURN_COST for every further 45 degrees; nothing for the first step without a known direction.
    """
    if not previous:
        return 0
    turn = abs(direction - previous)
    if turn > 4:
        turn = 8 - turn
    return max(0, turn - 1) * TURN_COST


def direction_number(yaw):
    """
    The number of the direction of DIRECTIONS (1..8) nearest to the angle yaw (radians, atan2(dy, dx) of the map).
    """
    best = None
    for number, (dx, dy) in enumerate(DIRECTIONS):
        delta = abs((math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi)
        if best is None or delta < best[0]:
            best = (delta, number + 1)
    return best[1]


def ring_offsets(limit):
    """
    The offsets of the points within limit, from the nearest; equal distances go in the order of the rows.
    """
    if limit not in RING_OFFSETS:
        offsets = [(dx, dy) for dy in range(-limit, limit + 1) for dx in range(-limit, limit + 1)
                   if dx * dx + dy * dy <= limit * limit]
        offsets.sort(key=lambda d: d[0] * d[0] + d[1] * d[1])
        RING_OFFSETS[limit] = offsets
    return RING_OFFSETS[limit]


def line_points(a, b):
    """
    The points of the line from a to b by the steps of the 8 directions (Bresenham).
    """
    (x, y), (bx, by) = a, b
    dx, dy = abs(bx - x), -abs(by - y)
    step_x = 1 if x < bx else -1
    step_y = 1 if y < by else -1
    error = dx + dy
    points = [(x, y)]
    while (x, y) != (bx, by):
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x += step_x
        if doubled <= dx:
            error += dx
            y += step_y
        points.append((x, y))
    return points


def octile(dx, dy):
    dx, dy = abs(dx), abs(dy)
    return max(dx, dy) * 1024 + min(dx, dy) * (DIAGONAL - 1024)
