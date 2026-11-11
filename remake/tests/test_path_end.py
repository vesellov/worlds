"""
The leftovers of the movement over the grid of the paths: the cost of the last diagonal step of 5c46c0,
the tile types which tileDesc.reg does not list and the multiplier of the speed of the prototypes (5136d0).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import copy
import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import pathing
from engine import stats

#------------------------------------------------------------------------------


def open_grid(width, height):
    """
    A grid where every straight step costs 1024 and every diagonal one 1448, with its step tables.
    """
    grid = pathing.PassGrid.__new__(pathing.PassGrid)
    grid.width, grid.height = width, height
    tables = []
    for number, (dx, dy) in enumerate(pathing.DIRECTIONS):
        step = 1448 if number % 2 else 1024
        table = [-1] * (width * height)
        for y in range(height):
            for x in range(width):
                if 0 <= x + dx < width and 0 <= y + dy < height:
                    table[y * width + x] = step
        tables.append(table)
    return grid, tables

#------------------------------------------------------------------------------


class PathEndTest(unittest.TestCase):

    def test_target_factor(self):
        # the middle of a point (0.5 map units per point) is 1, its border 0
        self.assertEqual(pathing.target_factor(10.25), 1.0)
        self.assertEqual(pathing.target_factor(10.0), 0.0)
        self.assertEqual(pathing.target_factor(10.5), 0.0)
        self.assertAlmostEqual(pathing.target_factor(10.125), 0.5)

    def test_last_diagonal_step_costs_on_the_border(self):
        grid, tables = open_grid(6, 6)
        diagonal = [(0, 0), (1, 1), (2, 2), (3, 3)]
        middle = grid.smooth(tables, diagonal, end_factor=1.0)
        self.assertEqual(middle, diagonal)
        border = grid.smooth(tables, diagonal, end_factor=0.0)
        self.assertEqual(border[0], diagonal[0])
        self.assertEqual(border[-1], diagonal[-1])
        (ax, ay), (bx, by) = border[-2], border[-1]
        self.assertTrue(ax == bx or ay == by)


class TileTypesTest(unittest.TestCase):

    def test_unlisted_types(self):
        registry = {'0': {'Name': 'GRASS', 'Speed': 1024, 'CostMul': 2},
                    '13': {'Name': 'Lava', 'Speed': -1, 'CostMul': 1}}
        tiles = pathing.tile_descriptions(registry)
        self.assertEqual((tiles[0].speed, tiles[0].cost), (1024, 2))
        # 5b43c0 fills the types which the file does not list with Speed -1 and CostMul 1
        self.assertEqual((tiles[15].speed, tiles[15].cost), (pathing.UNLISTED_TILE_SPEED, pathing.UNLISTED_TILE_COST))
        self.assertEqual(tiles[15].cost, 1)
        # no file at all: the defaults of the remake
        self.assertEqual(pathing.tile_descriptions(None)[15].cost, 2)


class PrototypeSpeedTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def test_speed_scale_of_the_prototypes(self):
        units = self.game_data.units()
        self.assertTrue(all(p.speed_scale == 1.0 for p in units.prototypes))
        prototype = copy.copy(units.prototype('Human Hero'))
        prototype.speed_scale = 1.25
        self.assertEqual(stats.from_prototype(prototype).speed_scale, 1.25)


if __name__ == '__main__':
    unittest.main()
