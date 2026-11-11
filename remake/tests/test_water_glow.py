"""
G8: the self illumination of the materials of the liquids (6c6870) goes into the vertices of the water
and the shader adds it to the lit color, so the lava of zone16 and the red water of the camps glow at night.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import unittest

#------------------------------------------------------------------------------

from eilib import gamedata
from engine import terrain
from engine import water

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------


class WaterGlowTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.game_data = gamedata.GameData()

    def test_material_glow(self):
        header = self.game_data.map_header('zone16')
        material = next(m for m in header.materials if m['self_illumination'] > 0.0)
        r, g, b, _ = material['color']
        self.assertEqual(material['glow'], (min(1.0, r), min(1.0, g), min(1.0, b)))

    def test_glow_vertices(self):
        land = terrain.Terrain(self.game_data, 'zone16')
        meshes = water.sector_meshes(land, 2, 0, water.Wind((1.0, 0.0, 0.0), 0.5), water.random_phases())
        totals = []
        for (texture_index, material_index), (vertices, _) in meshes.items():
            material = land.header.materials[material_index]
            if material['self_illumination'] <= 0.0:
                continue
            self.assertTrue((vertices[:, 19:22] > 0.0).any())
            totals.append(vertices[:, 19:22].max())
        self.assertTrue(totals)


if __name__ == '__main__':
    unittest.main()
