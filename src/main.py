import os
import sys
import json

from kivy.config import Config
from kivy.core.window import Window
from kivy.app import App

import res
import rend
import dat
import scen

Window.size = (1400, 700)
Window.top = 100
Window.left = 100

_Debug = True


class AppRoot(App):

    known_templates = {}

    def build(self):
        land = dat.LandData()
        land.load_heightmap_file(heightmap_file_name='assets/heightmap.png')
        land.load_tilemap_file(tilemap_file_name='assets/encoded.png')
        land.load_cache_tiles_textures(textures_dir_path='assets/land')
        land.load_plants_data(plants_data_file_name='assets/plants.json')
        land.load_buildings_data(buildings_data_file_name='assets/buildings.json')
        scene = scen.Scene(land=land)
        scene.calculate_land_vertices()
        scene.calculate_scaled_elevation_map()
        renderer = rend.Renderer(app_root=self, scene=scene)
        self.known_templates = json.loads(open('assets/templates.json', 'rt').read())
        scene.renderer = renderer
        # self.test_id = sorted(self.known_templates.keys()).index
        # scene.init_scene(117, 835)
        # scene.init_scene(500, 550)
        # scene.init_scene(831,383)
        scene.init_scene(369,313)
        # scene.init_scene()
        return renderer


def main():
    res.download_res_file('data', 'figures.res', ['figures_res_0', 'figures_res_1', ])
    res.download_res_file('data', 'textures.res', ['textures_res_0', 'textures_res_1', 'textures_res_2', ])
    res.download_res_file('data', 'redress.res', ['redress_res_0', 'redress_res_1', ])
    AppRoot().run()


if __name__ == '__main__':
    main()
