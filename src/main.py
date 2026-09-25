import os
import sys
import json

import system


ROOT_PATH = ''

if system.is_android():
    ROOT_PATH = os.path.abspath(os.environ['ANDROID_ARGUMENT'])
elif system.is_osx():
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
elif system.is_ios():
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
else:
    ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


_Debug = True


if _Debug:
    print(f'MAIN: ROOT_PATH={ROOT_PATH}')


from kivy.config import Config
# Config.set('graphics', 'position', 'custom')
Config.set('graphics', 'top', '100')
Config.set('graphics', 'left', '100')
Config.set('graphics', 'width', '1400')
Config.set('graphics', 'height', '700')


from kivy.core.window import Window
from kivy.app import App

import res
import rend
import dat
import scen

# Window.size = (1400, 700)
# Window.top = 100
# Window.left = 100


class AppRoot(App):

    known_templates = {}
    known_figures_parts = {}

    def build(self):
        catalog = dat.CatalogData()
        catalog.load_figures(figures_file_name='catalog/figures.json')
        catalog.load_animations(animations_file_name='catalog/animations.json')
        catalog.load_armors(armors_file_name='catalog/armors.json')
        catalog.load_weapons(weapons_file_name='catalog/weapons.json')
        catalog.load_materials(materials_file_name='catalog/materials.json')
        land = dat.LandData()
        land.load_heightmap_file(heightmap_file_name='assets/heightmap.png')
        land.load_tilemap_file(tilemap_file_name='assets/encoded.png')
        land.load_cache_tiles_textures(textures_dir_path='assets/land')
        land.load_plants_data(plants_data_file_name='assets/plants.json')
        land.load_buildings_data(buildings_data_file_name='assets/buildings.json')
        scene = scen.Scene(land=land, catalog=catalog)
        scene.calculate_land_vertices()
        scene.calculate_scaled_elevation_map()
        renderer = rend.Renderer(app_root=self, scene=scene)
        self.known_templates = json.loads(open('catalog/figures_samples.json', 'rt').read())
        # self.known_figures_parts = json.loads(open('assets/catalog_figures.json', 'rt').read())
        scene.renderer = renderer
        scene.init_scene(257, 340)
        # scene.init_scene()
        # self.create_human_hero(scene)
        scene.create_hero(
            # model_name='unorma', weapon='stone battle axe.granite',
            # model_name='unmogo', texture='goblin00',
            # model_name='unmori', texture='rick',
            # model_name='unmosu', texture='succubus02',
            # model_name='unmotr', texture='troll02',
            model_name='unhuma',
            skin=41,
            hair=0,
            wears=[
              "hadagan brigand pants.thin",
              "hadagan brigand boots.thin",
              "hadagan brigand gloves.thin",
              "hadagan brigand leggins.thick",
              "hadagan brigand helm.thick",
              "hadagan brigand plate.thick"
            ],
            weapon='cheat dagger.bronze',
            # elevation_correction=0.5,
        )
        return renderer


def main():
    res.download_res_file('data', 'figures.res', ['figures_res_0', 'figures_res_1', ])
    res.download_res_file('data', 'textures.res', ['textures_res_0', 'textures_res_1', 'textures_res_2', ])
    res.download_res_file('data', 'redress.res', ['redress_res_0', 'redress_res_1', ])
    AppRoot().run()


if __name__ == '__main__':
    main()
